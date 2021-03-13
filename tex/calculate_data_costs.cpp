/*
 * Copyright (C) 2015, Nils Moehrle, Michael Waechter
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <numeric>

#include <mve/image_color.h>
#include <acc/bvh_tree.h>
#include <Eigen/Core>
#include <Eigen/LU>

#include "util.h"
#include "histogram.h"
#include "texturing.h"
#include "sparse_table.h"
#include "progress_counter.h"

typedef acc::BVHTree<unsigned int, math::Vec3f> BVHTree;

TEX_NAMESPACE_BEGIN

/**
 * Dampens the quality of all views in which the face's projection
 * has a much different color than in the majority of views.
 * Returns whether the outlier removal was successfull.
 *
 * @param infos contains information about one face seen from several views
 * @param settings runtime configuration.
 */
bool photometric_outlier_detection(std::vector<FaceProjectionInfo> * infos, Settings const & settings)
{
    if (infos->size() == 0)
        return true;

    /* Configuration variables. */

    double const gauss_rejection_threshold = 6e-3;

    /* If all covariances drop below this we stop outlier detection. */
    double const minimal_covariance = 5e-4;

    int const outlier_detection_iterations = 10;
    int const minimal_num_inliers = 4;

    float outlier_removal_factor = std::numeric_limits<float>::signaling_NaN();
    switch (settings.outlier_removal)
    {
    case OUTLIER_REMOVAL_NONE: return true;
    case OUTLIER_REMOVAL_GAUSS_CLAMPING:
        outlier_removal_factor = 1.0f;
        break;
    case OUTLIER_REMOVAL_GAUSS_DAMPING:
        outlier_removal_factor = 0.2f;
        break;
    }

    Eigen::MatrixX3d inliers(infos->size(), 3);
    std::vector<std::uint32_t> is_inlier(infos->size(), 1);
    for (std::size_t row = 0; row < infos->size(); ++row)
    {
        inliers.row(row) = mve_to_eigen(infos->at(row).mean_color).cast<double>();
    }

    Eigen::RowVector3d var_mean;
    Eigen::Matrix3d covariance;
    Eigen::Matrix3d covariance_inv;

    for (int i = 0; i < outlier_detection_iterations; ++i)
    {

        if (inliers.rows() < minimal_num_inliers)
        {
            return false;
        }

        /* Calculate the inliers' mean color and color covariance. */
        var_mean = inliers.colwise().mean();
        Eigen::MatrixX3d centered = inliers.rowwise() - var_mean;
        covariance = (centered.adjoint() * centered) / double(inliers.rows() - 1);

        /* If all covariances are very small we stop outlier detection
         * and only keep the inliers (set quality of outliers to zero). */
        if (covariance.array().abs().maxCoeff() < minimal_covariance)
        {
            for (std::size_t row = 0; row < infos->size(); ++row)
            {
                if (!is_inlier[row])
                    infos->at(row).quality = 0.0f;
            }
            return true;
        }

        /* Invert the covariance. FullPivLU is not the fastest way but
         * it gives feedback about numerical stability during inversion. */
        Eigen::FullPivLU<Eigen::Matrix3d> lu(covariance);
        if (!lu.isInvertible()) {
            return false;
        }
        covariance_inv = lu.inverse();

        /* Compute new number of inliers (all views with a gauss value above a threshold). */
        for (std::size_t row = 0; row < infos->size(); ++row) {
            Eigen::RowVector3d color = mve_to_eigen(infos->at(row).mean_color).cast<double>();
            double gauss_value = multi_gauss_unnormalized(color, var_mean, covariance_inv);
            is_inlier[row] = (gauss_value >= gauss_rejection_threshold ? 1 : 0);
        }
        /* Resize Eigen matrix accordingly and fill with new inliers. */
        inliers.resize(std::accumulate(is_inlier.begin(), is_inlier.end(), 0), Eigen::NoChange);
        for (std::size_t row = 0, inlier_row = 0; row < infos->size(); ++row) {
            if (is_inlier[row]) {
                inliers.row(inlier_row++) = mve_to_eigen(infos->at(row).mean_color).cast<double>();
            }
        }
    }

    covariance_inv *= outlier_removal_factor;
    for (FaceProjectionInfo & info : *infos) {
        Eigen::RowVector3d color = mve_to_eigen(info.mean_color).cast<double>();
        double gauss_value = multi_gauss_unnormalized(color, var_mean, covariance_inv);
        assert(0.0 <= gauss_value && gauss_value <= 1.0);
        switch(settings.outlier_removal) {
        case OUTLIER_REMOVAL_NONE: return true;
        case OUTLIER_REMOVAL_GAUSS_DAMPING:
            info.quality *= gauss_value;
            break;
        case OUTLIER_REMOVAL_GAUSS_CLAMPING:
            if (gauss_value < gauss_rejection_threshold) info.quality = 0.0f;
            break;
        }
    }
    return true;
}

/**
 * @brief calculate_face_projection_infos  计算网格上每个面到每个彩色图像上的投影信息（面积，梯度和，颜色均值等信息）
 * @param mesh
 * @param texture_views
 * @param settings
 * @param face_projection_infos
 */
void calculate_face_projection_infos(mve::TriangleMesh::ConstPtr mesh,
                                     std::vector<TextureView> * texture_views, Settings const & settings,
                                     FaceProjectionInfos * face_projection_infos)
{

    std::vector<unsigned int> const & faces = mesh->get_faces();//所有面的的顶点索引
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();//所有的顶点
    mve::TriangleMesh::NormalList const & face_normals = mesh->get_face_normals();//所有面的法线

    std::size_t const num_views = texture_views->size();

    util::WallTimer timer;
    std::cout << "\tBuilding BVH from " << faces.size() / 3 << " faces... " << std::flush;
    BVHTree bvh_tree(faces, vertices);
    std::cout << "done. (Took: " << timer.get_elapsed() << " ms)" << std::endl;

    ProgressCounter view_counter("\tCalculating face qualities", num_views);
#pragma omp parallel
    {
        ///*彩色图像索引*/    /*面索引*/  /**面投影到当前彩色图像的信息/
        std::vector<std::pair<std::size_t, FaceProjectionInfo> > projected_face_view_infos;//每个网格投影到所有彩色图像上的信息

#pragma omp for schedule(dynamic)
        for (std::uint16_t j = 0; j < static_cast<std::uint16_t>(num_views); ++j)
        {
            view_counter.progress<SIMPLE>();

            TextureView * texture_view = &texture_views->at(j);//取出彩色图像相关信息
            texture_view->load_image();//取出彩色图
            texture_view->generate_validity_mask();

            if (settings.data_term == DATA_TERM_GMI)
            {
                texture_view->generate_gradient_magnitude();
                texture_view->erode_validity_mask();
            }

            math::Vec3f const & view_pos = texture_view->get_pos();
            math::Vec3f const & viewing_direction = texture_view->get_viewing_direction();

            for (std::size_t i = 0; i < faces.size(); i += 3) //所有的面对应的顶点索引
            {
                std::size_t face_id = i / 3;//面的索引

                //面的三个顶点
                math::Vec3f const & v1 = vertices[faces[i]];
                math::Vec3f const & v2 = vertices[faces[i + 1]];
                math::Vec3f const & v3 = vertices[faces[i + 2]];

                math::Vec3f const & face_normal = face_normals[face_id];
                math::Vec3f const face_center = (v1 + v2 + v3) / 3.0f;

                /* Check visibility and compute quality */
                math::Vec3f view_to_face_vec = (face_center - view_pos).normalized();
                math::Vec3f face_to_view_vec = (view_pos - face_center).normalized();

                /* Backface and basic frustum culling */
                float viewing_angle = face_to_view_vec.dot(face_normal);
                if (viewing_angle < 0.0f || viewing_direction.dot(view_to_face_vec) < 0.0f)//判断是否时背面朝向相机
                {
                    continue;
                }

                if (std::acos(viewing_angle) > MATH_DEG2RAD(75.0f))//扫描的角度太大
                {
                    continue;
                }

                /* Projects into the valid part of the TextureView? */
                if (!texture_view->inside(v1, v2, v3))//面的所有顶点是否都头因到彩色图像范围内部
                {
                    continue;
                }

                if (settings.geometric_visibility_test) //测试几何是否可见
                {
                    /* Viewing rays do not collide? */
                    bool visible = true;
                    math::Vec3f const * samples[] = {&v1, &v2, &v3};
                    // TODO: random monte carlo samples...

                    for (std::size_t k = 0; k < sizeof(samples) / sizeof(samples[0]); ++k)
                    {
                        BVHTree::Ray ray;
                        ray.origin = *samples[k];
                        ray.dir = view_pos - ray.origin;
                        ray.tmax = ray.dir.norm();
                        ray.tmin = ray.tmax * 0.0001f;
                        ray.dir.normalize();

                        BVHTree::Hit hit;
                        if (bvh_tree.intersect(ray, &hit))
                        {
                            visible = false;
                            break;
                        }
                    }
                    if (!visible) continue;
                }

                FaceProjectionInfo     info = {j, 0.0f, 0.0f, math::Vec3f(0.0f, 0.0f, 0.0f), 0.0f};

                /* Calculate quality. */
                texture_view->get_face_info(v1, v2, v3, &info, settings);//计算面在彩色图像上投影的相关属性

                if (info.quality == 0.0)
                    continue;

                /* Change color space. */
                mve::image::color_rgb_to_ycbcr(*(info.mean_color));

                std::pair<std::size_t, FaceProjectionInfo> pair(face_id, info);
                projected_face_view_infos.push_back(pair);
            }

            texture_view->release_image();
            texture_view->release_validity_mask();
            if (settings.data_term == DATA_TERM_GMI) {
                texture_view->release_gradient_magnitude();
            }
            view_counter.inc();
        }

        //std::sort(projected_face_view_infos.begin(), projected_face_view_infos.end());

#pragma omp critical
        {
            for (std::size_t i = projected_face_view_infos.size(); 0 < i; --i)
            {
                std::size_t face_id = projected_face_view_infos[i - 1].first;
                FaceProjectionInfo const & info = projected_face_view_infos[i - 1].second;
                face_projection_infos->at(face_id).push_back(info);
            }
            projected_face_view_infos.clear();
        }
    }
}

/**
 * @brief calculate_face_projection_infos_by_normal
 * @param mesh
 * @param texture_views
 * @param settings
 * @param face_projection_infos
 */
void calculate_face_projection_infos_by_normal(mve::TriangleMesh::ConstPtr mesh,
                                     std::vector<TextureView> * texture_views, Settings const & settings,
                                     FaceProjectionInfos * face_projection_infos)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();//所有面的的顶点索引
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();//所有的顶点
    mve::TriangleMesh::NormalList const & face_normals = mesh->get_face_normals();//所有面的法线

    std::size_t const num_views = texture_views->size();

    util::WallTimer timer;
    std::cout << "\tBuilding BVH from " << faces.size() / 3 << " faces... " << std::flush;
    BVHTree bvh_tree(faces, vertices);
    std::cout << "done. (Took: " << timer.get_elapsed() << " ms)" << std::endl;

#pragma omp parallel
    {
        std::vector<std::pair<std::size_t, FaceProjectionInfo> > projected_face_view_infos;//每个网格投影到所有彩色图像上的信息

#pragma omp for schedule(dynamic)
        for (std::uint16_t j = 0; j < static_cast<std::uint16_t>(num_views); ++j)
        {
            TextureView * texture_view = &texture_views->at(j);//取出彩色图像相关信息
            texture_view->load_image();//取出彩色图
            texture_view->generate_validity_mask();

            //need gradient
            texture_view->generate_gradient_magnitude();
            texture_view->erode_validity_mask();

            math::Vec3f const & view_pos = texture_view->get_pos();
            math::Vec3f const & viewing_direction = texture_view->get_viewing_direction();

            for (std::size_t i = 0; i < faces.size(); i += 3) //所有的面对应的顶点索引
            {
                std::size_t face_id = i / 3;//面的索引

                //面的三个顶点
                math::Vec3f const & v1 = vertices[faces[i]];
                math::Vec3f const & v2 = vertices[faces[i + 1]];
                math::Vec3f const & v3 = vertices[faces[i + 2]];

                math::Vec3f const & face_normal = face_normals[face_id];
                math::Vec3f const face_center = (v1 + v2 + v3) / 3.0f;

                /* Check visibility and compute quality */
                math::Vec3f view_to_face_vec = (face_center - view_pos).normalized();
                math::Vec3f face_to_view_vec = (view_pos - face_center).normalized();

                /* Backface and basic frustum culling */
                float     viewing_angle = face_to_view_vec.dot(face_normal);
                if (viewing_angle < 0.0f || viewing_direction.dot(view_to_face_vec) < 0.0f)//判断是否时背面朝向相机
                {
                    continue;
                }

                if (std::acos(viewing_angle) > MATH_DEG2RAD(75.0f))//扫描的角度太大
                {
                    continue;
                }

                /* Projects into the valid part of the TextureView? */
                if (!texture_view->inside(v1, v2, v3))//面的所有顶点是否都头因到彩色图像范围内部
                {
                    continue;
                }

                if (settings.geometric_visibility_test) //测试几何是否可见
                {
                    /* Viewing rays do not collide? */
                    bool visible = true;
                    math::Vec3f const * samples[] = {&v1, &v2, &v3};
                    // TODO: random monte carlo samples...
                    for (std::size_t k = 0; k < sizeof(samples) / sizeof(samples[0]); ++k)
                    {
                        BVHTree::Ray ray;
                        ray.origin = *samples[k];
                        ray.dir = view_pos - ray.origin;
                        ray.tmax = ray.dir.norm();
                        ray.tmin = ray.tmax * 0.0001f;
                        ray.dir.normalize();

                        BVHTree::Hit hit;
                        if (bvh_tree.intersect(ray, &hit))
                        {
                            visible = false;
                            break;
                        }
                    }
                    if (!visible) // 不可见处理下一个
                    {
                        continue;
                    }
                }

                FaceProjectionInfo     info = {j, 0.0f, 0.0f, math::Vec3f(0.0f, 0.0f, 0.0f), 0.0f};
                /* Calculate quality. */
                texture_view->get_face_info(v1, v2, v3, &info, settings);//计算面在彩色图像上投影的相关属性

//                std::cout<<"-------------------->info.mean_color:"<<info.mean_color<<std::endl;
                //modify
                info.normviewangl = 1 - viewing_angle*viewing_angle;

                /* Change color space. */
//                mve::image::color_rgb_to_ycbcr(*(info.mean_color));

                std::pair<std::size_t, FaceProjectionInfo> pair(face_id, info);
                projected_face_view_infos.push_back(pair);
            }

            texture_view->release_image();
            texture_view->release_validity_mask();
            if (settings.data_term == DATA_TERM_GMI)
            {
                texture_view->release_gradient_magnitude();
            }
        }

        //using normal angle to evaluation the label for each face.
#pragma omp critical
        {
            for (std::size_t i = projected_face_view_infos.size(); 0 < i; --i)
            {
                std::size_t   face_id = projected_face_view_infos[i - 1].first;
                FaceProjectionInfo const & info = projected_face_view_infos[i - 1].second;
//                std::cout<<"--------info.normviewangl:"<<info.normviewangl<<"------------->"<<info.quality <<std::endl;
                face_projection_infos->at(face_id).push_back(info);
            }

            projected_face_view_infos.clear();
        }
    }
}
/**
 * @brief calculate_face_projection_infos_by_feature
 * @param mesh
 * @param texture_views
 * @param settings
 * @param face_projection_infos
 */
void calculate_face_projection_infos_by_feature(mve::TriangleMesh::ConstPtr mesh,
                                     std::vector<TextureView> * texture_views, Settings const & settings,
                                     FaceProjectionInfos * face_projection_infos)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();//所有面的的顶点索引
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();//所有的顶点
    mve::TriangleMesh::NormalList const & face_normals = mesh->get_face_normals();//所有面的法线

    std::size_t const num_views = texture_views->size();

    util::WallTimer timer;
    std::cout << "\tBuilding BVH from " << faces.size() / 3 << " faces... " << std::flush;
    BVHTree bvh_tree(faces, vertices);
    std::cout << "done. (Took: " << timer.get_elapsed() << " ms)" << std::endl;

#pragma omp parallel
    {
        std::vector<std::pair<std::size_t, FaceProjectionInfo> > projected_face_view_infos;//每个网格投影到所有彩色图像上的信息

#pragma omp for schedule(dynamic)
        for (std::uint16_t j = 0; j < static_cast<std::uint16_t>(num_views); ++j)
        {
            TextureView * texture_view = &texture_views->at(j);//取出彩色图像相关信息
            texture_view->load_image();//取出彩色图
            texture_view->generate_validity_mask();

            //need gradient
            texture_view->generate_gradient_magnitude();
            texture_view->erode_validity_mask();

            math::Vec3f const & view_pos = texture_view->get_pos();
            math::Vec3f const & viewing_direction = texture_view->get_viewing_direction();

            for (std::size_t i = 0; i < faces.size(); i += 3) //所有的面对应的顶点索引
            {
                std::size_t face_id = i / 3;//面的索引

                //面的三个顶点
                math::Vec3f const & v1 = vertices[faces[i]];
                math::Vec3f const & v2 = vertices[faces[i + 1]];
                math::Vec3f const & v3 = vertices[faces[i + 2]];

                math::Vec3f const & face_normal = face_normals[face_id];
                math::Vec3f const face_center = (v1 + v2 + v3) / 3.0f;

                /* Check visibility and compute quality */
                math::Vec3f view_to_face_vec = (face_center - view_pos).normalized();
                math::Vec3f face_to_view_vec = (view_pos - face_center).normalized();

                /* Backface and basic frustum culling */
                float     viewing_angle = face_to_view_vec.dot(face_normal);
                if (viewing_angle < 0.0f || viewing_direction.dot(view_to_face_vec) < 0.0f)//判断是否时背面朝向相机
                {
                    continue;
                }

                if (std::acos(viewing_angle) > MATH_DEG2RAD(75.0f))//扫描的角度太大
                {
                    continue;
                }

                /* Projects into the valid part of the TextureView? */
                if (!texture_view->inside(v1, v2, v3))//面的所有顶点是否都头因到彩色图像范围内部
                {
                    continue;
                }

                if (settings.geometric_visibility_test) //测试几何是否可见
                {
                    /* Viewing rays do not collide? */
                    bool visible = true;
                    math::Vec3f const * samples[] = {&v1, &v2, &v3};
                    // TODO: random monte carlo samples...
                    for (std::size_t k = 0; k < sizeof(samples) / sizeof(samples[0]); ++k)
                    {
                        BVHTree::Ray ray;
                        ray.origin = *samples[k];
                        ray.dir = view_pos - ray.origin;
                        ray.tmax = ray.dir.norm();
                        ray.tmin = ray.tmax * 0.0001f;
                        ray.dir.normalize();

                        BVHTree::Hit hit;
                        if (bvh_tree.intersect(ray, &hit))
                        {
                            visible = false;
                            break;
                        }
                    }
                    if (!visible) // 不可见处理下一个
                    {
                        continue;
                    }
                }

                FaceProjectionInfo     info = {j, 0.0f, 0.0f, math::Vec3f(0.0f, 0.0f, 0.0f),0.0f};
                /* Calculate quality. */
                texture_view->get_face_info(v1, v2, v3, &info, settings);//计算面在彩色图像上投影的相关属性

//                std::cout<<"-------------------->info.mean_color:"<<info.mean_color<<std::endl;
                //modify
//                info.normviewangl = 1 - viewing_angle*viewing_angle;

                /* Change color space. */
//                mve::image::color_rgb_to_ycbcr(*(info.mean_color));

                std::pair<std::size_t, FaceProjectionInfo> pair(face_id, info);
                projected_face_view_infos.push_back(pair);
            }

            texture_view->release_image();
            texture_view->release_validity_mask();
            if (settings.data_term == DATA_TERM_GMI)
            {
                texture_view->release_gradient_magnitude();
            }
        }

        //using normal angle to evaluation the label for each face.
#pragma omp critical
        {
            for (std::size_t i = projected_face_view_infos.size(); 0 < i; --i)
            {
                std::size_t   face_id = projected_face_view_infos[i - 1].first;
                FaceProjectionInfo const & info = projected_face_view_infos[i - 1].second;
//                std::cout<<"--------info.normviewangl:"<<info.normviewangl<<"------------->"<<info.quality <<std::endl;
                face_projection_infos->at(face_id).push_back(info);
            }

            projected_face_view_infos.clear();
        }
    }
}


void postprocess_face_infos(Settings const & settings,
                            FaceProjectionInfos * face_projection_infos,
                            DataCosts * data_costs)
{
    ProgressCounter face_counter("\tPostprocessing face infos", face_projection_infos->size());
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < face_projection_infos->size(); ++i)//face到所有视口的投影
    {
        face_counter.progress<SIMPLE>();

        std::vector<FaceProjectionInfo> & infos = face_projection_infos->at(i);//每个面到所有视口的投影信息
        if (settings.outlier_removal != OUTLIER_REMOVAL_NONE)
        {
            photometric_outlier_detection(&infos, settings);//去掉离群视口

            infos.erase(std::remove_if(infos.begin(), infos.end(),
                                       [](FaceProjectionInfo const & info) -> bool {return info.quality == 0.0f;}),
                    infos.end());//移除投影质量为0的点。
        }
        std::sort(infos.begin(), infos.end());//安彩色图视口索引排序

        face_counter.inc();
    }

    //所有投影中最大的投影质量
    /* Determine the function for the normlization. */
    float max_quality = 0.0f;
    for (std::size_t i = 0; i < face_projection_infos->size(); ++i)
    {
        for (FaceProjectionInfo const & info : face_projection_infos->at(i))//面的所有彩色图像上的投影信息
        {
            max_quality = std::max(max_quality, info.quality);
        }
    }


    Histogram hist_qualities(0.0f, max_quality, 10000);//所有的投影质量建立histogram
    for (std::size_t i = 0; i < face_projection_infos->size(); ++i)
    {
        for (FaceProjectionInfo const & info : face_projection_infos->at(i))//面的所有彩色图像上的投影信息
        {
            hist_qualities.add_value(info.quality);//计算直方图
        }
    }

    float percentile = hist_qualities.get_approx_percentile(0.995f);//这个返回的时0.955的value时bin所在的位置。

    /* Calculate the costs. */
    for (std::uint32_t i = 0; i < face_projection_infos->size(); ++i)
    {
        for (FaceProjectionInfo const & info : face_projection_infos->at(i))
        {

            /* Clamp to percentile and normalize. */
            float normalized_quality = std::min(1.0f, info.quality / percentile);
            float data_cost = (1.0f - normalized_quality);
            data_costs->set_value(i, info.view_id, data_cost);
        }

        /* Ensure that all memory is freeed. */
        face_projection_infos->at(i) = std::vector<FaceProjectionInfo>();
    }

    std::cout << "\tMaximum quality of a face within an image: " << max_quality << std::endl;
    std::cout << "\tClamping qualities to " << percentile << " within normalization." << std::endl;
}

void postprocess_face_infos_and_feature(Settings const & settings,
                                        FaceProjectionInfos * face_projection_infos,
                                        DataCosts * data_costs, SmoothCosts * smooth_costs, DataCosts *feature_costs)
{
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < face_projection_infos->size(); ++i)//face到所有视口的投影
    {
        std::vector<FaceProjectionInfo> & infos = face_projection_infos->at(i);//每个面到所有视口的投影信息
        if (settings.outlier_removal != OUTLIER_REMOVAL_NONE)
        {
            photometric_outlier_detection(&infos, settings);//去掉离群视口

            infos.erase(std::remove_if(infos.begin(), infos.end(),
                                       [](FaceProjectionInfo const & info) -> bool {return info.quality == 0.0f;}),
                    infos.end());//移除投影质量为0的点。
        }
        std::sort(infos.begin(), infos.end());//安彩色图视口索引排序

    }

    //所有面在所有视口投影中最大的投影质量
    /* Determine the function for the normlization. */
    float max_quality = 0.0f;
    for (std::size_t i = 0; i < face_projection_infos->size(); ++i)
    {
        for (FaceProjectionInfo const & info : face_projection_infos->at(i))//面的所有彩色图像上的投影信息
        {
            max_quality = std::max(max_quality, info.quality);
        }
    }


    Histogram hist_qualities(0.0f, max_quality, 10000);//所有的投影质量建立histogram
    for (std::size_t i = 0; i < face_projection_infos->size(); ++i)
    {
        for (FaceProjectionInfo const & info : face_projection_infos->at(i))//面的所有彩色图像上的投影信息
        {
            hist_qualities.add_value(info.quality);//计算直方图
        }
    }

    float percentile = hist_qualities.get_approx_percentile(0.995f);//这个返回的时0.955的value时bin所在的位置。

    /* Calculate the costs. */
    for (std::uint32_t i = 0; i < face_projection_infos->size(); ++i)
    {
        for (FaceProjectionInfo const & info : face_projection_infos->at(i))
        {

            /* Clamp to percentile and normalize. */
            float normalized_quality = std::min(1.0f, info.quality / percentile);
            float data_cost = (1.0f - normalized_quality);//for area

            data_costs->set_value(i, info.view_id, data_cost);
//            data_costs->set_value(i, info.view_id, info.normviewangl);//normal

//            smooth_costs->set_value(i, info.view_id, info.quality);
//            smooth_costs->set_value(i, info.view_id, normalized_quality);

            float v = info.mean_color[0]*0.299 + info.mean_color[1]*0.587+info.mean_color[2]*0.114;//color-------->gray

            smooth_costs->set_value(i, info.view_id, info.mean_color);
            feature_costs->set_value(i,info.view_id,info.detailvalue);
//            feature_costs->set_value(i,info.view_id, v);
        }

        /* Ensure that all memory is freeed. */
        face_projection_infos->at(i) = std::vector<FaceProjectionInfo>();
    }
}

void calculate_data_costs(mve::TriangleMesh::ConstPtr mesh, std::vector<TextureView> * texture_views,
                          Settings const & settings, DataCosts * data_costs)
{

    std::size_t const num_faces = mesh->get_faces().size() / 3;
    std::size_t const num_views = texture_views->size();

    if (num_faces > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Exeeded maximal number of faces");
    if (num_views > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error("Exeeded maximal number of views");

    FaceProjectionInfos face_projection_infos(num_faces);//每个面对应的每个视口的投影属性（外列表是每个面，列表里面的表格是这个面到所有彩色图的投影信息）

    calculate_face_projection_infos_by_feature(mesh, texture_views, settings, &face_projection_infos);//计算每个面到每个视口的投影信息

    postprocess_face_infos(settings, &face_projection_infos, data_costs);//对面进行处理，并计算代价方程
}

void calculate_data_costs_and_feature(mve::TriangleMesh::ConstPtr mesh,
                                      TextureViews * texture_views, Settings const & settings,
                                      DataCosts * data_costs, SmoothCosts * smooth_costs, DataCosts *feature_costs)
{
    std::size_t const num_faces = mesh->get_faces().size() / 3;
    std::size_t const num_views = texture_views->size();

    if (num_faces > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Exeeded maximal number of faces");
    if (num_views > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error("Exeeded maximal number of views");

    FaceProjectionInfos face_projection_infos(num_faces);//每个面对应的每个视口的投影属性（外列表是每个面，列表里面的表格是这个面到所有彩色图的投影信息）

    //    calculate_face_projection_infos(mesh, texture_views, settings, &face_projection_infos);//计算每个面到每个视口的投影信息

    //根据法线与视线之间的夹角来计算数据项
    calculate_face_projection_infos_by_feature(mesh, texture_views, settings, &face_projection_infos);//color, area, gradi
    //
    postprocess_face_infos_and_feature(settings, &face_projection_infos, data_costs, smooth_costs, feature_costs);//对面进行处理，并计算代价方程
}

void calculate_data_costs_and_detail(mve::TriangleMesh::ConstPtr mesh,
                                     TextureViews * texture_views, Settings const & settings,
                                     DataCosts * data_costs, SmoothCosts * smooth_costs, DataCosts *detail_costs)
{
    std::size_t const num_faces = mesh->get_faces().size() / 3;
    std::size_t const num_views = texture_views->size();

    if (num_faces > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Exeeded maximal number of faces");
    if (num_views > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error("Exeeded maximal number of views");

    FaceProjectionInfos face_projection_infos(num_faces);//每个面对应的每个视口的投影属性（外列表是每个面，列表里面的表格是这个面到所有彩色图的投影信息）

//    calculate_face_projection_infos(mesh, texture_views, settings, &face_projection_infos);//计算每个面到每个视口的投影信息

    //根据法线与视线之间的夹角来计算数据项
    calculate_face_projection_infos_by_normal(mesh, texture_views, settings, &face_projection_infos);//color, area, gradi
    //
    postprocess_face_infos_and_feature(settings, &face_projection_infos, data_costs, smooth_costs, detail_costs);//对面进行处理，并计算代价方程
}

void calculate_twopass_data_costs(mve::TriangleMesh::ConstPtr mesh,
                                  TextureViews * texture_views, Settings const & settings,
                                  DataCosts * data_costs)
{
    std::size_t const num_faces = mesh->get_faces().size() / 3;
    std::size_t const num_views = texture_views->size();

    if (num_faces > std::numeric_limits<std::uint32_t>::max())
        throw std::runtime_error("Exeeded maximal number of faces");
    if (num_views > std::numeric_limits<std::uint16_t>::max())
        throw std::runtime_error("Exeeded maximal number of views");

    FaceProjectionInfos face_projection_infos(num_faces);//每个面对应的每个视口的投影属性（外列表是每个面，列表里面的表格是这个面到所有彩色图的投影信息）

    calculate_face_projection_infos(mesh, texture_views, settings, &face_projection_infos);//计算每个面到每个视口的投影信息

    postprocess_face_infos(settings, &face_projection_infos, data_costs);//对面进行处理，并计算代价方程
}
TEX_NAMESPACE_END
