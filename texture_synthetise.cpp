/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */


#include "texture_synthetise.h"
#include <acc/bvh_tree.h>
#include <mve/mesh_io_ply.h>
#include <mve/scene.h>

typedef acc::BVHTree<unsigned int, math::Vec3f> BVHTree;

TEX_NAMESPACE_BEGIN
Eigen::Vector3f getInterRGBFromImg(cv::Mat img, float u, float v)
{
    int x = std::floor(u);
    int y = std::floor(v);
    float offsetx = u - x;
    float offsety = v - y;

    float valueb = (1-offsetx)*(1-offsety)*img.at<cv::Vec3b>(y, x)[0]
            +(1-offsetx)*offsety*img.at<cv::Vec3b>(y+1, x)[0]
            +(1-offsety)*offsetx*img.at<cv::Vec3b>(y, x+1)[0]
            +offsetx*offsety*img.at<cv::Vec3b>(y + 1, x + 1)[0];

    float valueg = (1-offsetx)*(1-offsety)*img.at<cv::Vec3b>(y, x)[1]
            +(1-offsetx)*offsety*img.at<cv::Vec3b>(y+1, x)[1]
            +(1-offsety)*offsetx*img.at<cv::Vec3b>(y, x+1)[1]
            +offsetx*offsety*img.at<cv::Vec3b>(y + 1, x + 1)[1];

    float valuer = (1-offsetx)*(1-offsety)*img.at<cv::Vec3b>(y, x)[2]
            +(1-offsetx)*offsety*img.at<cv::Vec3b>(y+1, x)[2]
            +(1-offsety)*offsetx*img.at<cv::Vec3b>(y, x+1)[2]
            +offsetx*offsety*img.at<cv::Vec3b>(y + 1, x + 1)[2];

    Eigen::Vector3f color(valueb/255.0f, valueg/255.0f, valuer/255.0f);

    return color;
}

void generateDepthImageByRayCasting(mve::TriangleMesh::ConstPtr mesh, const mve::MeshInfo &mesh_info,
                                    tex::TextureView &view)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    std::vector<math::Vec3f> const & normals = mesh->get_vertex_normals();

    BVHTree bvhtree(faces, vertices);

    int level = view.level;
    float div = float(1 << level);
    int width = G2LTexConfig::get().IMAGE_WIDTH/div;
    int height = G2LTexConfig::get().IMAGE_HEIGHT/div;

    float cx = G2LTexConfig::get().IMAGE_CX/div;
    float cy = G2LTexConfig::get().IMAGE_CY/div;
    float fx = G2LTexConfig::get().IMAGE_FX/div;
    float fy = G2LTexConfig::get().IMAGE_FY/div;


    math::Vec3f origin = view.get_pos();
    math::Matrix3f c2w_rot = view.getCamToWordRotation();
    view.depthImage = cv::Mat(height, width, CV_16UC1, cv::Scalar(0));
    view.normImage = cv::Mat(height, width, CV_32FC4, cv::Scalar(0.0,0.0,0.0,0.0));
    view.remapingweigthImg = cv::Mat(height, width, CV_32FC3, cv::Scalar(0.0, 0.0, 0.0));
    view.faceIndexImg = cv::Mat(height, width, CV_32SC1, cv::Scalar(-1));

#pragma omp parallel for schedule(static,8)
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            BVHTree::Ray ray;
            ray.origin = origin;
            float p_u = x;
            float p_v = y;
            math::Vec3f v((p_u - cx)/fx, (p_v - cy)/fy, 1.0);
            ray.dir = c2w_rot.mult(v.normalized()).normalize();

            //modify
            math::Vec3f  v_dir(0,0,1);
            v_dir = c2w_rot.mult(v_dir.normalized()).normalize();

            ray.tmin = 0.0f;
            ray.tmax = std::numeric_limits<float>::infinity();
            BVHTree::Hit hit;

            if (bvhtree.intersect(ray, &hit))
            {
                math::Vec3f const & n1 = normals[faces[hit.idx * 3 + 0]];
                math::Vec3f const & n2 = normals[faces[hit.idx * 3 + 1]];
                math::Vec3f const & n3 = normals[faces[hit.idx * 3 + 2]];

                math::Vec3f const & w = hit.bcoords;
                //                Eigen::Vector3f dir_d = hit.t * ray.dir;
                //                float new_d = (hit.t * ray.dir).norm()*1000;//m-->mm

                int faceid = hit.idx;

                //modify
                float new_d = v_dir.dot(hit.t * ray.dir)*1000;
                view.depthImage.at<ushort>(y, x) = (ushort)new_d;

                math::Vec3f normal = math::interpolate(n1, n2, n3, w[0], w[1], w[2]).normalize();
                math::Vec3f vert2view = -ray.dir;
                float f_d = new_d/1000.0f;
                float weight = vert2view.dot(normal)*vert2view.dot(normal)/(f_d*f_d);

//                std::cout<<"-----weight:"<<weight<<"  d:"<<f_d<<std::endl;
                view.normImage.at<cv::Vec4f>(y, x) [0] = normal[0];
                view.normImage.at<cv::Vec4f>(y, x) [1] = normal[1];
                view.normImage.at<cv::Vec4f>(y, x) [2] = normal[2];
                view.normImage.at<cv::Vec4f>(y, x) [3] = weight;

                view.faceIndexImg.at<int>(y, x) = faceid;
                view.remapingweigthImg.at<cv::Vec3f>(y, x)[0] =  w[0];
                view.remapingweigthImg.at<cv::Vec3f>(y, x)[1] =  w[1];
                view.remapingweigthImg.at<cv::Vec3f>(y, x)[2] =  w[2];

            }
        }
    }
}

bool checkDepth(cv::Mat depth, float u, float v, float d, float threshold)
{
    int x = floor(u);
    int y = floor(v);

    int count = 0;
    float sum = 0.0;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j< 2; j++)
        {
            int px = x + i;
            int py = y + j;
            if(px < 0 || px > (depth.cols - 1) || py < 0 || py > (depth.rows - 1))
            {
                continue;
            }
            ushort p_d = depth.at<ushort>(py, px);
            if(p_d > 0)
            {
                sum += (p_d/1000.0f);
                count++;
            }
        }
    }
    //    std::cout<<"--------sum:"<<sum<<"  count:"<<count<<"  d:"<<d<<std::endl;
    if(count == 0)
    {
        sum = 0.0;
    }
    else
    {
        sum /= count;
    }

    //    std::cout<<"--------sum:"<<sum<<"  count:"<<count<<"  sumd:"<<sum<<"  d:"<<d/1000.0f<<" absD:"<<std::abs(sum - d)<<std::endl;

    bool flag = false;
    if(std::abs(sum - d) < threshold)
    {
        flag = true;
    }
    return flag;
}

Eigen::Vector3f generateColorByTwopassTextureimage(std::vector<TextureView> texture_views, int curlabel, int twopasslabel, int px, int py)
{
    Eigen::Vector3f  summodel(0.0, 0.0, 0.0);
    float cx = G2LTexConfig::get().IMAGE_CX;
    float cy = G2LTexConfig::get().IMAGE_CY;
    float fx = G2LTexConfig::get().IMAGE_FX;
    float fy = G2LTexConfig::get().IMAGE_FY;

    tex::TextureView cur_view = texture_views[curlabel - 1];
    math::Matrix4f c_mat = cur_view.getWorldToCamMatrix();
    Eigen::Matrix4f w2c = mathToEigen(c_mat);
    //    std::cout<<"--------------------1--------------"<<std::endl;

    //    std::cout<<"c label:"<<curlabel - 1 <<"-------w2c:"<<std::endl<<w2c<<std::endl;
    Eigen::Matrix4f c2w = w2c.inverse();
    ushort d = cur_view.depthImage.at<ushort>(py, px);

    if(d <= 0)
    {
        return summodel;
    }

    float z = d/1000.0f;
    Eigen::Vector4f v_p((px - cx) * z/fx, (py -cy) * z/fy, z, 1.0) ;
    v_p =  c2w*v_p;//world coordinate

    tex::TextureView ref_view = texture_views[twopasslabel - 1];
    math::Matrix4f   r_mat = ref_view.getWorldToCamMatrix();
    //    std::cout<<"r label:"<<twopasslabel - 1<<"------------r_mat:"<<std::endl<<r_mat<<std::endl;
    Eigen::Matrix4f  rw2c = mathToEigen(r_mat);
    Eigen::Vector4f  c_p = rw2c*v_p;

    float rp_x = c_p(0)*fx/c_p(2) + cx;
    float rp_y = c_p(1)*fy/c_p(2) + cy;

    if(rp_x < 0 || rp_x > (G2LTexConfig::get().IMAGE_WIDTH - 1) || rp_y < 0 || rp_y > (G2LTexConfig::get().IMAGE_HEIGHT - 1))
    {
        return summodel;
    }

    cv::Mat  depth = ref_view.depthImage;
    //    std::cout<<"------------>d:"<<depth.at<ushort>(rp_y, rp_x)<<"  d2:"<<c_p(2)<<std::endl;
    if(checkDepth(depth, rp_x, rp_y, c_p(2), 0.01) == false)
    {
        return summodel;
    }

    cv::Mat cimg = ref_view.targeImage;
    //    cv::Mat cimg = ref_view.sourceImage;

    Eigen::Vector3f color = getInterRGBFromImg(cimg, rp_x, rp_y);
    //    std::cout<<"1 color:"<<color<<std::endl;
    return color;
}

Eigen::Vector3f generateRGBByRemapping(mve::TriangleMesh::ConstPtr mesh, std::vector<std::vector<TextureView> > PyramidViews,
                                       int curlabel, int adjlabel, int px, int py, int level, int &count)
{
    count = 1;
    Eigen::Vector3f  summodel(0.0, 0.0, 0.0);
    float div = 1 << level;
    float cx = G2LTexConfig::get().IMAGE_CX/div;
    float cy = G2LTexConfig::get().IMAGE_CY/div;
    float fx = G2LTexConfig::get().IMAGE_FX/div;
    float fy = G2LTexConfig::get().IMAGE_FY/div;

    tex::TextureView cur_view = PyramidViews[curlabel - 1][level];
    math::Matrix4f c_mat = cur_view.getWorldToCamMatrix();
    Eigen::Matrix4f w2c = mathToEigen(c_mat);
    //    std::cout<<"--------------------1--------------"<<std::endl;

    //    std::cout<<"c label:"<<curlabel - 1 <<"-------w2c:"<<std::endl<<w2c<<std::endl;
    int faceID = cur_view.faceIndexImg.at<int>(py, px);//the raycasting face id of the the pixel on the surface


    if(faceID < 0)
    {
        //        std::cout<<"-------------1-------------"<<std::endl;
        count = 2;
        return summodel;
    }

    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    std::vector<math::Vec3f> const & normals = mesh->get_vertex_normals();
    //相交三角面的三个顶点
    math::Vec3f  sv1 = vertices[faces[faceID*3]];
    math::Vec3f  sv2 = vertices[faces[faceID*3 + 1]];
    math::Vec3f  sv3 = vertices[faces[faceID*3 + 2]];

    Eigen::Vector4f  v1(sv1[0], sv1[1], sv1[2], 1.0);
    Eigen::Vector4f  v2(sv2[0], sv2[1], sv2[2], 1.0);
    Eigen::Vector4f  v3(sv3[0], sv3[1], sv3[2], 1.0);

    //质心坐标权重
    float  bw1 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[0];
    float  bw2 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[1];
    float  bw3 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[2];



    tex::TextureView ref_view = PyramidViews[adjlabel - 1][level];
    math::Matrix4f   r_mat = ref_view.getWorldToCamMatrix();
    Eigen::Matrix4f  rw2c = mathToEigen(r_mat);

    Eigen::Vector4f  r_v1 = rw2c*v1;
    Eigen::Vector4f  r_v2 = rw2c*v2;
    Eigen::Vector4f  r_v3 = rw2c*v3;

    float rv1_x = r_v1(0)*fx/r_v1(2) + cx;
    float rv1_y = r_v1(1)*fy/r_v1(2) + cy;

    float rv2_x = r_v2(0)*fx/r_v2(2) + cx;
    float rv2_y = r_v2(1)*fy/r_v2(2) + cy;

    float rv3_x = r_v3(0)*fx/r_v3(2) + cx;
    float rv3_y = r_v3(1)*fy/r_v3(2) + cy;

    if(rv1_x < 0 || rv1_x > (cur_view.width - 1) || rv1_y < 0 || rv1_y > (cur_view.height - 1)
            || rv2_x < 0 || rv2_x > (cur_view.width - 1) || rv2_y < 0 || rv2_y > (cur_view.height - 1)
            || rv3_x < 0 || rv3_x > (cur_view.width - 1) || rv3_y < 0 || rv3_y > (cur_view.height - 1))
    {
        //        std::cout<<"-------------2-------------"<<std::endl;

        count = 3;
        return summodel;
    }

    //插值
//    float c_x = bw1*rv1_x + bw2*rv2_x + bw3*rv3_x;
//    float c_y = bw1*rv1_y + bw2*rv2_y + bw3*rv3_y;
    float c_x = bw1*rv1_x + bw2*rv2_x + bw3*rv3_x;
    float c_y = bw1*rv1_y + bw2*rv2_y + bw3*rv3_y;

    cv::Mat  depth = ref_view.depthImage;
    //    std::cout<<"------------>d:"<<depth.at<ushort>(rp_y, rp_x)<<"  d2:"<<c_p(2)<<std::endl;
    if(checkDepth(depth, rv1_x, rv1_y, r_v1(2), 0.02) == false
            || checkDepth(depth, rv2_x, rv2_y, r_v2(2), 0.02) == false
            || checkDepth(depth, rv3_x, rv3_y, r_v3(2), 0.02) == false)
    {
        //        std::cout<<"-------------3-------------"<<std::endl;

        count = 4;
        return summodel;
    }

    cv::Mat cimg = ref_view.targeImage;
//        cv::Mat cimg = ref_view.sourceImage;
//        cv::Mat cimg = ref_view.targeImage;


    summodel = getInterRGBFromImg(cimg, c_x, c_y);
//    summodel(0) = cimg.at<cv::Vec3b>(int(c_y+0.5), int(c_x+0.5))[0]/255.0f;
//    summodel(1) = cimg.at<cv::Vec3b>(int(c_y+0.5), int(c_x+0.5))[1]/255.0f;
//    summodel(2) = cimg.at<cv::Vec3b>(int(c_y+0.5), int(c_x+0.5))[2]/255.0f;
//    std::cout<<"1 color:"<<summodel<<std::endl;

    math::Vec3f v;
    v(0) = v1(0)*rv1_x + v2(0)*rv2_x + v2(0)*rv3_x;
    v(1) = v1(1)*rv1_x + v2(1)*rv2_x + v2(1)*rv3_x;
    v(2) = v1(2)*rv1_x + v2(2)*rv2_x + v2(2)*rv3_x;

    math::Vec3f const & n1 = normals[faces[faceID * 3 + 0]];
    math::Vec3f const & n2 = normals[faces[faceID * 3 + 1]];
    math::Vec3f const & n3 = normals[faces[faceID * 3 + 2]];

    math::Vec3f origin = ref_view.get_pos();
    math::Matrix3f c2w_rot = ref_view.getCamToWordRotation();
    math::Vec3f dir = c2w_rot.mult(v.normalized()).normalize();
    math::Vec3f vert2view = -dir;
    math::Vec3f normal = math::interpolate(n1, n2, n3, bw1, bw1, bw1).normalize();


    float weight = vert2view.dot(normal)*vert2view.dot(normal)/(v(2)*v(2));

    return summodel;
}


Eigen::Vector3f generateRGBAndWeightByRemapping(mve::TriangleMesh::ConstPtr mesh, std::vector<std::vector<TextureView> > PyramidViews,
                                       int curlabel, int adjlabel, int px, int py, int level, int &count, float &weight)
{
    weight = 1.0;
    count = 1;
    Eigen::Vector3f  summodel(0.0, 0.0, 0.0);
    float div = 1 << level;
    float cx = G2LTexConfig::get().IMAGE_CX/div;
    float cy = G2LTexConfig::get().IMAGE_CY/div;
    float fx = G2LTexConfig::get().IMAGE_FX/div;
    float fy = G2LTexConfig::get().IMAGE_FY/div;

    tex::TextureView cur_view = PyramidViews[curlabel - 1][level];
    math::Matrix4f c_mat = cur_view.getWorldToCamMatrix();
    Eigen::Matrix4f w2c = mathToEigen(c_mat);
    //    std::cout<<"--------------------1--------------"<<std::endl;

    //    std::cout<<"c label:"<<curlabel - 1 <<"-------w2c:"<<std::endl<<w2c<<std::endl;
    int faceID = cur_view.faceIndexImg.at<int>(py, px);//the raycasting face id of the the pixel on the surface


    if(faceID < 0)
    {
        //        std::cout<<"-------------1-------------"<<std::endl;
        count = 2;
        return summodel;
    }

    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    std::vector<math::Vec3f> const & normals = mesh->get_vertex_normals();
    //相交三角面的三个顶点
    math::Vec3f  sv1 = vertices[faces[faceID*3]];
    math::Vec3f  sv2 = vertices[faces[faceID*3 + 1]];
    math::Vec3f  sv3 = vertices[faces[faceID*3 + 2]];

    Eigen::Vector4f  v1(sv1[0], sv1[1], sv1[2], 1.0);
    Eigen::Vector4f  v2(sv2[0], sv2[1], sv2[2], 1.0);
    Eigen::Vector4f  v3(sv3[0], sv3[1], sv3[2], 1.0);

    //质心坐标权重
    float  bw1 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[0];
    float  bw2 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[1];
    float  bw3 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[2];



    tex::TextureView ref_view = PyramidViews[adjlabel - 1][level];
    math::Matrix4f   r_mat = ref_view.getWorldToCamMatrix();
    Eigen::Matrix4f  rw2c = mathToEigen(r_mat);

    Eigen::Vector4f  r_v1 = rw2c*v1;
    Eigen::Vector4f  r_v2 = rw2c*v2;
    Eigen::Vector4f  r_v3 = rw2c*v3;

    float rv1_x = r_v1(0)*fx/r_v1(2) + cx;
    float rv1_y = r_v1(1)*fy/r_v1(2) + cy;

    float rv2_x = r_v2(0)*fx/r_v2(2) + cx;
    float rv2_y = r_v2(1)*fy/r_v2(2) + cy;

    float rv3_x = r_v3(0)*fx/r_v3(2) + cx;
    float rv3_y = r_v3(1)*fy/r_v3(2) + cy;

    if(rv1_x < 0 || rv1_x > (cur_view.width - 1) || rv1_y < 0 || rv1_y > (cur_view.height - 1)
            || rv2_x < 0 || rv2_x > (cur_view.width - 1) || rv2_y < 0 || rv2_y > (cur_view.height - 1)
            || rv3_x < 0 || rv3_x > (cur_view.width - 1) || rv3_y < 0 || rv3_y > (cur_view.height - 1))
    {
        //        std::cout<<"-------------2-------------"<<std::endl;

        count = 3;
        return summodel;
    }

    //插值
//    float c_x = bw1*rv1_x + bw2*rv2_x + bw3*rv3_x;
//    float c_y = bw1*rv1_y + bw2*rv2_y + bw3*rv3_y;
    float c_x = bw1*rv1_x + bw2*rv2_x + bw3*rv3_x;
    float c_y = bw1*rv1_y + bw2*rv2_y + bw3*rv3_y;

    cv::Mat  depth = ref_view.depthImage;
    //    std::cout<<"------------>d:"<<depth.at<ushort>(rp_y, rp_x)<<"  d2:"<<c_p(2)<<std::endl;
    if(checkDepth(depth, rv1_x, rv1_y, r_v1(2), 0.02) == false
            || checkDepth(depth, rv2_x, rv2_y, r_v2(2), 0.02) == false
            || checkDepth(depth, rv3_x, rv3_y, r_v3(2), 0.02) == false)
    {
        //        std::cout<<"-------------3-------------"<<std::endl;

        count = 4;
        return summodel;
    }

    cv::Mat cimg = ref_view.targeImage;
//        cv::Mat cimg = ref_view.sourceImage;
//        cv::Mat cimg = ref_view.targeImage;


    summodel = getInterRGBFromImg(cimg, c_x, c_y);
//    summodel(0) = cimg.at<cv::Vec3b>(int(c_y+0.5), int(c_x+0.5))[0]/255.0f;
//    summodel(1) = cimg.at<cv::Vec3b>(int(c_y+0.5), int(c_x+0.5))[1]/255.0f;
//    summodel(2) = cimg.at<cv::Vec3b>(int(c_y+0.5), int(c_x+0.5))[2]/255.0f;
//    std::cout<<"1 color:"<<summodel<<std::endl;

    math::Vec3f v;
    v(0) = v1(0)*bw1 + v2(0)*bw2 + v3(0)*bw3;
    v(1) = v1(1)*bw1 + v2(1)*bw2 + v3(1)*bw3;
    v(2) = v1(2)*bw1 + v2(2)*bw2 + v3(2)*bw3;

    math::Vec3f const & n1 = normals[faces[faceID * 3 + 0]];
    math::Vec3f const & n2 = normals[faces[faceID * 3 + 1]];
    math::Vec3f const & n3 = normals[faces[faceID * 3 + 2]];
    math::Vec3f normal = math::interpolate(n1, n2, n3, bw1, bw2, bw3).normalize();

    math::Vec3f origin = ref_view.get_pos();
    math::Matrix3f c2w_rot = ref_view.getCamToWordRotation();
    math::Vec3f view_to_face_vec = (v - origin).normalized();
    math::Vec3f face_to_view_vec = (origin - v).normalized();

    math::Vec3f  v_dir(0,0,1);
    v_dir = c2w_rot.mult(v_dir.normalized()).normalize();
    float new_d = v_dir.dot(origin - v);

//    weight = face_to_view_vec.dot(normal)*face_to_view_vec.dot(normal)/(v(2)*v(2));
    weight = face_to_view_vec.dot(normal)*face_to_view_vec.dot(normal)/(new_d*new_d);

//    std::cout<<"-----weigh:"<<weight<<"  d:"<<v(2)<<std::endl;
    return summodel;
}

Eigen::Vector3f generateRGBByRemappingFortesting(mve::TriangleMesh::ConstPtr mesh, std::vector<std::vector<TextureView> > PyramidViews,
                                                 int curlabel, int adjlabel, int px, int py, int level, int &count)
{
    count = 1;
    Eigen::Vector3f  summodel(0.0, 0.0, 0.0);
    float div = 1 << level;
//    float cx = G2LTexConfig::get().IMAGE_CX/div;
//    float cy = G2LTexConfig::get().IMAGE_CY/div;
//    float fx = G2LTexConfig::get().IMAGE_FX/div;
//    float fy = G2LTexConfig::get().IMAGE_FY/div;

    float cx = G2LTexConfig::get().IMAGE_CX;
    float cy = G2LTexConfig::get().IMAGE_CY;
    float fx = G2LTexConfig::get().IMAGE_FX;
    float fy = G2LTexConfig::get().IMAGE_FY;

    tex::TextureView cur_view = PyramidViews[curlabel - 1][level];
    math::Matrix4f c_mat = cur_view.getWorldToCamMatrix();
    Eigen::Matrix4f w2c = mathToEigen(c_mat);
    //    std::cout<<"--------------------1--------------"<<std::endl;

    //    std::cout<<"c label:"<<curlabel - 1 <<"-------w2c:"<<std::endl<<w2c<<std::endl;
    int faceID = cur_view.faceIndexImg.at<int>(py, px);//the raycasting face id of the the pixel on the surface


    if(faceID < 0)
    {
        //        std::cout<<"-------------1-------------"<<std::endl;
        count = 2;
        return summodel;
    }

    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    //相交三角面的三个顶点
    math::Vec3f  sv1 = vertices[faces[faceID*3 + 0]];
    math::Vec3f  sv2 = vertices[faces[faceID*3 + 1]];
    math::Vec3f  sv3 = vertices[faces[faceID*3 + 2]];

    Eigen::Vector4f  v1(sv1[0], sv1[1], sv1[2], 1.0);
    Eigen::Vector4f  v2(sv2[0], sv2[1], sv2[2], 1.0);
    Eigen::Vector4f  v3(sv3[0], sv3[1], sv3[2], 1.0);

    //质心坐标权重
    float  bw1 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[0];
    float  bw2 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[1];
    float  bw3 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[2];

    tex::TextureView ref_view = PyramidViews[adjlabel - 1][0];
    math::Matrix4f   r_mat = ref_view.getWorldToCamMatrix();
    Eigen::Matrix4f  rw2c = mathToEigen(r_mat);

//    Eigen::Vector4f  center = (bw1*v1 + bw2*v2 + bw3*v3)/(bw1+bw2+bw3);

    Eigen::Vector4f  center = (bw1*v1 + bw2*v2 + bw3*v3);
    Eigen::Vector4f  r_v1 = rw2c*center;
//    Eigen::Vector4f  r_v2 = rw2c*v2;
//    Eigen::Vector4f  r_v3 = rw2c*v3;

    float rv1_x = r_v1(0)*fx/r_v1(2) + cx;
    float rv1_y = r_v1(1)*fy/r_v1(2) + cy;

//    float rv2_x = r_v2(0)*fx/r_v2(2) + cx;
//    float rv2_y = r_v2(1)*fy/r_v2(2) + cy;

//    float rv3_x = r_v3(0)*fx/r_v3(2) + cx;
//    float rv3_y = r_v3(1)*fy/r_v3(2) + cy;

    if(rv1_x < 0 || rv1_x > (ref_view.width - 1) || rv1_y < 0 || rv1_y > (ref_view.height - 1))
    {
        //        std::cout<<"-------------2-------------"<<std::endl;

        count = 3;
        return summodel;
    }

    //插值
//    float c_x = bw1*rv1_x + bw2*rv2_x + bw3*rv3_x;
//    float c_y = bw1*rv1_y + bw2*rv2_y + bw3*rv3_y;
//    float c_x = bw1*rv1_x + bw2*rv2_x + bw3*rv3_x;
//    float c_y = bw1*rv1_y + bw2*rv2_y + bw3*rv3_y;

    cv::Mat  depth = ref_view.depthImage;
    //    std::cout<<"------------>d:"<<depth.at<ushort>(rp_y, rp_x)<<"  d2:"<<c_p(2)<<std::endl;
    if(checkDepth(depth, rv1_x, rv1_y, r_v1(2), 0.02) == false)
    {
        //        std::cout<<"-------------3-------------"<<std::endl;

        count = 4;
        return summodel;
    }

    cv::Mat cimg = ref_view.targeImage;
//        cv::Mat cimg = ref_view.sourceImage;

    summodel = getInterRGBFromImg(cimg, rv1_x, rv1_y);
    //    std::cout<<"1 color:"<<summodel<<std::endl;
    return summodel;
}



Eigen::Vector3f generateRGBByPrespective(mve::TriangleMesh::ConstPtr mesh,
                                         std::vector<std::vector<TextureView> > PyramidViews,
                                         int curlabel, int adjlabel, int px, int py,
                                         int level, int &count)
{
    count = 1;
    Eigen::Vector3f  summodel(0.0, 0.0, 0.0);

    float div = 1 << level;
    float cx = G2LTexConfig::get().IMAGE_CX/div;
    float cy = G2LTexConfig::get().IMAGE_CY/div;
    float fx = G2LTexConfig::get().IMAGE_FX/div;
    float fy = G2LTexConfig::get().IMAGE_FY/div;

    tex::TextureView cur_view = PyramidViews[curlabel - 1][level];
    math::Matrix4f c_mat = cur_view.getWorldToCamMatrix();
    Eigen::Matrix4f w2c = mathToEigen(c_mat);//current frame to world;
    Eigen::Matrix4f c2w = w2c.inverse();

    tex::TextureView ref_view = PyramidViews[adjlabel - 1][level];
    math::Matrix4f   r_mat = ref_view.getWorldToCamMatrix();
    Eigen::Matrix4f  rw2c = mathToEigen(r_mat);//world to ref frame;

    cv::Mat  cdepth = cur_view.depthImage;
    ushort d = cdepth.at<ushort>(py, px);
    if(d <= 0)
    {
        count = 0;
        return summodel;
    }
    float cv_z = d/1000.0f;
    float cv_x = (px - cx)*cv_z/fx;
    float cv_y = (py - cy)*cv_z/fy;
    Eigen::Vector4f  v(cv_x, cv_y, cv_z, 1);

    v = rw2c*c2w*v;//transform to ref frame;
    float rv_x = v(0)*fx/v(2) + cx;
    float rv_y = v(1)*fy/v(2) + cy;
    if(rv_x < 0 || rv_x > (ref_view.width - 1) ||
            rv_y < 0 || rv_y > (ref_view.height -1))
    {
        count = 0;
        return summodel;
    }
    cv::Mat  depth = ref_view.depthImage;

    if(checkDepth(depth, rv_x, rv_y, v(2), 0.02) == false)
    {
        count = 0;
        return summodel;
    }
    cv::Mat cimg = ref_view.targeImage;
//    cv::Mat cimg = ref_view.sourceImage;


    summodel = getInterRGBFromImg(cimg, rv_x, rv_y);
    return summodel;
}


Eigen::Vector3f generateRGBByRemappingByViews(mve::TriangleMesh::ConstPtr mesh, std::vector<TextureView> texture_views,
                                              int curlabel, int twopasslabel, int px, int py, int &count )
{
    count = 1;
    Eigen::Vector3f  summodel(0.0, 0.0, 0.0);
    float cx = G2LTexConfig::get().IMAGE_CX;
    float cy = G2LTexConfig::get().IMAGE_CY;
    float fx = G2LTexConfig::get().IMAGE_FX;
    float fy = G2LTexConfig::get().IMAGE_FY;

    tex::TextureView cur_view = texture_views[curlabel - 1];
    math::Matrix4f c_mat = cur_view.getWorldToCamMatrix();
    Eigen::Matrix4f w2c = mathToEigen(c_mat);
    //    std::cout<<"--------------------1--------------"<<std::endl;

    //    std::cout<<"c label:"<<curlabel - 1 <<"-------w2c:"<<std::endl<<w2c<<std::endl;
    int faceID = cur_view.faceIndexImg.at<int>(py, px);//the raycasting face id of the the pixel on the surface


    if(faceID < 0)
    {
        count = 0;
        return summodel;
    }

    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    //相交三角面的三个顶点
    math::Vec3f  sv1 = vertices[faces[faceID*3]];
    math::Vec3f  sv2 = vertices[faces[faceID*3 + 1]];
    math::Vec3f  sv3 = vertices[faces[faceID*3 + 2]];

    Eigen::Vector4f  v1(sv1[0], sv1[1], sv1[2], 1.0);
    Eigen::Vector4f  v2(sv2[0], sv2[1], sv2[2], 1.0);
    Eigen::Vector4f  v3(sv3[0], sv3[1], sv3[2], 1.0);

    //质心坐标权重
    float  bw1 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[0];
    float  bw2 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[1];
    float  bw3 = cur_view.remapingweigthImg.at<cv::Vec3f>(py, px)[2];

    tex::TextureView ref_view = texture_views[twopasslabel - 1];
    math::Matrix4f   r_mat = ref_view.getWorldToCamMatrix();
    Eigen::Matrix4f  rw2c = mathToEigen(r_mat);

    Eigen::Vector4f  r_v1 = rw2c*v1;
    Eigen::Vector4f  r_v2 = rw2c*v2;
    Eigen::Vector4f  r_v3 = rw2c*v3;

    float rv1_x = r_v1(0)*fx/r_v1(2) + cx;
    float rv1_y = r_v1(1)*fy/r_v1(2) + cy;

    float rv2_x = r_v2(0)*fx/r_v2(2) + cx;
    float rv2_y = r_v2(1)*fy/r_v2(2) + cy;

    float rv3_x = r_v3(0)*fx/r_v3(2) + cx;
    float rv3_y = r_v3(1)*fy/r_v3(2) + cy;

    if(rv1_x < 0 || rv1_x > (cur_view.width - 1) || rv1_y < 0 || rv1_y > (cur_view.height - 1)
            || rv2_x < 0 || rv2_x > (cur_view.width - 1) || rv2_y < 0 || rv2_y > (cur_view.height - 1)
            || rv3_x < 0 || rv3_x > (cur_view.width - 1) || rv3_y < 0 || rv3_y > (cur_view.height - 1))
    {
        count = 0;
        return summodel;
    }

    //插值
    float c_x = bw1*rv1_x + bw2*rv2_x + bw3*rv3_x;
    float c_y = bw1*rv1_y + bw2*rv2_y + bw3*rv3_y;

    cv::Mat  depth = ref_view.depthImage;
    //    std::cout<<"------------>d:"<<depth.at<ushort>(rp_y, rp_x)<<"  d2:"<<c_p(2)<<std::endl;
    if(checkDepth(depth, rv1_x, rv1_y, r_v1(2), 0.02) == false
            || checkDepth(depth, rv2_x, rv2_y, r_v2(2), 0.02) == false
            || checkDepth(depth, rv3_x, rv3_y, r_v3(2), 0.02) == false)
    {
        count = 0;
        return summodel;
    }

    cv::Mat cimg = ref_view.targeImage;
    //    cv::Mat cimg = ref_view.sourceImage;

    summodel = getInterRGBFromImg(cimg, c_x, c_y);
    //    std::cout<<"1 color:"<<color<<std::endl;
    return summodel;
}


TEX_NAMESPACE_END
