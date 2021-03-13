/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_TEXTUREVIEW_HEADER
#define TEX_TEXTUREVIEW_HEADER

#include <string>
#include <vector>

#include <math/vector.h>
#include <mve/camera.h>
#include <mve/image.h>
#include <Eigen/Eigen>

#include "tri.h"
#include "settings.h"
#include "G2LTexConfig.h"
#include "seam_leveling.h"
//#include "texture_synthetise.h"
#include "counterinfo.h"

TEX_NAMESPACE_BEGIN

/** Struct containing the quality and mean color of a face within a view. */
struct FaceProjectionInfo
{
    std::uint16_t view_id;//对应的视口索引
    float              quality;//面的质量（面积或者梯度和）
    float              normviewangl;//法线与视线的夹角（cos）
    math::Vec3f   mean_color;//投影区域内所有采样点颜色均值
    float              detailvalue;//特征图掩码


    bool operator<(FaceProjectionInfo const & other) const
    {
        return view_id < other.view_id;
    }
};

/**
  * Class representing a view with specialized functions for texturing.
  */
class TextureView {
    private:
        std::size_t id;  //视口的索引

        math::Vec3f      pos;//相机在世界坐标中的位置
        math::Vec3f      viewdir;//相机在世界坐标系下的朝向
        math::Matrix3f   depth_projection;//内参，投影矩阵
        math::Matrix3f   image_projection;//内参，投影矩阵
        math::Matrix4f   world_to_cam;//从世界坐标变换到相机坐标的变换矩阵

        mve::ByteImage::Ptr   gradient_magnitude;//图像的梯度图
        std::vector<bool>       validity_mask;

public:
        std::string                  image_file;//视口对应的图像文件名称

        int width;
        int height;
public:
        int level;//金字塔的层数
        //用来进行纹理合成或者纹理缝合
        std::vector<std::vector<VertexProjectionInfo> > seamCounters;//每个图像上所有的缝隙的轮廓线
        mve::ByteImage::Ptr    image;//视口对应的纹理图像
        cv::Mat   FeatureMap;//记录每个图像的特征图
        cv::Mat   detailMap;//记录细节层。

        std::vector<std::vector<cv::Point2f> >  contours;//对象上对应的轮廓线。用来计算合成图像时的权重


        //金字塔
        cv::Mat   sourceImage;//原图，直接从文件读取
        cv::Mat   targeImage;//目标图，从原图生成，用来进行纹理映射
        cv::Mat   referenceImage;//目标图，从原图生成，用来进行纹理映射

        cv::Mat   depthImage;//深度图，利用光线追踪获得，应为深度图和彩色图的分辨率可能不一样。
        cv::Mat   normImage;

//        cv::Mat   seamInfoImg;//记录每个像素到缝隙的距离，二次剖分的标签等
        PixelInfoMatrix* seamInfoImg;

        //for remapping
        cv::Mat  remapingweigthImg;//每个像素点对应到网格的权重。
        cv::Mat  faceIndexImg;//每个像素对应的三角面（-1表示不可见）
public:
//        mve::ByteImage::Ptr   imagemask;//缝隙周围的mask图
//        mve::ByteImage::Ptr   imagebelieve;//缝隙周围的置信图（权重图）

        cv::Mat   imagemask;//缝隙周围的mask图
        cv::Mat   imagebelieve;//缝隙周围的置信图（权重图）
        void initmaskimage();

    public:

        void generateFeaturemap(std::string filename);
        void generateDetailmap(std::string  filename);
        void myGenerateDetailImage(std::string filename, int index = 0);

        /** Returns the id of the TexureView which is consistent for every run. */
        std::size_t get_id(void) const;

        /** Returns the 2D pixel coordinates of the given vertex projected into the view. */
        math::Vec2f get_pixel_coords(math::Vec3f const & vertex) const;

        math::Vec2f get_pixel_coords_noshift(math::Vec3f const & vertex) const;
        /** Returns the RGB pixel values [0, 1] for the given vertex projected into the view, calculated by linear interpolation. */
        math::Vec3f get_pixel_values(math::Vec3f const & vertex) const;

        math::Vec2f get_depth_pixel_coords(math::Vec3f const & vertex) const;

        /** Returns whether the pixel location is valid in this view.
          * The pixel location is valid if its inside the visible area and,
          * if a validity mask has been generated, all surrounding (integer coordinate) pixels are valid in the validity mask.
          */
        bool valid_pixel(math::Vec2f pixel) const;

        /** TODO */
        bool inside(math::Vec3f const & v1, math::Vec3f const & v2, math::Vec3f const & v3) const;

        //add 2017-10-24
        inline bool insideByBorder(math::Vec3f const & v1, math::Vec3f const & v2, math::Vec3f const & v3) const;

        /** Returns the RGB pixel values [0, 1] for the give pixel location. */
        math::Vec3f get_pixel_values(math::Vec2f const & pixel) const;

        /** Constructs a TextureView from the give mve::CameraInfo containing the given image. */
        TextureView(std::size_t id, mve::CameraInfo const & camera, std::string const & image_file);

        TextureView();

        /** Returns the position. */
        math::Vec3f get_pos(void);
        /** Returns the viewing direction. */
        math::Vec3f get_viewing_direction(void) const;
        /** Returns the width of the corresponding image. */
        int get_width(void) const;
        /** Returns the height of the corresponding image. */
        int get_height(void) const;
        /** Returns a reference pointer to the corresponding image. */
        mve::ByteImage::Ptr get_image(void) const;

        /** Exchange encapsulated image. */
        void bind_image(mve::ByteImage::Ptr new_image);

        /** Loads the corresponding image. */
        void load_image(void);
        /** Generates the validity mask. */
        void generate_validity_mask(void);
        /** Generates the gradient magnitude image for the encapsulated image. */
        void generate_gradient_magnitude(void);

        /** Releases the validity mask. */
        void release_validity_mask(void);
        /** Releases the gradient magnitude image. */
        void release_gradient_magnitude(void);
        /** Releases the corresponding image. */
        void release_image(void);

        /** Erodes the validity mask by one pixel. */
        void erode_validity_mask(void);

        void get_face_info(math::Vec3f const & v1, math::Vec3f const & v2, math::Vec3f const & v3,
            FaceProjectionInfo * face_info, Settings const & settings) const;

        void export_triangle(math::Vec3f v1, math::Vec3f v2, math::Vec3f v3, std::string const & filename) const;

        void export_validity_mask(std::string const & filename) const;


        math::Matrix3f getCamToWordRotation() const;
        math::Matrix4f  getWorldToCamMatrix() const;
        void  setWorldToCamMatrix(Eigen::Matrix4f mat);
        math::Matrix3f  getDepthCameraProjection() const;
        math::Matrix3f  getImageCameraProjection() const;
};


inline std::size_t TextureView::get_id(void) const {
    return id;
}

inline math::Vec3f TextureView::get_pos(void)
{
    Eigen::Matrix4f  mat;
    mat(0,0) = world_to_cam[0]; mat(0,1) = world_to_cam[1]; mat(0,2) = world_to_cam[2]; mat(0,3) = world_to_cam[3];
    mat(1,0) = world_to_cam[4]; mat(1,1) = world_to_cam[5]; mat(1,2) = world_to_cam[6]; mat(1,3) = world_to_cam[7];
    mat(2,0) = world_to_cam[8]; mat(2,1) = world_to_cam[9]; mat(2,2) = world_to_cam[10]; mat(2,3) = world_to_cam[11];
    mat(3,0) = world_to_cam[12]; mat(3,1) = world_to_cam[13]; mat(3,2) = world_to_cam[14]; mat(3,3) = world_to_cam[15];
    mat = mat.inverse().eval();
//    pos[0] = -world_to_cam[0] * world_to_cam[3] - world_to_cam[4] * world_to_cam[7] - world_to_cam[8] * world_to_cam[11];
//    pos[1] = -world_to_cam[1] * world_to_cam[3] - world_to_cam[5] * world_to_cam[7] - world_to_cam[9] * world_to_cam[11];
//    pos[2] = -world_to_cam[2] * world_to_cam[3] - world_to_cam[6] * world_to_cam[7] - world_to_cam[10] * world_to_cam[11];

    pos[0] = mat(0,3);
    pos[1] = mat(1,3);
    pos[2] = mat(2,3);


    return pos;
}

inline math::Vec3f
TextureView::get_viewing_direction(void) const {
    return viewdir;
}

inline int
TextureView::get_width(void) const {
    return width;
}

inline int
TextureView::get_height(void) const
{
    return height;
}

inline mve::ByteImage::Ptr TextureView::get_image(void) const
{
    assert(image != NULL);
    return image;
}

/**
 * @brief TextureView::inside  三角网格的三个顶点是否都在图像内（未越界）
 * @param v1
 * @param v2
 * @param v3
 * @return
 */
inline bool TextureView::inside(math::Vec3f const & v1, math::Vec3f const & v2, math::Vec3f const & v3) const
{
    math::Vec2f p1 = get_pixel_coords_noshift(v1);
    math::Vec2f p2 = get_pixel_coords_noshift(v2);
    math::Vec2f p3 = get_pixel_coords_noshift(v3);
//    bool  fp1 = (p1(0) >= G2LTexConfig::get().BOARD_IGNORE && p1(0) <= G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE &&
//            p1(1) >= G2LTexConfig::get().BOARD_IGNORE && p1(1) <= G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE);
//    bool  fp2 = (p2(0) >= G2LTexConfig::get().BOARD_IGNORE && p2(0) <= G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE &&
//            p2(1) >= G2LTexConfig::get().BOARD_IGNORE && p2(1) <= G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE);
//    bool  fp3 = (p3(0) >= G2LTexConfig::get().BOARD_IGNORE && p3(0) <= G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE &&
//            p3(1) >= G2LTexConfig::get().BOARD_IGNORE && p3(1) <= G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE);

//    std::cout<<"p1:"<<p1(0)<<" "<<p1(1)<<" p2:"<<p2(0)<<" "<<p2(1)<<" p3:"<<p3(0)<<" "<<p3(1)<<std::endl;
//    std::cout<<"fp1:"<<fp1<<" fp2:"<<fp2<<" fp3:"<<fp3<<"  toal:"<<(fp1&& fp2&& fp3)<<std::endl;

//    return fp1 && fp2 && fp3;

    return valid_pixel(p1) && valid_pixel(p2) && valid_pixel(p3);
}

inline bool TextureView::insideByBorder(math::Vec3f const & v1, math::Vec3f const & v2, math::Vec3f const & v3) const
{
    math::Vec2f p1 = get_pixel_coords(v1);
    math::Vec2f p2 = get_pixel_coords(v2);
    math::Vec2f p3 = get_pixel_coords(v3);
    return p1(0) > G2LTexConfig::get().BOARD_IGNORE && p1(0) < (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
              p1(1) > G2LTexConfig::get().BOARD_IGNORE && p1(1) < (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) &&
              p2(0) > G2LTexConfig::get().BOARD_IGNORE && p2(0) < (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
              p2(1) > G2LTexConfig::get().BOARD_IGNORE && p2(1) < (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) &&
              p3(0) > G2LTexConfig::get().BOARD_IGNORE && p3(0) < (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
              p3(1) > G2LTexConfig::get().BOARD_IGNORE && p3(1) < (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE);
}


/**
 * @brief TextureView::get_pixel_coords  从三位点坐标变换相机坐标，然后投影到图像平面并归一化坐标
 * @param vertex
 * @return
 */
inline math::Vec2f TextureView::get_pixel_coords(math::Vec3f const & vertex) const
{
    math::Vec3f pixel = image_projection * world_to_cam.mult(vertex, 1.0f);//投影到相机平面
    pixel /= pixel[2];
    return math::Vec2f(pixel[0] - 0.5f, pixel[1] - 0.5f);
//    return math::Vec2f(pixel[0], pixel[1]);
//     return math::Vec2f(pixel[0] + 0.5f, pixel[1] + 0.5f);


}

inline math::Vec2f TextureView::get_pixel_coords_noshift(math::Vec3f const & vertex) const
{
    math::Vec3f pixel = image_projection * world_to_cam.mult(vertex, 1.0f);//投影到相机平面
    pixel /= pixel[2];
    return math::Vec2f(pixel[0], pixel[1]);
//    return math::Vec2f(pixel[0], pixel[1]);
//     return math::Vec2f(pixel[0] + 0.5f, pixel[1] + 0.5f);


}


inline math::Vec3f
TextureView::get_pixel_values(math::Vec3f const & vertex) const {
    math::Vec2f pixel = get_pixel_coords_noshift(vertex);
    return get_pixel_values(pixel);
}

inline math::Vec3f
TextureView::get_pixel_values(math::Vec2f const & pixel) const {
    assert(image != NULL);
    math::Vec3uc values;
    image->linear_at(pixel[0], pixel[1], *values);
    return math::Vec3f(values) / 255.0f;
}

inline void
TextureView::bind_image(mve::ByteImage::Ptr new_image) {
    image = new_image;
}

inline void
TextureView::release_validity_mask(void) {
    assert(validity_mask.size() == static_cast<std::size_t>(width * height));
    validity_mask = std::vector<bool>();
}

inline void
TextureView::release_gradient_magnitude(void) {
    assert(gradient_magnitude != NULL);
    gradient_magnitude.reset();
}

inline void
TextureView::release_image(void) {
    assert(image != NULL);
    image.reset();
}

TEX_NAMESPACE_END

#endif /* TEX_TEXTUREVIEW_HEADER */
