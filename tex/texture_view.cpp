/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <list>

#include <math/matrix.h>
#include <mve/image_io.h>
#include <mve/image_tools.h>

#include "texture_view.h"
#include "detail/load_EXR.h"
#include "detail/linear_bf.h"

TEX_NAMESPACE_BEGIN

typedef Image_file::EXR::image_type image_type;
typedef image_type::channel_type    channel_type;


inline double log_function(const double x){

    static const double inv_log_base = 1.0 / log(10.0);

    return log(x) * inv_log_base;
}


inline double exp_function(const double x){

    return pow(10.0,x);
}

void TextureView::myGenerateDetailImage(std::string filename, int index)
{
    cv::Mat  img = cv::imread(filename);
//    outimg = cv::Mat(img.rows, img.cols, CV_8UC4, cv::Scalar(0,0,0,0));
    detailMap = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
    //detail level
    const unsigned width  = img.cols;
    const unsigned height = img.rows;

    channel_type intensity_channel(width,height);
    channel_type log_intensity_channel(width,height);
    channel_type::iterator i = intensity_channel.begin();
    channel_type::iterator l = log_intensity_channel.begin();

    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
//            double kk = (float(img.at<cv::Vec3b>(i, j)[0])*0.3f+ float(img.at<cv::Vec3b>(i, j)[1])*0.59f + float(img.at<cv::Vec3b>(i, j)[2])*0.11);
            double kk = 1.0f + (float(img.at<cv::Vec3b>(i, j)[0])*20.0f+ float(img.at<cv::Vec3b>(i, j)[1])*40.0 + float(img.at<cv::Vec3b>(i, j)[2])) / 61.0f;

            intensity_channel.at(j, i) = kk;
            double vv= log_function(kk);//why we nedd log;
            log_intensity_channel.at(j, i) = vv;
        }
    }

    channel_type filtered_log_intensity_channel(width,height);

    FFT::Support_3D::set_fftw_flags(FFTW_ESTIMATE); // parameter for FFTW

    const double space_sigma = 0.02 * std::min(width,height);
    const double range_sigma = 0.5;

    Image_filter::linear_BF(log_intensity_channel,
                            space_sigma,
                            range_sigma,
                            &filtered_log_intensity_channel);


    channel_type detail_channel(width,height);
    cv::Mat detailimg = cv::Mat(height,width,CV_32FC1, cv::Scalar(0.0f));

    int count = 0;
    for(channel_type::iterator
        l     = log_intensity_channel.begin(),
        l_end = log_intensity_channel.end(),
        f     = filtered_log_intensity_channel.begin(),
        d     = detail_channel.begin();
        l != l_end;
        l++,f++,d++)
    {

        *d = (*l) - (*f);
        detailimg.at<float>(count%height, count/height) =  (*l) - (*f);
// detailimg.at<float>(count%width, count/width) =  (*l) - (*f);
        count++;
    }

    double src_min, src_max;
    cv::minMaxLoc(detailimg, &src_min, &src_max);
//    std::cout<<"min:"<<src_min<<" max:"<<src_max<<std::endl;
    float dist = src_max - src_min;

    for(int h = 0; h < img.rows; h++)
    {
        for(int w = 0; w < img.cols; w++)
        {
//            outimg.at<cv::Vec4b>(h, w)[0] = img.at<cv::Vec3b>(h,w)[0];
//            outimg.at<cv::Vec4b>(h, w)[1] = img.at<cv::Vec3b>(h,w)[1];
//            outimg.at<cv::Vec4b>(h, w)[2] = img.at<cv::Vec3b>(h,w)[2];
//            outimg.at<cv::Vec4b>(h, w)[3] = (detailimg.at<float>(h, w) - src_min)*255/dist;
            detailMap.at<uchar>(h, w) = (detailimg.at<float>(h, w) - src_min)*255/dist;
        }
    }

//    std::vector<cv::Mat> channels;
//    cv::Mat imageBlueChannel;
//    cv::split(outimg, channels);
//    imageBlueChannel = channels.at(3);
//    char buf[256];
//    sprintf(buf, "detail%02d.png", index);
//    cv::imwrite(buf, detailMap);
}


TextureView::TextureView(std::size_t id, mve::CameraInfo const & camera, std::string const & image_file)
    : id(id), image_file(image_file)
{

    mve::image::ImageHeaders header;
    try
    {
         header = mve::image::load_file_headers(image_file);
    }
    catch (util::Exception e)
    {
        std::cerr << "Could not load image header of " << image_file << std::endl;
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    width = header.width;
    height = header.height;


    //保存相机内参矩阵
    camera.fill_calibration(*depth_projection, *image_projection);//设置相机内参
    camera.fill_camera_pos(*pos);//设置相机在全局坐标系下的位置
    camera.fill_viewing_direction(*viewdir);//相机在全局坐标系下的朝向
    camera.fill_world_to_cam(*world_to_cam);//设置相机从世界坐标变换到相机坐标的变换矩阵
}

TextureView::TextureView()
{

}

//mve::Image::Ptr TextureView::get_gradient_image() const
//{
//    assert(gradient_magnitude != NULL);

//    return gradient_magnitude;
//}

//void TextureView::generate_validity_mask(void)
//{
//    assert(image != NULL);
//    validity_mask.resize(width * height, true);//全部是有效的
//    mve::ByteImage::Ptr checked = mve::ByteImage::create(width, height, 1);//1个通道

//    std::list<math::Vec2i> queue;

//    /* 图片的四个角. */
//    queue.push_back(math::Vec2i(0,0));
//    checked->at(0, 0, 0) = 255;
//    queue.push_back(math::Vec2i(0, height - 1));
//    checked->at(0, height - 1, 0) = 255;
//    queue.push_back(math::Vec2i(width - 1, 0));
//    checked->at(width - 1, 0, 0) = 255;
//    queue.push_back(math::Vec2i(width - 1, height - 1));
//    checked->at(width - 1, height - 1, 0) = 255;

//    while (!queue.empty())
//    {
//        math::Vec2i pixel = queue.front();
//        queue.pop_front();

//        int const x = pixel[0];
//        int const y = pixel[1];

//        int sum = 0;
//        for (int c = 0; c < image->channels(); ++c)
//        {
//            sum += image->at(x, y, c);//图像上三个通道颜色的和
//        }

//        if (sum == 0) //无效点
//        {
//            validity_mask[x + y * width] = false;
//            std::vector<math::Vec2i> neighbours;
//            neighbours.push_back(math::Vec2i(x + 1, y));
//            neighbours.push_back(math::Vec2i(x, y + 1));
//            neighbours.push_back(math::Vec2i(x - 1, y));
//            neighbours.push_back(math::Vec2i(x, y - 1));

//            for (std::size_t i = 0; i < neighbours.size(); ++i)
//            {
//                math::Vec2i npixel = neighbours[i];
//                int const nx = npixel[0];
//                int const ny = npixel[1];
//                if (0 <= nx && nx < width && 0 <= ny && ny < height)//modify by fuyp 2017-09-22
//                {
//                    if (checked->at(nx, ny, 0) == 0)//还没有被检测
//                    {
//                        queue.push_front(npixel);//压入队列等待检测
//                        checked->at(nx, ny, 0) = 255;//标记为检测
//                    }
//                }
//            }
//        }
//    }
//}

//modify by fuyp 2017-09-22
void TextureView::generate_validity_mask(void)
{
    assert(image != NULL);
    validity_mask.resize(width * height, true);//全部是有效的
    mve::ByteImage::Ptr checked = mve::ByteImage::create(width, height, 1);//1个通道

    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            if( i < G2LTexConfig::get().BOARD_IGNORE || i > height - G2LTexConfig::get().BOARD_IGNORE || j < G2LTexConfig::get().BOARD_IGNORE || j > width - G2LTexConfig::get().BOARD_IGNORE)
            {
                validity_mask[j + i * width] = false;
            }
        }
    }
}

void TextureView::load_image(void) {
    if(image != NULL)
        return;
    image = mve::image::load_file(image_file);
}

/**
 * @brief TextureView::generate_gradient_magnitude  纹理图像生成梯度图
 */
void TextureView::generate_gradient_magnitude(void)
{
    assert(image != NULL);
    mve::ByteImage::Ptr bw = mve::image::desaturate<std::uint8_t>(image, mve::image::DESATURATE_LUMINANCE);
    gradient_magnitude = mve::image::sobel_edge<std::uint8_t>(bw);
}

void TextureView::erode_validity_mask(void)
{
    std::vector<bool> eroded_validity_mask(validity_mask);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1)//初始化的时候已经定义了边界上的点无效
            {
                validity_mask[x + y * width] = false;//图像四周边界上的点认为是无效的点
                continue;
            }

            if (validity_mask[x + y * width])
            {
                continue;
            }
            //无效点周围的点也无效
            for (int j = -1; j <= 1; ++j)
            {
                for (int i = -1; i <= 1; ++i)
                {
                    int const nx = x + i;
                    int const ny = y + j;
                    eroded_validity_mask[nx + ny * width] = false;
                }
            }
        }
    }

    validity_mask.swap(eroded_validity_mask);
}

void TextureView::get_face_info(math::Vec3f const & v1, math::Vec3f const & v2,
                                math::Vec3f const & v3, FaceProjectionInfo * face_info, Settings const & settings) const
{

    assert(image != NULL);
    assert(settings.data_term != DATA_TERM_GMI || gradient_magnitude != NULL);

    //在图像上的投影坐标
    math::Vec2f p1 = get_pixel_coords_noshift(v1);
    math::Vec2f p2 = get_pixel_coords_noshift(v2);
    math::Vec2f p3 = get_pixel_coords_noshift(v3);
//    std::cout<<"p1:"<<p1(0)<<" "<<p1(1)<<std::endl;
//    std::cout<<"p2:"<<p2(0)<<" "<<p2(1)<<std::endl;
//    std::cout<<"p3:"<<p3(0)<<" "<<p3(1)<<std::endl;

    assert(valid_pixel(p1) && valid_pixel(p2) && valid_pixel(p3));

    Tri tri(p1, p2, p3);
    float area = tri.get_area();//投影区域面积

    if (area < std::numeric_limits<float>::epsilon()) //面积太小丢弃
    {
        face_info->quality = 0.0f;
        return;
    }

    std::size_t num_samples = 0;
    math::Vec3d colors(0.0);
    double gmi = 0.0;
    float  featurevalue = 0.0;//记录featuremap的值
    float  detailvalue = 0.0;//记录featuremap的值

    bool sampling_necessary = settings.data_term != DATA_TERM_AREA || settings.outlier_removal != OUTLIER_REMOVAL_NONE;

//    std::cout<<"-----sam----"<<std::endl;
    if (sampling_necessary && area > 0.5f)//如果不使用面积或者不移出离群点则进行采样（默认需要）
    {
//        std::cout<<"------------sampling-------"<<std::endl;
        /* Sort pixels in ascending order of y */
        while (true)//三个点按照y的坐标升序排列P1<P2<P3
        {
            if(p1[1] <= p2[1])
            {
                if(p2[1] <= p3[1])
                {
                    break;
                }
                else
                {
                    std::swap(p2, p3);
                }
            }
            else
            {
                std::swap(p1, p2);
            }
        }

        /* Calculate line equations. */
        float const m1 = (p1[1] - p3[1]) / (p1[0] - p3[0]);//斜率
        float const b1 = p1[1] - m1 * p1[0];//截距

        /* area != 0.0f => m1 != 0.0f. */
        float const m2 = (p1[1] - p2[1]) / (p1[0] - p2[0]);
        float const b2 = p1[1] - m2 * p1[0];

        float const m3 = (p2[1] - p3[1]) / (p2[0] - p3[0]);
        float const b3 = p2[1] - m3 * p2[0];

        bool fast_sampling_possible = std::isfinite(m1) && m2 != 0.0f && std::isfinite(m2) && m3 != 0.0f && std::isfinite(m3);

        Rect<float> aabb = tri.get_aabb();//三角形坐标最大和最小构成的矩形包围盒
        for (int y = std::floor(aabb.min_y); y < std::ceil(aabb.max_y); ++y)//在包围盒内部进行均匀采样计算
        {
            float min_x = aabb.min_x - 0.5f;
            float max_x = aabb.max_x + 0.5f;

            if (fast_sampling_possible)
            {
                float const cy = static_cast<float>(y) + 0.5f;

                min_x = (cy - b1) / m1;
                if (cy <= p2[1])
                    max_x = (cy - b2) / m2;
                else
                    max_x = (cy - b3) / m3;

                if (min_x >= max_x)
                    std::swap(min_x, max_x);

                if (min_x < aabb.min_x || min_x > aabb.max_x)
                    continue;
                if (max_x < aabb.min_x || max_x > aabb.max_x)
                    continue;
            }

            for (int x = std::floor(min_x + 0.5f); x < std::ceil(max_x - 0.5f); ++x)
            {
                math::Vec3d color;

                const float cx = static_cast<float>(x) + 0.5f;
                const float cy = static_cast<float>(y) + 0.5f;
                if (!fast_sampling_possible && !tri.inside(cx, cy))//采样点不在三角面片里面
                    continue;

//                if (settings.outlier_removal != OUTLIER_REMOVAL_NONE)//默认OUTLIER_REMOVAL_NONE
                {
                    for (std::size_t i = 0; i < 3; i++)
                    {
                         color[i] = static_cast<double>(image->at(x, y, i)) / 255.0;
                    }
                    colors += color;//投影区域内颜色和
                    featurevalue += FeatureMap.at<float>(y,x);
                    detailvalue += detailMap.at<uchar>(y, x)/255.0f;

                }

                if (settings.data_term == DATA_TERM_GMI)
                {
                    gmi += static_cast<double>(gradient_magnitude->at(x, y, 0)) / 255.0;//投影区域内梯度所有采样点梯度和
                }
                ++num_samples;//采样点的个数
            }
        }
    }

    if (settings.data_term == DATA_TERM_GMI)
    {
        if (num_samples > 0)
        {
            gmi = (gmi / num_samples) * area;//归一化
        }
        else
        {
            //没有采样点则直接根据三个顶点的颜色进行平均
            double gmv1 = static_cast<double>(gradient_magnitude->linear_at(p1[0], p1[1], 0)) / 255.0;
            double gmv2 = static_cast<double>(gradient_magnitude->linear_at(p2[0], p2[1], 0)) / 255.0;
            double gmv3 = static_cast<double>(gradient_magnitude->linear_at(p3[0], p3[1], 0)) / 255.0;
            gmi = ((gmv1 + gmv2 + gmv3) / 3.0) * area;
        }
    }

//    if (settings.outlier_removal != OUTLIER_REMOVAL_NONE)
    {
        if (num_samples > 0)
        {
            face_info->mean_color = colors / num_samples;//颜色平均值
//            face_info->featuremapvalue = featurevalue/num_samples;//featuremap平均图
            face_info->detailvalue = detailvalue/num_samples;

        }
        else
        {
            math::Vec3d c1, c2, c3;
            float f1, f2, f3;
            for (std::size_t i = 0; i < 3; ++i)
            {
                 c1[i] = static_cast<double>(image->linear_at(p1[0], p1[1], i)) / 255.0;
                 c2[i] = static_cast<double>(image->linear_at(p2[0], p2[1], i)) / 255.0;
                 c3[i] = static_cast<double>(image->linear_at(p3[0], p3[1], i)) / 255.0;
            }
            face_info->mean_color = ((c1 + c2 + c3) / 3.0);//没有采样点直接计算三个顶点的平均值

//            f1 = FeatureMap.at<float>(p1[1],p1[0]);
//            f2 = FeatureMap.at<float>(p2[1],p2[0]);
//            f3 = FeatureMap.at<float>(p3[1],p3[0]);
//            face_info->featuremapvalue = (f1+f2+f3)/3.0;

            float de1, de2, de3;
            de1 = detailMap.at<uchar>(p1[1],p1[0])/255.0f;
            de2 = detailMap.at<uchar>(p2[1],p2[0])/255.0f;
            de3 = detailMap.at<uchar>(p3[1],p3[0])/255.0f;
            face_info->detailvalue = (de1+de2+de3)/3.0f;
        }
    }

    switch (settings.data_term)
    {
    case DATA_TERM_AREA:
        face_info->quality = area;//面积
        break;
    case DATA_TERM_GMI:
        face_info->quality = gmi;//梯度
        break;
    }
}


void TextureView::initmaskimage()
{
    imagemask = cv::Mat(height, width, CV_8UC3, cv::Scalar(0,0,0));
    imagebelieve = cv::Mat(height, width, CV_32FC1, cv::Scalar(0.0));
}

void TextureView::generateFeaturemap(std::string filename)
{
    cv::Mat  inputimg = cv::imread(filename);
    cv::Mat gray_src, dst;
    cv::cvtColor(inputimg, gray_src, CV_BGR2GRAY);

    //cv::imwrite("gray.png", gray_src);

    cv::Mat  grad_x,grad_y, gradImage;
    cv::Sobel(gray_src, grad_x, CV_16S, 1, 0);
    cv::Sobel(gray_src, grad_y, CV_16S, 1, 0);
    gradImage = abs(grad_x) + abs(grad_y);

//    gradImage = 0.5*(grad_x) + 0.5*(grad_y);

    double minGrad, maxGrad;
    cv::minMaxLoc(gradImage, &minGrad, &maxGrad);
    cv::Mat gradImage_8U;
    gradImage.convertTo(gradImage_8U, CV_8U, 255./maxGrad);
    cv::Mat thresholdedImage;//阈值化后的二值图
    cv::threshold(gradImage_8U, thresholdedImage, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);

//    cv::imwrite("bin.png", thresholdedImage);


    cv::Mat erodeStruct = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(thresholdedImage, dst, erodeStruct);

    char buf[256];

//    sprintf(buf,"2featuremap%02d.png", id);
//    cv::imwrite(buf, dst);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat dst2;
    cv::morphologyEx(dst, dst2, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 1);
//    sprintf(buf,"featuremap%02d.png", id);
//    cv::imwrite(buf, dst2);
    dst2.convertTo(FeatureMap, CV_32FC1, 1/255.0f);
//    sprintf(buf,"F_featuremap%02d.png", id);
//    cv::imwrite(buf, FeatureMap);

}

void TextureView::generateDetailmap(std::string filename)
{
    cv::Mat  img = cv::imread(filename);
//    detailMap = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));//存储纹理细节

    //detail level
    cv::Mat  src_gray, src_b;
    cv::cvtColor(img, src_gray, CV_RGB2GRAY);
//    cv::bilateralFilter(src_gray, src_b, 10, 10, 10);
    cv::bilateralFilter(src_gray, src_b, 15, 20, 50);

    cv::Mat  gray32, detail32;
    src_gray.convertTo(gray32, CV_32FC1, 1.0f/255.0f);//原图的亮度度
    src_b.convertTo(detail32, CV_32FC1, 1.0f/255.0f);

    cv::Mat src_log, d_log;
    cv::log(gray32, src_log);
    cv::log(detail32,d_log);
    src_log = src_log - d_log;
    src_log.convertTo(detailMap, CV_8UC1, 255.0f);

    char buf[256];
    sprintf(buf,"detail%02d.png", id);
    cv::imwrite(buf, detailMap);
}

math::Vec2f TextureView::get_depth_pixel_coords(const math::Vec3f &vertex) const
{
    math::Vec3f pixel = depth_projection * world_to_cam.mult(vertex, 1.0f);//投影到相机平面
    pixel /= pixel[2];
    return math::Vec2f(pixel[0] - 0.5f, pixel[1] - 0.5f);//这里到底使用+0.5还是-0.5呢？
//    return math::Vec2f(pixel[0], pixel[1]);//这里到底使用+0.5还是-0.5呢？

}

/**
 * @brief TextureView::valid_pixel  检测投影后的点是否在图像平面上，并且都是有效的点
 * @param pixel
 * @return
 */
//bool TextureView::valid_pixel(math::Vec2f pixel) const
//{
//    float const x = pixel[0];
//    float const y = pixel[1];

//    /* The center of a pixel is in the middle. */
////    bool valid = (x >= 0.0f && x < static_cast<float>(width - 1) && y >= 0.0f && y < static_cast<float>(height - 1));
//    bool valid = (x >= G2LTexConfig::get().BOARD_IGNORE && x <= static_cast<float>(G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
//                  y >= G2LTexConfig::get().BOARD_IGNORE && y <= static_cast<float>(G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE));

////    std::cout<<"valid:"<<valid<<std::endl;

////    if (valid && validity_mask.size() == static_cast<std::size_t>(width * height))
////    {
////        /* Only pixel which can be correctly interpolated are valid. */
////        float cx = std::max(0.0f, std::min(static_cast<float>(width - 1), x));
////        float cy = std::max(0.0f, std::min(static_cast<float>(height - 1), y));
////        int const floor_x = static_cast<int>(cx);
////        int const floor_y = static_cast<int>(cy);
////        int const floor_xp1 = std::min(floor_x + 1, width - 1);
////        int const floor_yp1 = std::min(floor_y + 1, height - 1);

////        /* We screw up if weights would be zero
////         * e.g. we lose valid pixel in the border of images... */

////        valid = validity_mask[floor_x + floor_y * width] &&
////                validity_mask[floor_x + floor_yp1 * width] &&
////                validity_mask[floor_xp1 + floor_y * width] &&
////                validity_mask[floor_xp1 + floor_yp1 * width];
////    }

//    return valid;
//}


/**
 * @brief TextureView::valid_pixel  检测投影后的点是否在图像平面上，并且都是有效的点
 * @param pixel
 * @return
 */
bool TextureView::valid_pixel(math::Vec2f pixel) const
{
    float const x = pixel[0];
    float const y = pixel[1];

    /* The center of a pixel is in the middle. */
    bool valid = (x >= 0.0f && x < static_cast<float>(width - 1) && y >= 0.0f && y < static_cast<float>(height - 1));
//    bool valid = (x > G2LTexConfig::get().BOARD_IGNORE && x < static_cast<float>(width - G2LTexConfig::get().BOARD_IGNORE) && y > G2LTexConfig::get().BOARD_IGNORE && y < static_cast<float>(height - G2LTexConfig::get().BOARD_IGNORE));


    if (valid && validity_mask.size() == static_cast<std::size_t>(width * height))
    {
        /* Only pixel which can be correctly interpolated are valid. */
        float cx = std::max(0.0f, std::min(static_cast<float>(width - 1), x));
        float cy = std::max(0.0f, std::min(static_cast<float>(height - 1), y));
        int const floor_x = static_cast<int>(cx);
        int const floor_y = static_cast<int>(cy);
        int const floor_xp1 = std::min(floor_x + 1, width - 1);
        int const floor_yp1 = std::min(floor_y + 1, height - 1);

        /* We screw up if weights would be zero
         * e.g. we lose valid pixel in the border of images... */

        valid = validity_mask[floor_x + floor_y * width] &&
                validity_mask[floor_x + floor_yp1 * width] &&
                validity_mask[floor_xp1 + floor_y * width] &&
                validity_mask[floor_xp1 + floor_yp1 * width];
    }

    return valid;
}


void
TextureView::export_triangle(math::Vec3f v1, math::Vec3f v2, math::Vec3f v3,
    std::string const & filename) const {
    assert(image != NULL);
    math::Vec2f p1 = get_pixel_coords_noshift(v1);
    math::Vec2f p2 = get_pixel_coords_noshift(v2);
    math::Vec2f p3 = get_pixel_coords_noshift(v3);

    assert(valid_pixel(p1) && valid_pixel(p2) && valid_pixel(p3));

    Tri tri(p1, p2, p3);

    Rect<float> aabb = tri.get_aabb();
    const int width = ceil(aabb.width());
    const int height = ceil(aabb.height());
    const int left = floor(aabb.min_x);
    const int top = floor(aabb.max_y);

    assert(width > 0 && height > 0);
    mve::image::save_png_file(mve::image::crop(image, width, height, left, top,
        *math::Vec3uc(255, 0, 255)), filename);
}

void
TextureView::export_validity_mask(std::string const & filename) const {
    assert(validity_mask.size() == static_cast<std::size_t>(width * height));
    mve::ByteImage::Ptr img = mve::ByteImage::create(width, height, 1);
    for (std::size_t i = 0; i < validity_mask.size(); ++i) {
        img->at(static_cast<int>(i), 0) = validity_mask[i] ? 255 : 0;
    }
    mve::image::save_png_file(img, filename);
}

math::Matrix3f TextureView::getCamToWordRotation() const
{
    math::Matrix3f mat;
    mat[0]  = world_to_cam[0]; mat[1] = world_to_cam[4]; mat[2] = world_to_cam[8];
    mat[3]  = world_to_cam[1]; mat[4] = world_to_cam[5]; mat[5] = world_to_cam[9];
    mat[6]  = world_to_cam[2]; mat[7] = world_to_cam[6]; mat[8] = world_to_cam[10];
    return mat;
}

math::Matrix4f TextureView::getWorldToCamMatrix() const
{
    return world_to_cam;
}

void TextureView::setWorldToCamMatrix(Eigen::Matrix4f mat)
{
    world_to_cam[0] = mat(0,0); world_to_cam[1]  =  mat(0,1); world_to_cam[2]  =  mat(0,2); world_to_cam[3]  =  mat(0,3);
    world_to_cam[4] = mat(1,0); world_to_cam[5]  =  mat(1,1); world_to_cam[6]  =  mat(1,2); world_to_cam[7]  =  mat(1,3);
    world_to_cam[8] = mat(2,0); world_to_cam[9]  =  mat(2,1); world_to_cam[10] =  mat(2,2); world_to_cam[11] =  mat(2,3);
    world_to_cam[12] = mat(3,0); world_to_cam[13] = mat(3,1); world_to_cam[14] =  mat(3,2); world_to_cam[15] =  mat(3,3);
//    world_to_cam = mat;
}

math::Matrix3f TextureView::getDepthCameraProjection() const
{
    return  depth_projection;
}

math::Matrix3f TextureView::getImageCameraProjection() const
{
    return  image_projection;
}

TEX_NAMESPACE_END
