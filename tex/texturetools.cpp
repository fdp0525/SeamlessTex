/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */


#include "texturing.h"
TEX_NAMESPACE_BEGIN

float getInterColorFromRGBImgV2(cv::Mat  img, float u, float v)
{
    int x = floor(u);
    int y = floor(v);
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

    float value = 0.114*valueb + 0.299*valueg + 0.587*valuer;
    return value;
}

float getInterColorFromGrayImgV2(cv::Mat  img, float u, float v)
{
    int x = floor(u);
    int y = floor(v);
    float offsetx = u - x;
    float offsety = v - y;
    float value = (1-offsetx)*(1-offsety)*img.at<float>(y, x)
            +(1-offsetx)*offsety*img.at<float>(y+1, x)
            +(1-offsety)*offsetx*img.at<float>(y, x+1)
            +offsetx*offsety*img.at<float>(y + 1, x + 1);
    return value;
}


float getInterColorFromGrayImg(cv::Mat  img, float u, float v)
{
    int x = floor(u);
    int y = floor(v);
    float offsetx = u - x;
    float offsety = v - y;
    float value = (1-offsetx)*(1-offsety)*img.at<uchar>(y, x)
            +(1-offsetx)*offsety*img.at<uchar>(y+1, x)
            +(1-offsety)*offsetx*img.at<uchar>(y, x+1)
            +offsetx*offsety*img.at<uchar>(y + 1, x + 1);
    return value/255.0f;
}


void generateGradImgV2(cv::Mat&  in, cv::Mat& outx, cv::Mat& outy)
{
    int width = in.cols;
    int height = in.rows;
    outx = cv::Mat(height, width, CV_32FC1);
    outy = cv::Mat(height, width, CV_32FC1);
    //    out = cv::Mat(height,width, CV_16SC1);

    float gsx3x3[9] = {0.52201,  0.00000, -0.52201,
                       0.79451, -0.00000, -0.79451,
                       0.52201,  0.00000, -0.52201};

    float gsy3x3[9] = {0.52201, 0.79451, 0.52201,
                       0.00000, 0.00000, 0.00000,
                       -0.52201, -0.79451, -0.52201};

    for( int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++ )
        {
            float dxVal = 0;
            float dyVal = 0;

            int kernelIndex = 8;
            for(int j = std::max(y - 1, 0); j <= std::min(y + 1, height - 1); j++)
            {
                for(int i = std::max(x - 1, 0); i <= std::min(x + 1, width - 1); i++)
                {
                    dxVal += (float)in.at<uchar>(j, i) * gsx3x3[kernelIndex];
                    dyVal += (float)in.at<uchar>(j, i) * gsy3x3[kernelIndex];
                    --kernelIndex;
                }
            }

            outx.at<float>(y,x) = dxVal;
            outy.at<float>(y,x) = dyVal;
            //            out.at<short>(y,x) = std::sqrt(dxVal*dxVal+dyVal*dyVal);
        }
    }

}

bool checkNAN(Eigen::Matrix4f mat)
{
    if(std::isnan(mat(0, 0)) || std::isnan(mat(0, 1)) || std::isnan(mat(0, 2)) || std::isnan(mat(0, 3)) ||
            std::isnan(mat(1, 0)) || std::isnan(mat(1, 1)) || std::isnan(mat(1, 2)) || std::isnan(mat(1, 3)) ||
            std::isnan(mat(2, 0)) || std::isnan(mat(2, 1)) || std::isnan(mat(2, 2)) || std::isnan(mat(2, 3)) ||
            std::isnan(mat(3, 0)) || std::isnan(mat(3, 1)) || std::isnan(mat(3, 2)) || std::isnan(mat(3, 3)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

Eigen::Matrix4f  mathToEigen(math::Matrix4f  mat)
{
    Eigen::Matrix4f  wtc;
    wtc<<mat[0],mat[1],mat[2],mat[3],
         mat[4],mat[5],mat[6],mat[7],
         mat[8],mat[9],mat[10],mat[11],
         mat[12],mat[13],mat[14],mat[15];
    return wtc;
}

math::Matrix4f  eigenToMath(Eigen::Matrix4f init)
{
    math::Matrix4f mat;
    mat[0] = init(0,0); mat[1]  =  init(0,1); mat[2]  =  init(0,2); mat[3]  =  init(0,3);
    mat[4] = init(1,0); mat[5]  =  init(1,1); mat[6]  =  init(1,2); mat[7]  =  init(1,3);
    mat[8] = init(2,0); mat[9]  =  init(2,1); mat[10] =  init(2,2); mat[11] =  init(2,3);
    mat[12] = init(3,0); mat[13] = init(3,1); mat[14] =  init(3,2); mat[15] =  init(3,3);
}

TEX_NAMESPACE_END
