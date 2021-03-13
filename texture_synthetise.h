/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */


#ifndef TEXTURE_SYNTHETISE_H
#define TEXTURE_SYNTHETISE_H
#include "patchmatch.h"
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tex/texturing.h>
#include <vector>

TEX_NAMESPACE_BEGIN

Eigen::Vector3f getInterRGBFromImg(cv::Mat  img, float u, float v);
void generateDepthImageByRayCasting(mve::TriangleMesh::ConstPtr mesh, mve::MeshInfo const & mesh_info,
                                    tex::TextureView &view);

Eigen::Vector3f  generateColorByTwopassTextureimage(std::vector<TextureView> texture_views, int curlabel,
                                                    int twopasslabel, int px, int py);

Eigen::Vector3f generateRGBByRemapping(mve::TriangleMesh::ConstPtr mesh, std::vector<std::vector<TextureView> > PyramidViews,
                                       int curlabel, int twopasslabel, int px, int py, int level, int &count);

Eigen::Vector3f generateRGBAndWeightByRemapping(mve::TriangleMesh::ConstPtr mesh, std::vector<std::vector<TextureView> > PyramidViews,
                                       int curlabel, int twopasslabel, int px, int py, int level, int &count, float &weight);


Eigen::Vector3f generateRGBByRemappingFortesting(mve::TriangleMesh::ConstPtr mesh, std::vector<std::vector<TextureView> > PyramidViews,
                                       int curlabel, int twopasslabel, int px, int py, int level, int &count);

Eigen::Vector3f generateRGBByPrespective(mve::TriangleMesh::ConstPtr mesh, std::vector<std::vector<TextureView> > PyramidViews,
                                       int curlabel, int twopasslabel, int px, int py, int level, int &count);

Eigen::Vector3f generateRGBByRemappingByViews(mve::TriangleMesh::ConstPtr mesh, std::vector<TextureView> texture_views,
                                       int curlabel, int twopasslabel, int px, int py, int &count);
TEX_NAMESPACE_END

#endif // TEXTURE_SYNTHETISE_H
