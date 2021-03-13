#ifndef PATCHMACTHGPU_H
#define PATCHMACTHGPU_H
#include <opencv2/opencv.hpp>

void patchmatchGPU(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd,
                   int patch_w, int pm_iters);
#endif // PATCHMACTHGPU_H
