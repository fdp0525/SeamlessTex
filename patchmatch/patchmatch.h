#ifndef PATCHMATCH_H
#define PATCHMATCH_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace PatchMatch {


void patchmatch(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd);
void patchmatchwithSearchArea(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd, float factor = 0.1);
void PatchMatchWithGPU(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd);

void patchmatchBySeams(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd,
                       std::vector<std::vector<cv::Point2f> >  contours);
void patchmatchBySeamsDistacne(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd,
                       cv::Mat  seaminfo);

void BDSCompleteness(cv::Mat img, cv::Mat nn, cv::Mat nnd, cv::Mat &cohSumMat);
void BDSCoherence(cv::Mat img, cv::Mat nn, cv::Mat nnd, cv::Mat &cohSumMat);


}
#endif // PATCHMATCH_H
