#ifndef LL_H
#define LL_H
#include <ceres/ceres.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


struct BaseAndDetail
{
    cv::Mat   baseLayer;//基础层
    cv::Mat   detailLayer;//细节层
};

struct Lab_L2
{
    double A_v, B_v;//值
    double A_a, A_b;//LAB颜色偏移
};

class ColorHarmonValue//两
{
public:
    ColorHarmonValue()
    {
        matches.clear();
        nums = 0;
    }

    ~ColorHarmonValue()
    {
        matches.clear();
        nums = 0;
    }

    void mypush_back(Lab_L2  ll)
    {
        matches.push_back(ll);
        nums++;
    }

    Lab_L2 at(int index)
    {
        if(index < nums)
        {
            return matches[index];
        }
        else
        {
            std::cout<<"-------------------error---------------"<<std::endl;
            std::exit(0);
        }
    }

public:
    std::vector<Lab_L2>  matches;
    int nums;
};

//ceres
struct colorCostFunctor1
{
    colorCostFunctor1(double l1, double a, double b, double of_a, double of_b)
        :l1(l1), a(a), b(b),off_a(of_a),off_b(of_b)
    {

    }

    template <typename T>
    bool operator()(const T* const scale1, const T* const offset1,
                                           T* residual) const
    {
        const T m1 = scale1[0]*T(a) + offset1[0];
        const T m2 = T(off_a)*T(b) + T(off_b);
        residual[0] = T(ceres::sqrt(l1)*(m1 - m2));
        return true;
    }

    static ceres::CostFunction* create(double l1, double a, double b, double of_a, double of_b)
    {
        return (new ceres::AutoDiffCostFunction<colorCostFunctor1, 1, 1, 1>(new colorCostFunctor1(l1, a, b, of_a, of_b)));
    }
public:
    double l1;
    double a;
    double b;
    double off_a;
    double off_b;
};

struct colorCostFunctor2 {
    colorCostFunctor2(double l2) :l2(l2)
    {

    }

   template <typename T>
   bool operator()(const T* const scale, const T* const offset, T* residual) const
   {

      residual[0] = T(ceres::sqrt(l2)) * (scale[0] - T(1.0));
      residual[1] = T(ceres::sqrt(l2)) * (offset[0]);

      return true;
   }

   static ceres::CostFunction* Create(const double l2)
    {
        return (new ceres::AutoDiffCostFunction<colorCostFunctor2, 2,1, 1>(new colorCostFunctor2(l2)));
   }
   const double l2;
};

#endif // LL_H
