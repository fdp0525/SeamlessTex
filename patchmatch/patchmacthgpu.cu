#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <curand_kernel.h>
#include "time.h"
#include "patchmacthgpu.h"

using namespace cv;
using namespace cv::gpu;
using namespace std;


__host__ __device__ unsigned int XY_TO_INT(int x, int y) {
    return ((y) << 12) | (x);
}
__host__ __device__ int INT_TO_X(unsigned int v) {
    return (v)&((1 << 12) - 1);
}
__host__ __device__ int INT_TO_Y(unsigned int v) {
    return (v >> 12)&((1 << 12) - 1);
}
__host__ __device__ int cuMax(int a, int b) {
    if (a>b) {
        return a;
    }
    else {
        return b;
    }
}
__host__ __device__ int cuMin(int a, int b) {
    if (a<b) {
        return a;
    }
    else {
        return b;
    }
}

__host__ int dist_p(Mat a, Mat b, int ax, int ay, int bx, int by) {
    Vec3b ai = a.at<Vec3b>(ay, ax);
    Vec3b bi = b.at<Vec3b>(by, bx);
    int dr = abs(ai.val[2] - bi.val[2]);
    int dg = abs(ai.val[1] - bi.val[1]);
    int db = abs(ai.val[0] - bi.val[0]);
    return dr*dr + dg*dg + db*db;
}

/* nearest voting */
__host__ Mat reconstruct_nearest(Mat a, Mat b, unsigned int * ann, int patch_w) {
    Mat c;
    a.copyTo(c);

    int ystart = 0, yend = a.rows, ychange = 1;
    int xstart = 0, xend = a.cols, xchange = 1;
    unsigned int ybest = 0, xbest = 0, v = 0;
    //difference of pixel
    int ** pnnd;
    unsigned int ** pnn;
    pnn = new unsigned int *[a.rows];
    pnnd = new int *[a.rows];
    for (int i = 0; i < a.rows; i++)
    {
        pnn[i] = new unsigned int[a.cols];
        pnnd[i] = new int[a.cols];
        memset(pnn[i], 0, a.cols);
    }

    // initialization
    for (int ay = 0; ay < a.rows; ay++) {
        for (int ax = 0; ax < a.cols; ax++) {
            pnn[ay][ax] = ann[ay*a.cols + ax];
            v = ann[ay*a.cols + ax];
            xbest = INT_TO_X(v);
            ybest = INT_TO_Y(v);
            pnnd[ay][ax] = dist_p(a, b, ax, ay, xbest, ybest);

        }
    }


    for (int ay = ystart; ay != yend; ay += ychange) {
        for (int ax = xstart; ax != xend; ax += xchange) {
            v = ann[ay*a.cols+ax];
            xbest = INT_TO_X(v);
            ybest = INT_TO_Y(v);
            //find its corresponding patch in B
            for (int dy = -patch_w/2; dy < patch_w/2; dy++) {
                for (int dx = -patch_w/2; dx < patch_w/2; dx++) {
                    if (// if the pixels are both in A or B, then count there distance and update if possible
                        (ay + dy) < a.rows && (ay + dy) >= 0 && (ax + dx) < a.cols && (ax + dx) >= 0
                        &&
                        (ybest + dy) < b.rows && (ybest + dy) >= 0 && (xbest + dx) < b.cols && (xbest + dx) >= 0
                        ) {
                        if (pnnd[ay + dy][ax + dx]>dist_p(a, b, ax + dx, ay + dy, xbest + dx, ybest + dy)) {
                            pnn[ay + dy][ax + dx] = XY_TO_INT(xbest + dx, ybest + dy);
                            pnnd[ay + dy][ax + dx] = dist_p(a, b, ax + dx, ay + dy, xbest + dx, ybest + dy);
                        }
                    }
                }

            }
        }
    }

    for (int ay = ystart; ay < a.rows; ay++) {
        for (int ax = xstart; ax < a.cols; ax++)
        {
            v = pnn[ay][ax];
            xbest = INT_TO_X(v);
            ybest = INT_TO_Y(v);

            Vec3b bi = b.at<Vec3b>(ybest, xbest);
            c.at<Vec3b>(ay, ax).val[2] = bi.val[2];
            c.at<Vec3b>(ay, ax).val[1] = bi.val[1];
            c.at<Vec3b>(ay, ax).val[0] = bi.val[0];
        }
    }
    return c;
}

/* center voting */
__host__ Mat reconstruct_center(Mat a, Mat b, unsigned int * ann, int patch_w) {
    Mat c;
    a.copyTo(c);
    for (int ay = 0; ay < a.rows; ay++) {
        for (int ax = 0; ax < a.cols; ax++)
        {
            unsigned int v = ann[ay*a.cols + ax];
            int xbest = INT_TO_X(v);
            int ybest = INT_TO_Y(v);

            Vec3b bi = b.at<Vec3b>(ybest, xbest);
            c.at<Vec3b>(ay, ax).val[2] = b.at<Vec3b>(ybest, xbest).val[2];
            c.at<Vec3b>(ay, ax).val[1] = b.at<Vec3b>(ybest, xbest).val[1];
            c.at<Vec3b>(ay, ax).val[0] = b.at<Vec3b>(ybest, xbest).val[0];
        }
    }
    return c;
}

__host__ Mat reconstruct_flow(Mat a, Mat b, unsigned int * ann, int patch_w) {
    Mat flow;
    a.copyTo(flow);
    for (int ay = 0; ay < a.rows; ay++) {
        for (int ax = 0; ax < a.cols; ax++)
        {
            unsigned int v = ann[ay*a.cols + ax];
            int xbest = INT_TO_X(v);
            int ybest = INT_TO_Y(v);

            Vec3b bi = b.at<Vec3b>(ybest, xbest);
            flow.at<Vec3b>(ay, ax).val[0] = (uchar)255 * ((float)xbest / b.cols);
            flow.at<Vec3b>(ay, ax).val[2] = 0;
            flow.at<Vec3b>(ay, ax).val[1] = (uchar)255 * ((float)ybest / b.rows);
        }
    }
    return flow;
}

__host__ Mat reconstruct_error(Mat a, Mat b, unsigned int * ann, int patch_w) {
    Mat c;
    a.copyTo(c);

    int * err, err_min=INT_MAX, err_max=0;
    err = new int[a.rows*a.cols];

    for (int ay = 0; ay < a.rows; ay++) {
        for (int ax = 0; ax < a.cols; ax++)
        {
            unsigned int v = ann[ay*a.cols + ax];
            int xbest = INT_TO_X(v);
            int ybest = INT_TO_Y(v);

            Vec3b bi = b.at<Vec3b>(ybest, xbest);
            Vec3b ai = a.at<Vec3b>(ay, ax);
            int err_0 = ai.val[0] - bi.val[0];
            int err_1 = ai.val[1] - bi.val[1];
            int err_2 = ai.val[2] - bi.val[2];
            int error = err_0*err_0 + err_1*err_1 + err_2*err_2;
            if (error<err_min)
            {
                err_min = error;
            }

            if (error>err_max)
            {
                err_max = error;
            }

        }
    }

    for (int ay = 0; ay < a.rows; ay++) {
        for (int ax = 0; ax < a.cols; ax++)
        {
            c.at<Vec3b>(ay, ax).val[1] = (uchar)255 * ((float)(err[ay*a.cols+ax]-err_min)/(err_max-err_min));
            c.at<Vec3b>(ay, ax).val[2] = 0;
            c.at<Vec3b>(ay, ax).val[0] = 0;

        }
    }

    return c;
}

__host__ Mat reconstruct_energy(Mat a, Mat b, unsigned int * ann, float * annd, int patch_w) {
    Mat c;
    a.copyTo(c);
    float dmin=annd[0], dmax=annd[0];
    for (int ay = 0; ay < a.rows; ay++) {
        for (int ax = 0; ax < a.cols; ax++)
        {

            float dbest = annd[ay*a.cols + ax];
            if (dmin > dbest)
            {
                dmin = dbest;
            }

            if (dmax<dbest)
            {
                dmax = dbest;
            }

        }
    }

    for (int ay = 0; ay < a.rows; ay++) {
        for (int ax = 0; ax < a.cols; ax++)
        {
            float dbest = annd[ay*a.cols + ax];
            c.at<Vec3b>(ay, ax).val[2] = (uchar)255 * ((float)(dbest - dmin) / (dmax - dmin));
            c.at<Vec3b>(ay, ax).val[0] = 0;
            c.at<Vec3b>(ay, ax).val[1] = 0;
        }
    }


    return c;
}

__host__ __device__ float dist(int * a, int * b, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int bx, int by, int patch_w, float cutoff = INT_MAX) {//this is the average number of all matched pixel
    //suppose patch_w is an odd number
    float pixel_sum = 0, pixel_no = 0, pixel_dist=0;//number of pixels realy counted
//    for (int dy = -patch_w/2; dy <= patch_w/2; dy++) {
//        for (int dx = -patch_w / 2; dx <= patch_w/2; dx++) {
    for (int dy = 0; dy < patch_w; dy++) {
        for (int dx = 0; dx < patch_w; dx++) {
            if (
                (ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
                &&
                (by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
                )//the pixel in a should exist and pixel in b should exist
            {
                int dr = a[(ay + dy) * 3 * a_cols + (ax + dx) * 3 + 2] - b[(by + dy) * 3 * b_cols + (bx + dx) * 3 + 2];
                int dg = a[(ay + dy) * 3 * a_cols + (ax + dx) * 3 + 1] - b[(by + dy) * 3 * b_cols + (bx + dx) * 3 + 1];
                int db = a[(ay + dy) * 3 * a_cols + (ax + dx) * 3 + 0] - b[(by + dy) * 3 * b_cols + (bx + dx) * 3 + 0];
                pixel_sum += (float)(dr*dr + dg*dg + db*db);
                pixel_no += 1;
            }
        }

    }
    pixel_dist = pixel_sum / pixel_no;
    if (pixel_dist >= cutoff) { return cutoff; }
    else {
        return pixel_dist;
    }
}

__host__ void convertMatToArray(Mat mat, int *& arr) {
    arr = new int[mat.rows*mat.cols * 3];
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j <mat.cols; j++)
        {
            Vec3b rgb = mat.at<Vec3b>(i, j);
            arr[i*mat.cols * 3 + j * 3 + 0] = rgb.val[0];
            arr[i*mat.cols * 3 + j * 3 + 1] = rgb.val[1];
            arr[i*mat.cols * 3 + j * 3 + 2] = rgb.val[2];
        }
    }
}

__host__ void convertArrayToMat(Mat & mat, unsigned int *arr) {
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j <mat.cols; j++)
        {
            mat.at<Vec3b>(i, j).val[0] = arr[i*mat.cols * 3 + j * 3 + 0];
            mat.at<Vec3b>(i, j).val[1] = arr[i*mat.cols * 3 + j * 3 + 1];
            mat.at<Vec3b>(i, j).val[2] = arr[i*mat.cols * 3 + j * 3 + 2];
        }
    }
}

__host__ void convertMatToANNArray(Mat & mat, unsigned int *arr) {
    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j <mat.cols; j++)
        {
            mat.at<int>(i, j) = arr[i*mat.cols +j];
        }
    }
}

__host__ void initialAnn(unsigned int *& ann, float *& annd, int aw, int ah, int bw, int bh, int a_cols, int a_rows, int b_cols, int b_rows, int * a, int * b, int patch_w) {
    for (int ay = 0; ay < a_rows; ay++) {
        for (int ax = 0; ax < a_cols; ax++) {
            int bx = rand() % b_cols;
            int by = rand() % b_rows;

            ann[ay*a_cols + ax] = XY_TO_INT(bx, by);
            annd[ay*a_cols + ax] = dist(a, b, ah, aw, bh, bw, ax, ay, bx, by, patch_w);

        }
    }
}

__host__ void print(String string) {
    cout << string;
}

__device__ void improve_guess(int * a, int * b, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int &xbest, int &ybest, float &dbest, int xp, int yp, int patch_w) {
    float d = 0;
    d = dist(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w, dbest);

    if (d < dbest) {
        xbest = xp;
        ybest = yp;
        dbest = d;
    }
}

__device__ float cuRand(unsigned int * seed) {//random number in cuda
    unsigned long a = 16807;
    unsigned long m = 2147483647;
    unsigned long x = (unsigned long)* seed;
    x = (a*x) % m;
    *seed = (unsigned int)x;
    return ((float)x / m);
}

/*get the approximate nearest neighbor and set it into ann
********************************************************
params: 7
-----------------------------------------------
0 - a_rows
1 - a_cols
2 - b_rows
3 - b_cols
4 - patch_w
5 - pm_iter
6 - rs_max
*********************************************************/
__global__ void patchmatch(int * a, int * b, unsigned int *ann, float *annd, int * params) {

    int ax = blockIdx.x*blockDim.x + threadIdx.x;
    int ay = blockIdx.y*blockDim.y + threadIdx.y;

    //assign params
    int a_rows = params[0];
    int a_cols = params[1];
    int b_rows = params[2];
    int b_cols = params[3];
    int patch_w = params[4];
    int pm_iters = params[5];
    int rs_max = params[6];

    int iterDirection = 1;//left->right, up->down

    if (ax < a_cols&&ay < a_rows) {

        // for random number
        unsigned int seed = ay*a_cols + ax;

        for (int iter = 0; iter < pm_iters; iter++) {
            /* In each iteration, improve the NNF, by jumping flooding. */
            if (iter%2==0)
            {
                iterDirection = 1;
            }
            else
            {
                iterDirection = -1;
            }

            /* Current (best) guess. */
            unsigned int v = ann[ay*a_cols + ax];
            int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
            float dbest = annd[ay*a_cols + ax];

            for (int jump = 8; jump > 0; jump /= 2) {

                //jump = jump * iterDirection;

                /* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
                if ((ax - jump) < a_cols&&(ax - jump) >= 0)//left
                {
                    unsigned int vp = ann[ay*a_cols + ax - jump];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp) + jump, yp = INT_TO_Y(vp);//the propagated match from vp
                    if (xp < b_cols && xp>=0)
                    {
                        //improve guress
                        improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);

                    }
                }
                ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
                annd[ay*a_cols + ax] = dbest;


                if ((ax + jump) < a_cols)//right
                {
                    unsigned int vp = ann[ay*a_cols + ax + jump];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp) - jump, yp = INT_TO_Y(vp);
                    if (xp >= 0&&xp<b_cols)
                    {
                        //improve guress
                        improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
                    }
                }

                ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
                annd[ay*a_cols + ax] = dbest;

                if ((ay - jump) < a_rows && (ay - jump) >=0)//up
                {
                    unsigned int vp = ann[(ay - jump)*a_cols + ax];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + jump;
                    if (yp >= 0 && yp <b_rows)
                    {
                        //improve guress
                        improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
                    }
                }

                ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
                annd[ay*a_cols + ax] = dbest;

                if ((ay + jump) < a_rows)//down
                {
                    unsigned int vp = ann[(ay + jump)*a_cols + ax];//the pixel coordinates in image b
                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - jump;
                    if (yp >= 0)
                    {
                        //improve guress
                        improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);
                    }
                }

                ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
                annd[ay*a_cols + ax] = dbest;
                __syncthreads();

            }

            /* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
            int rs_start = rs_max;
            if (rs_start > cuMax(b_cols, b_rows)) {
                rs_start = cuMax(b_cols, b_rows);
            }
            for (int mag = rs_start; mag >= 1; mag /= 2) {
                /* Sampling window */
                int xmin = cuMax(xbest - mag, 0), xmax = cuMin(xbest + mag + 1, b_cols);
                int ymin = cuMax(ybest - mag, 0), ymax = cuMin(ybest + mag + 1, b_rows);
                int xp = xmin + (int)(cuRand(&seed)*(xmax - xmin)) % (xmax - xmin);
                int yp = ymin + (int)(cuRand(&seed)*(ymax - ymin)) % (ymax - ymin);

                improve_guess(a, b, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w);

            }

            ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
            annd[ay*a_cols + ax] = dbest;
            __syncthreads();
        }

    }
}

void patchmatchGPU(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd,
                   int patch_w, int pm_iters)
{
    cudaEvent_t start, stop;
    float elapsedTime;

    unsigned int * newann, *ann_host, *ann_device;
    float *annd_host, *newannd, *annd_device;
    int * a_host, *b_host, *a_device, *b_device, *params_host, *params_device;
    int sizeOfAnn = a.rows*a.cols;
//    ann_host = (unsigned int *)malloc(sizeOfAnn * sizeof(unsigned int));
//    annd_host = (float *)malloc(sizeOfAnn * sizeof(float));
//    newann = (unsigned int *)malloc(sizeOfAnn * sizeof(unsigned int));
//    newannd = (float *)malloc(sizeOfAnn * sizeof(float));

//    const int patch_w = 3;
//    int pm_iters = 5;
    int sizeOfParams = 7;
    int rs_max = INT_MAX;

    int a_size = a.rows*a.cols * 3;
    int b_size = b.rows*b.cols * 3;


    dim3 blocksPerGrid(a.cols / 32 + 1, a.rows / 32 + 1, 1);
    dim3 threadsPerBlock(32, 32, 1);

    /* initialization */
    ann_host = (unsigned int *)malloc(sizeOfAnn * sizeof(unsigned int));
    annd_host = (float *)malloc(sizeOfAnn * sizeof(float));
    newann = (unsigned int *)malloc(sizeOfAnn * sizeof(unsigned int));
    newannd = (float *)malloc(sizeOfAnn * sizeof(float));

    params_host = (int *)malloc(sizeOfParams * sizeof(int));
    params_host[0] = a.rows;
    params_host[1] = a.cols;
    params_host[2] = b.rows;
    params_host[3] = b.cols;
    params_host[4] = patch_w;
    params_host[5] = pm_iters;
    params_host[6] = rs_max;


    convertMatToArray(a, a_host);
    convertMatToArray(b, b_host);
    initialAnn(ann_host, annd_host, a.cols, a.rows, b.cols, b.rows, a.cols, a.rows, b.cols, b.rows, a_host, b_host, patch_w);

    cudaMalloc(&a_device, a_size * sizeof(int));
    cudaMalloc(&b_device, b_size * sizeof(int));
    cudaMalloc(&annd_device, sizeOfAnn * sizeof(float));
    cudaMalloc(&ann_device, sizeOfAnn * sizeof(unsigned int));
    cudaMalloc(&params_device, sizeOfParams * sizeof(int));

    cudaMemcpy(a_device, a_host, a_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_host, b_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ann_device, ann_host, sizeOfAnn * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(annd_device, annd_host, sizeOfAnn * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(params_device, params_host, sizeOfParams * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    patchmatch<<<blocksPerGrid, threadsPerBlock>>>(a_device, b_device, ann_device, annd_device, params_device);
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
//    printf("Elapsed time : %f ms\n", elapsedTime);
    cudaMemcpy(newann, ann_device, sizeOfAnn * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(newannd, annd_device, sizeOfAnn * sizeof(float), cudaMemcpyDeviceToHost);

//    convertArrayToMat(ann, newann);
//    convertArrayToMat(annd, newannd);
    convertMatToANNArray(ann, newann);

    delete [] a_host;
    delete [] b_host;
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(ann_device);
    cudaFree(annd_device);
    cudaFree(params_device);

    free(ann_host);
    free(annd_host);
    free(newann);
    free(newannd);
    free(params_host);

}
