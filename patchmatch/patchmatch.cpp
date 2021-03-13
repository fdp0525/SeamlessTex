#include "patchmatch/patchmatch.h"
#include "patchmatch/patchmacthgpu.h"

#ifndef MAX
#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#endif

/* -------------------------------------------------------------------------
   BITMAP: Minimal image class
   ------------------------------------------------------------------------- */

namespace PatchMatch {

/* -------------------------------------------------------------------------
   PatchMatch, using L2 distance between upright patches that translate only
   ------------------------------------------------------------------------- */

int patch_w  = 7;
int pm_iters = 15;
int rs_max   = INT_MAX;

#define XY_TO_INT(x, y) (((y)<<12)|(x))
#define INT_TO_X(v) ((v)&((1<<12)-1))
#define INT_TO_Y(v) ((v)>>12)

/* Measure distance between 2 patches with upper left corners (ax, ay) and (bx, by), terminating early if we exceed a cutoff distance.
   You could implement your own descriptor here. */
//int dist(BITMAP *a, BITMAP *b, int ax, int ay, int bx, int by, int cutoff=INT_MAX)
int dist(cv::Mat a, cv::Mat b, int ax, int ay, int bx, int by, int cutoff=INT_MAX)
{
    int ans = 0;

    //注意这里的patch计算的方法。
    for (int dy = 0; dy < patch_w; dy++)
    {
        for (int dx = 0; dx < patch_w; dx++)
        {
            int dr = a.at<cv::Vec3b>(ay + dy, ax + dx)[0] - b.at<cv::Vec3b>(by + dy, bx + dx)[0];
            int dg = a.at<cv::Vec3b>(ay + dy, ax + dx)[1] - b.at<cv::Vec3b>(by + dy, bx + dx)[1];
            int db = a.at<cv::Vec3b>(ay + dy, ax + dx)[2] - b.at<cv::Vec3b>(by + dy, bx + dx)[2];
            ans += dr*dr + dg*dg + db*db;
        }

        if (ans >= cutoff)
        {
            return cutoff;
        }
    }
    return ans;
}

void improve_guess(cv::Mat a, cv::Mat b, int ax, int ay, int &xbest, int &ybest, int &dbest, int bx, int by)
{
    int d = dist(a, b, ax, ay, bx, by, dbest);
    if (d < dbest)
    {
        dbest = d;
        xbest = bx;
        ybest = by;
    }
}

/* Match image a to image b, returning the nearest neighbor field mapping a => b coords, stored in an RGB 24-bit image as (by<<12)|bx. */
void patchmatch(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd)
{
    /* Initialize with random nearest neighbor field (NNF). */
    //    ann = new BITMAP(a->w, a->h);
    //    annd = new BITMAP(a->w, a->h);
    //    ann = cv::Mat(a.rows, a.cols, CV_32SC1, cv::Scalar(0));
    //    annd = cv::Mat(a.rows, a.cols, CV_32SC1, cv::Scalar(0));

    int aew = a.cols - patch_w + 1, aeh = a.rows - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
    int bew = b.cols - patch_w + 1, beh = b.rows - patch_w + 1;
    for (int ay = 0; ay < aeh; ay++)
    {
        for (int ax = 0; ax < aew; ax++)
        {
            int bx = rand()%bew;
            int by = rand()%beh;
            ann.at<int>(ay, ax) = XY_TO_INT(bx, by);
            annd.at<int>(ay,ax) = dist(a, b, ax, ay, bx, by);
        }
    }

    for (int iter = 0; iter < pm_iters; iter++)
    {
        /* In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. */
        int ystart = 0, yend = aeh, ychange = 1;
        int xstart = 0, xend = aew, xchange = 1;
        if (iter % 2 == 1)
        {
            xstart = xend-1; xend = -1; xchange = -1;
            ystart = yend-1; yend = -1; ychange = -1;
        }
        for (int ay = ystart; ay != yend; ay += ychange)
        {
            for (int ax = xstart; ax != xend; ax += xchange)
            {
                /* Current (best) guess. */
                //                int v = (*ann)[ay][ax];
                int v = ann.at<int>(ay, ax);
                int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
                //                int dbest = (*annd)[ay][ax];
                int dbest = annd.at<int>(ay, ax);

                /* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
                if ((unsigned) (ax - xchange) < (unsigned) aew)
                {
                    //                    int vp = (*ann)[ay][ax-xchange];
                    int vp = ann.at<int>(ay, ax - xchange);

                    int xp = INT_TO_X(vp) + xchange, yp = INT_TO_Y(vp);
                    if ((unsigned) xp < (unsigned) bew)
                    {
                        improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                    }
                }

                if ((unsigned) (ay - ychange) < (unsigned) aeh)
                {
                    //                    int vp = (*ann)[ay-ychange][ax];
                    int vp = ann.at<int>(ay - ychange, ax);
                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
                    if ((unsigned) yp < (unsigned) beh)
                    {
                        improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                    }
                }

                /* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
                int rs_start = rs_max;
                if (rs_start > MAX(b.rows, b.cols))
                {
                    rs_start = MAX(b.rows, b.cols);
                }
                for (int mag = rs_start; mag >= 1; mag /= 2)
                {
                    /* Sampling window */
                    int xmin = MAX(xbest-mag, 0), xmax = MIN(xbest+mag+1,bew);
                    int ymin = MAX(ybest-mag, 0), ymax = MIN(ybest+mag+1,beh);
                    int xp = xmin+rand()%(xmax-xmin);
                    int yp = ymin+rand()%(ymax-ymin);
                    improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                }

                ann.at<int>(ay, ax) = XY_TO_INT(xbest, ybest);
                annd.at<int>(ay, ax) = dbest;
            }
        }
    }
}

void BDSCompleteness(cv::Mat img, cv::Mat nn, cv::Mat nnd, cv::Mat &cohSumMat)
{
#pragma omp parallel for schedule(static, 8)
    for(int y = 0; y < img.rows; y++)
    {
        for(int x = 0; x < img.cols; x++)
        {
            int v = nn.at<int>(y, x);
            int xoffset = INT_TO_X(v);
            int yoffset = INT_TO_Y(v);
            //            std::cout<<"y:"<<y<<" x:" <<x<<"  yoffset:"<<yoffset<<"  xoffset:"<<xoffset<<std::endl;
            if(v == -1)
            {
                continue;
            }

            for(int dy = 0; dy < patch_w; dy++)
            {
                for(int dx = 0; dx < patch_w; dx++)
                {
                    if((yoffset + dy) < 0 || (yoffset + dy) > (img.rows - 1) || (xoffset + dx) < 0 || (xoffset + dx) > (img.cols - 1)
                            || (y + dy) < 0 || (y + dy) > (img.rows - 1) || (x + dx )< 0 ||  (x+ dx) > (img.cols - 1))
                    {
                        continue;
                    }
                    cohSumMat.at<cv::Vec4f>(y+dy, x+dx)[0] += img.at<cv::Vec3b>(yoffset+dy, xoffset+dx)[0]/255.0f;
                    cohSumMat.at<cv::Vec4f>(y+dy, x+dx)[1] += img.at<cv::Vec3b>(yoffset+dy, xoffset+dx)[1]/255.0f;
                    cohSumMat.at<cv::Vec4f>(y+dy, x+dx)[2] += img.at<cv::Vec3b>(yoffset+dy, xoffset+dx)[2]/255.0f;
                    cohSumMat.at<cv::Vec4f>(y+dy, x+dx)[3] += 1;

                }
            }
        }
    }

}

void BDSCoherence(cv::Mat img, cv::Mat nn, cv::Mat nnd, cv::Mat &cohSumMat)
{
#pragma omp parallel for schedule(static,8)
    for(int y = 0; y < img.rows; y++)
    {
        for(int x = 0; x < img.cols; x++)
        {
            //compelsss
            int v = nn.at<int>(y, x);
            if(v == -1 )
            {
                continue;
            }
            int xoffset = INT_TO_X(v);
            int yoffset = INT_TO_Y(v);
            for(int dy = 0; dy < patch_w; dy++)
            {
                for(int dx = 0; dx < patch_w; dx++)
                {
                    if((yoffset + dy) < 0 || (yoffset + dy) > (img.rows - 1) || (xoffset + dx) < 0 || (xoffset + dx) > (img.cols-1)
                            || (y + dy) < 0 || (y + dy) > (img.rows-1) || (x + dx )< 0 ||  (x + dx) > (img.cols - 1))
                    {
                        continue;
                    }
                    cohSumMat.at<cv::Vec4f>(yoffset + dy, xoffset+ dx)[0] += img.at<cv::Vec3b>(y + dy, x + dx)[0]/255.0f;
                    cohSumMat.at<cv::Vec4f>(yoffset + dy, xoffset+ dx)[1] += img.at<cv::Vec3b>(y + dy, x + dx)[1]/255.0f;
                    cohSumMat.at<cv::Vec4f>(yoffset + dy, xoffset+ dx)[2] += img.at<cv::Vec3b>(y + dy, x + dx)[2]/255.0f;
                    cohSumMat.at<cv::Vec4f>(yoffset + dy, xoffset+ dx)[3] += 1;
                }
            }
        }
    }
}

void patchmatchBySeams(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd, std::vector<std::vector<cv::Point2f> >  contours)
{
    int aew = a.cols - patch_w + 1, aeh = a.rows - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
    int bew = b.cols - patch_w + 1, beh = b.rows - patch_w + 1;
    cv::Mat  bitmat = cv::Mat(a.rows, a.cols, CV_16UC1, cv::Scalar(0));

    for (int ay = 0; ay < aeh; ay++)
    {
        for (int ax = 0; ax < aew; ax++)
        {
            cv::Point2f pt(ax, ay);
            bool  pmflag = false;
            for(int co_idx = 0; co_idx< contours.size(); co_idx++)
            {
                float kk = pointPolygonTest(contours[co_idx], pt, true);
                if(std::abs(kk) < 50)//只要离一个轮廓小于40个像素那么就需要合成
                {
                    pmflag = true;
                    break;
                }
            }

            if(pmflag == true)
            {
                int bx = rand()%bew;
                int by = rand()%beh;
                ann.at<int>(ay, ax) = XY_TO_INT(bx, by);
                annd.at<int>(ay,ax) = dist(a, b, ax, ay, bx, by);
                bitmat.at<ushort>(ay, ax) = 1; //  需要合成
            }
            else//不需要合成
            {
                ann.at<int>(ay, ax) = XY_TO_INT(ax, ay);
                annd.at<int>(ay,ax) = dist(a, b, ax, ay, ax, ay);
                bitmat.at<ushort>(ay, ax) = 0; //  不需要合成

            }

        }
    }

    for (int iter = 0; iter < pm_iters; iter++)
    {
        /* In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. */
        int ystart = 0, yend = aeh, ychange = 1;
        int xstart = 0, xend = aew, xchange = 1;
        if (iter % 2 == 1)
        {
            xstart = xend-1; xend = -1; xchange = -1;
            ystart = yend-1; yend = -1; ychange = -1;
        }
        for (int ay = ystart; ay != yend; ay += ychange)
        {
            for (int ax = xstart; ax != xend; ax += xchange)
            {

                ushort  bitflag = bitmat.at<ushort>(ay, ax);
                if(bitflag == 0)//有效区域以外不需要合成。直接使用原图。
                {
                    continue;
                }
                /* Current (best) guess. */
                //                int v = (*ann)[ay][ax];
                int v = ann.at<int>(ay, ax);
                int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
                //                int dbest = (*annd)[ay][ax];
                int dbest = annd.at<int>(ay, ax);

                /* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
                if ((unsigned) (ax - xchange) < (unsigned) aew)
                {
                    //                    int vp = (*ann)[ay][ax-xchange];
                    int vp = ann.at<int>(ay, ax - xchange);

                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp);
                    if ((unsigned) xp < (unsigned) bew)
                    {
                        improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                    }
                }

                if ((unsigned) (ay - ychange) < (unsigned) aeh)
                {
                    //                    int vp = (*ann)[ay-ychange][ax];
                    int vp = ann.at<int>(ay - ychange, ax);
                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp);
                    if ((unsigned) yp < (unsigned) beh)
                    {
                        improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                    }
                }

                /* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
                int rs_start = rs_max;
                if (rs_start > MAX(b.rows, b.cols))
                {
                    rs_start = MAX(b.rows, b.cols);
                }
                for (int mag = rs_start; mag >= 1; mag /= 2)
                {
                    /* Sampling window */
                    int xmin = MAX(xbest-mag, 0), xmax = MIN(xbest+mag+1,bew);
                    int ymin = MAX(ybest-mag, 0), ymax = MIN(ybest+mag+1,beh);
                    int xp = xmin+rand()%(xmax-xmin);
                    int yp = ymin+rand()%(ymax-ymin);
                    improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                }

                ann.at<int>(ay, ax) = XY_TO_INT(xbest, ybest);
                annd.at<int>(ay, ax) = dbest;
            }
        }
    }

}

void patchmatchBySeamsDistacne(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd, cv::Mat seaminfo)
{
    int aew = a.cols - patch_w + 1, aeh = a.rows - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
    int bew = b.cols - patch_w + 1, beh = b.rows - patch_w + 1;
    cv::Mat  bitmat = cv::Mat(a.rows, a.cols, CV_16UC1, cv::Scalar(0));

    for (int ay = 0; ay < aeh; ay++)
    {
        for (int ax = 0; ax < aew; ax++)
        {
            cv::Point2f pt(ax, ay);
//            bool  pmflag = false;
//            for(int co_idx = 0; co_idx< contours.size(); co_idx++)
//            {
//                float kk = pointPolygonTest(contours[co_idx], pt, true);
//                if(std::abs(kk) < 50)//只要离一个轮廓小于40个像素那么就需要合成
//                {
//                    pmflag = true;
//                    break;
//                }
//            }
            float  mindist = seaminfo.at<cv::Vec3f>(ay, ax)[0];

            if(std::abs(mindist) < 50)
            {
                int bx = rand()%bew;
                int by = rand()%beh;
                ann.at<int>(ay, ax) = XY_TO_INT(bx, by);
                annd.at<int>(ay,ax) = dist(a, b, ax, ay, bx, by);
                bitmat.at<ushort>(ay, ax) = 1; //  需要合成
            }
            else//不需要合成
            {
                ann.at<int>(ay, ax) = XY_TO_INT(ax, ay);
                annd.at<int>(ay,ax) = dist(a, b, ax, ay, ax, ay);
                bitmat.at<ushort>(ay, ax) = 0; //  不需要合成

            }

        }
    }

    for (int iter = 0; iter < pm_iters; iter++)
    {
        /* In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. */
        int ystart = 0, yend = aeh, ychange = 1;
        int xstart = 0, xend = aew, xchange = 1;
        if (iter % 2 == 1)
        {
            xstart = xend-1; xend = -1; xchange = -1;
            ystart = yend-1; yend = -1; ychange = -1;
        }
        for (int ay = ystart; ay != yend; ay += ychange)
        {
            for (int ax = xstart; ax != xend; ax += xchange)
            {

                ushort  bitflag = bitmat.at<ushort>(ay, ax);
                if(bitflag == 0)//有效区域以外不需要合成。直接使用原图。
                {
                    continue;
                }
                /* Current (best) guess. */
                //                int v = (*ann)[ay][ax];
                int v = ann.at<int>(ay, ax);
                int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
                //                int dbest = (*annd)[ay][ax];
                int dbest = annd.at<int>(ay, ax);

                /* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
                if ((unsigned) (ax - xchange) < (unsigned) aew)
                {
                    //                    int vp = (*ann)[ay][ax-xchange];
                    int vp = ann.at<int>(ay, ax - xchange);

                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp);
                    if ((unsigned) xp < (unsigned) bew)
                    {
                        improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                    }
                }

                if ((unsigned) (ay - ychange) < (unsigned) aeh)
                {
                    //                    int vp = (*ann)[ay-ychange][ax];
                    int vp = ann.at<int>(ay - ychange, ax);
                    int xp = INT_TO_X(vp), yp = INT_TO_Y(vp);
                    if ((unsigned) yp < (unsigned) beh)
                    {
                        improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                    }
                }

                /* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
                int rs_start = rs_max;
                if (rs_start > MAX(b.rows, b.cols))
                {
                    rs_start = MAX(b.rows, b.cols);
                }
                for (int mag = rs_start; mag >= 1; mag /= 2)
                {
                    /* Sampling window */
                    int xmin = MAX(xbest-mag, 0), xmax = MIN(xbest+mag+1,bew);
                    int ymin = MAX(ybest-mag, 0), ymax = MIN(ybest+mag+1,beh);
                    int xp = xmin+rand()%(xmax-xmin);
                    int yp = ymin+rand()%(ymax-ymin);
                    improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                }

                ann.at<int>(ay, ax) = XY_TO_INT(xbest, ybest);
                annd.at<int>(ay, ax) = dbest;
            }
        }
    }

}
void PatchMatchWithGPU(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd)
{
    patchmatchGPU(a, b, ann, annd, patch_w, pm_iters);
//    cv::imwrite("kk.png", ann);
}

void patchmatchwithSearchArea(cv::Mat a, cv::Mat b, cv::Mat &ann, cv::Mat &annd, float factor)
{
#pragma omp parallel
    {
        int searchw = int(factor*a.cols);
        int searchh = int(factor*a.rows);


        int aew = a.cols - patch_w + 1, aeh = a.rows - patch_w + 1;       /* Effective width and height (possible upper left corners of patches). */
        int bew = b.cols - patch_w + 1, beh = b.rows - patch_w + 1;

        for (int ay = 0; ay < aeh; ay++)
        {
            for (int ax = 0; ax < aew; ax++)
            {
                //            int bx = rand()%bew;
                //            int by = rand()%beh;
                //            int bx = rand()%searchw + ax - searchw;
                //            if(bx > a.cols - 1 || bx < 0)
                //            {
                //                bx = ax;
                //            }
                //            int by = rand()%searchh + ay - searchh;
                //            if(by > a.rows - 1 || by < 0)
                //            {
                //                by = ay;
                //            }

                int bx = ax;
                int by = ay;
                ann.at<int>(ay, ax) = XY_TO_INT(bx, by);
                annd.at<int>(ay,ax) = dist(a, b, ax, ay, bx, by);
            }
        }

        for (int iter = 0; iter < pm_iters; iter++)
        {
            /* In each iteration, improve the NNF, by looping in scanline or reverse-scanline order. */
            int ystart = 0, yend = aeh, ychange = 1;
            int xstart = 0, xend = aew, xchange = 1;
            if (iter % 2 == 1)
            {
                xstart = xend-1; xend = -1; xchange = -1;
                ystart = yend-1; yend = -1; ychange = -1;
            }

            for(int ay = ystart; ay != yend; ay += ychange)
            {
                for (int ax = xstart; ax != xend; ax += xchange)
                {
                    /* Current (best) guess. */
                    //                int v = (*ann)[ay][ax];
                    int v = ann.at<int>(ay, ax);
                    int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
                    //                int dbest = (*annd)[ay][ax];
                    int dbest = annd.at<int>(ay, ax);

                    /* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
                    if ((unsigned) (ax - xchange) < (unsigned) aew)
                    {
                        //                    int vp = (*ann)[ay][ax-xchange];
                        int vp = ann.at<int>(ay, ax - xchange);

                        int xp = INT_TO_X(vp) + xchange , yp = INT_TO_Y(vp);
                        if ((unsigned) xp < (unsigned) bew)
                        {
                            improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                        }
                    }

                    if ((unsigned) (ay - ychange) < (unsigned) aeh)
                    {
                        //                    int vp = (*ann)[ay-ychange][ax];
                        int vp = ann.at<int>(ay - ychange, ax);
                        int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
                        if ((unsigned) yp < (unsigned) beh)
                        {
                            improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);
                        }
                    }

                    int rs_start = MAX(searchw, searchh);

                    for (int mag = rs_start; mag >= 1; mag /= 2)
                    {
                        int xmin = MAX(xbest-mag, 0), xmax = MIN(xbest+mag+1,bew);
                        int ymin = MAX(ybest-mag, 0), ymax = MIN(ybest+mag+1,beh);

                        int xp = xmin+rand()%(xmax-xmin);
                        int yp = ymin+rand()%(ymax-ymin);

                        improve_guess(a, b, ax, ay, xbest, ybest, dbest, xp, yp);

                    }
                    ann.at<int>(ay, ax) = XY_TO_INT(xbest, ybest);
                    annd.at<int>(ay, ax) = dbest;
                }
            }
        }
    }
}


}//end namespace

