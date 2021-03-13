/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */

#include "texturing.h"
#include "edge_node.h"
#include "patchmatch/patchmatch.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <texture_synthetise.h>
#include <mve/image_io.h>
#include <util/timer.h>

#define patch_size 7

int planecolor[10][3]={{255,0,0},//红
                       {255,255,255},//白
                       {0,255,0},//     蓝
                       {0,0,255},//绿
                       {255,0,255},//洋红
                       {0,255,255},//青色
                       {255, 255, 0},//纯黄
                       {255,20,100},//深粉色
                       {112,128,144},//石板灰
                       {139,69,19}};// 马鞍棕色

TEX_NAMESPACE_BEGIN

bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
    //Number of key points
    int N = key_point.size();

    //构造矩阵X
    cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            for (int k = 0; k < N; k++)
            {
                X.at<double>(i, j) = X.at<double>(i, j) +
                        std::pow(key_point[k].x, i + j);
            }
        }
    }

    //构造矩阵Y
    cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    for (int i = 0; i < n + 1; i++)
    {
        for (int k = 0; k < N; k++)
        {
            Y.at<double>(i, 0) = Y.at<double>(i, 0) +
                    std::pow(key_point[k].x, i) * key_point[k].y;
        }
    }

    A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
    //求解矩阵A
    cv::solve(X, Y, A, cv::DECOMP_LU);
    return true;
}


void myDrawContours(cv::Mat img, std::vector<std::vector<cv::Point2f> >  contours, std::string name)
{
    //    std::vector<std::vector<cv::Point2f > > contours;
    //    contours.push_back(contour);
    cv::Mat  newimg = img.clone();

    for( int i = 0; i < contours.size(); i++)
        //        for( int i = 0; i < 1; i++)
    {
        std::vector<cv::Point> points;
        std::vector<cv::Point2f> contour = contours[i];
        for(int c_ix = 0; c_ix< contour.size(); c_ix++)
        {
            cv::Point2f  p = contour[c_ix];
            cv::Point pp;
            pp.x = int(p.x);
            pp.y = int(p.y);
            //            std::cout<<"--x:"<<pp.x<<" y:"<<pp.y<<std::endl;
            points.push_back(pp);
        }
        //        std::cout<<"i:"<<i<<"  "<<planecolor[i%10][0]<<"  "<<planecolor[i%10][0]<<"  "<<planecolor[i%10][0]<<std::endl;
//        cv::polylines(img, points, true, cv::Scalar(planecolor[i%10][0], planecolor[i%10][1], planecolor[i%10][2]), 2, 8);
        cv::polylines(newimg, points, true, cv::Scalar(255, 0, 0), 2, 8);

    }
    cv::imwrite(name, newimg);
}
void drawViewCountor(std::vector<TextureView>  &texture_views)
{
    for(int view_id = 0; view_id < texture_views.size(); view_id++)
    {
        TextureView  &texture_view =  texture_views.at(view_id);//当前chart对应的视口
        if(texture_view.seamCounters.size() == 0)//
        {
            continue;
        }
        cv::Mat  newimg = texture_view.sourceImage.clone();


        for(int s_idx = 0; s_idx < texture_view.seamCounters.size(); s_idx++)
        {
            std::vector<cv::Point2f>  contour = texture_view.contours[s_idx];
//            for( int i = 0; i < contours.size(); i++)
            {
                std::vector<cv::Point> points;
//                std::vector<cv::Point2f> contour = contours[i];
                for(int c_ix = 0; c_ix< contour.size(); c_ix++)
                {
                    cv::Point2f  p = contour[c_ix];
                    cv::Point pp;
                    pp.x = int(p.x);
                    pp.y = int(p.y);
                    //            std::cout<<"--x:"<<pp.x<<" y:"<<pp.y<<std::endl;
                    points.push_back(pp);
                }
                //        std::cout<<"i:"<<i<<"  "<<planecolor[i%10][0]<<"  "<<planecolor[i%10][0]<<"  "<<planecolor[i%10][0]<<std::endl;
        //        cv::polylines(img, points, true, cv::Scalar(planecolor[i%10][0], planecolor[i%10][1], planecolor[i%10][2]), 2, 8);
                cv::polylines(newimg, points, true, cv::Scalar(255, 0, 0), 2, 8);

            }
        }
        char buf[256];
        sprintf(buf,"viewcounter%02d.png",view_id);
        cv::imwrite(buf, newimg);


    }

}

void texturepatchseamless(UniGraph const  & graph, mve::TriangleMesh::ConstPtr mesh,
                          std::vector<TextureView>* texture_views, mve::MeshInfo const & mesh_info,
                          std::vector<myFaceInfo>   &faceInfoList, std::vector<std::vector<std::size_t> > &subgraphs,
                          tex::VertexProjectionInfos  &vertexinfos, tex::VertexProjectionInfos  &edgevertexinfos
                          , std::string const & indir)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();

    for(int i = 0; i< texture_views->size(); i++)
    {
        //        std::cout<<"----------view:"<<i<<std::endl;
        TextureView  &texture_view =  texture_views->at(i);
        //        texture_view.initmaskimage();
        char buf[256];
        sprintf(buf,"%s/color_%02d.jpg", indir.c_str(),i);
        //初始化原图和目标图，准备图像合成
        texture_view.sourceImage = cv::imread(buf);
        texture_view.targeImage = texture_view.sourceImage.clone();
        texture_view.width = texture_view.sourceImage.cols;
        texture_view.height = texture_view.sourceImage.rows;

        texture_view.referenceImage= texture_view.sourceImage.clone();
        texture_view.level = 0;

        tex::generateDepthImageByRayCasting(mesh, mesh_info,  texture_view);//利用光线追踪生成深度图，防止深度图与彩色图像分辨率不一致
    }

    std::vector<std::vector<int> > edges_vertex_in_order;//边界上所有顶点的索引
    edges_vertex_in_order.resize(edgevertexinfos.size());

#pragma omp parallel for
    for(int i = 0; i < edgevertexinfos.size(); i++)
    {
        std::vector<std::vector<VertexProjectionInfo> >  multil_order_verter_infos;//当前chart上所有的轮廓线段，中间断裂的需要分段计算
        std::vector<VertexProjectionInfo>  edgevertices = edgevertexinfos.at(i);//当前chart上所有的边界点。

        if(edgevertices.size() <= 0)
        {
            continue;
        }
        VertexProjectionInfo  v_info;

        int chartlabel = edgevertices[0].v_label;
        TextureView  &texture_view =  texture_views->at(chartlabel - 1);
//        std::cout<<"-----------chart label:"<<chartlabel-1<<std::endl;

        std::map<int, bool>   processflag;
        std::vector<int>         headlist;
        for(int v_idx = 0; v_idx < edgevertices.size(); v_idx++)
        {
            VertexProjectionInfo n_info = edgevertices[v_idx];
            if(n_info.isModelBoundaryVer == true)
            {
                headlist.push_back(v_idx);
            }
        }
//        std::cout<<"-------------headlist seze:"<<headlist.size()<<std::endl;

        if(headlist.size() == 0)
        {
//
            int h_id = 0;
            std::vector<VertexProjectionInfo>  order_verter_info;
            for(int e_idx = 0; e_idx < edgevertices.size(); e_idx++)
            {
                VertexProjectionInfo tem_vert = edgevertices[e_idx];
                if(tem_vert.adj_charts.size() > 1)
                {
                    h_id = e_idx;
                    break;
                }
            }
            v_info = edgevertices[h_id];
            VertexProjectionInfo h_info = v_info;
            order_verter_info.push_back(v_info);
            processflag[h_id] = true;
            bool continuflag = true;
            while(continuflag == true)
            {
                int v_idx = 0;
                for(; v_idx < edgevertices.size(); v_idx++)
                {
                    VertexProjectionInfo   n_info = edgevertices[v_idx];
                    if(processflag.count(v_idx) != 0)
                    {
//                        std::cout<<"---------1-------"<<std::endl;
                        continue;
                    }
//                    std::cout<<"------2-----------"<<n_info.v_label<<std::endl;

                    bool flag = false;
                    int  tempindex = n_info.edge_verts.size() - 1;
                    for(int f_idx = 0; f_idx < n_info.edge_verts.size(); f_idx++)
                    {
                        if(n_info.edge_verts[f_idx] == v_info.vertex_id)
                        {
                            std::vector<std::size_t>  edge_faces;
                            mesh_info.get_faces_for_edge(n_info.vertex_id, v_info.vertex_id, &edge_faces);
                            std::size_t  face_id1 = edge_faces[0];
                            std::size_t  face_id2 = edge_faces[1];
                            std::size_t  face_label1 = graph.get_label(face_id1);//label
                            std::size_t  face_label2 = graph.get_label(face_id2);//label

                            if(face_label1 == face_label2)
                            {
                                continue;
                            }
                            flag = true;
                            break;
                        }
                        else
                        {
                            tempindex = f_idx;
                        }
                    }

                    if(flag == true)
                    {

                        if(order_verter_info.size() == edgevertices.size() - 1 || (n_info.edge_verts[tempindex] == h_info.vertex_id && order_verter_info.size() > edgevertices.size()/2))
                        {
                            processflag[v_idx] = true;
                            v_info = n_info;
                            order_verter_info.push_back(n_info);
                            continuflag = false;
                            break;
                        }

                        processflag[v_idx] = true;
                        v_info = n_info;
                        order_verter_info.push_back(n_info);

                    }//end flag == true

                }//end for v_idx

            }
//            std::cout<<"------->size:"<<order_verter_info.size()<<std::endl;

            //            texture_view.seamCounters.push_back(order_verter_info);
            multil_order_verter_infos.push_back(order_verter_info);
        }
        else
        {
//            std::cout<<"------no loop"<<std::endl;
            std::map<int, bool>  headprocessflag;
            for(int h_idx = 0; h_idx < headlist.size(); h_idx++)
            {
                //                std::cout<<"---------h_idx:"<<h_idx<<"/"<<headlist.size()<<std::endl;
                int h_id = headlist[h_idx];
                if(headprocessflag.count(h_idx) != 0)
                {
                    continue;
                }

                headprocessflag[h_idx] = true;
                std::vector<VertexProjectionInfo>  order_verter_info;
                v_info = edgevertices[h_id];
                //                order_verter_ids.push_back(v_info.vertex_id);//把头节点放入线段队列
                order_verter_info.push_back(v_info);
                processflag[h_id] = true;

                bool continuflag = true;
                while(continuflag == true)
                {
                    int v_idx = 0;
                    for(; v_idx < edgevertices.size(); v_idx++)
                    {
                        VertexProjectionInfo n_info = edgevertices[v_idx];
                        if(processflag.count(v_idx) != 0)
                        {
                            continue;
                        }
                        bool flag = false;

                        for(int f_idx = 0; f_idx < n_info.edge_verts.size(); f_idx++)
                        {
                            if(n_info.edge_verts[f_idx] == v_info.vertex_id)//找到想邻的边界顶点
                            {
                                flag = true;
                                break;
                            }
                        }

                        if(flag == true)
                        {
                            processflag[v_idx] = true;
                            v_info = n_info;
                            //                            order_verter_ids.push_back(n_info.vertex_id);
                            order_verter_info.push_back(n_info);


                            if(v_info.isModelBoundaryVer == true)
                            {

                                continuflag = false;
                                for(int h_idx = 0; h_idx < headlist.size(); h_idx++)
                                {
                                    int h_id = headlist[h_idx];
                                    if(h_id == v_idx)
                                    {
                                        headprocessflag[h_idx] = true;
                                    }
                                }
                            }
                            break;
                        }
                    }

                    if(v_idx == edgevertices.size())
                    {
                        continuflag = false;
                    }
                }

//                std::cout<<"----seg size:"<<order_verter_info.size()<<std::endl;
                //                multil_order_verter_ids.push_back(order_verter_ids);
                //                texture_view.seamCounters.push_back(order_verter_info);
                multil_order_verter_infos.push_back(order_verter_info);
            }
        }//end no chart loop


        //        edges_vertex_in_order.at(i) = order_verter_ids;



        for(int c_idx = 0; c_idx < multil_order_verter_infos.size(); c_idx++)
        {
            std::vector<VertexProjectionInfo>   order_verter_ids = multil_order_verter_infos[c_idx];//记录所有的边界节点

            VertexProjectionInfo  h_info = order_verter_ids[0];

            //countour
            std::vector<cv::Point2f>  contours;
            math::Vec3f  pos = vertices[h_info.vertex_id];
            math::Vec2f pixel = texture_view.get_pixel_coords_noshift(pos);
            if(pixel[0] >= 0 && pixel[0] <= (texture_view.get_width() - 1)
                    && pixel[1] >= 0 && pixel[1] <= (texture_view.get_height() - 1))
            {
                cv::Point2f point(pixel[0], pixel[1]);
                contours.push_back(point);
            }

            //vertex
            std::vector<VertexProjectionInfo>   seg_order_verter_ids;
            //            int tp_label = h_info.v_twopass_label;

            std::vector<std::size_t> adj_charts = h_info.adj_charts;

            seg_order_verter_ids.push_back(h_info);

            for(int v_idx = 1; v_idx < order_verter_ids.size(); v_idx++)
            {
                VertexProjectionInfo  info = order_verter_ids[v_idx];
                if(info.adj_charts.size() > 1)
                {
                    seg_order_verter_ids.push_back(info);
                    std::vector<VertexProjectionInfo>   seg_reverse_verter_ids;
                    seg_reverse_verter_ids = seg_order_verter_ids;
                    std::reverse(seg_reverse_verter_ids.begin(), seg_reverse_verter_ids.end());
                    seg_order_verter_ids.insert(seg_order_verter_ids.begin(), seg_reverse_verter_ids.begin(), seg_reverse_verter_ids.end());//连接两个向量
                    if(seg_order_verter_ids.size() >= 3)
                    {
                        texture_view.seamCounters.push_back(seg_order_verter_ids);
                    }

                    //更新信息
                    //                    tp_label = info.v_twopass_label;
                    seg_order_verter_ids.clear();
                    seg_order_verter_ids.push_back(info);

                    //                    contours.clear();
                    //countour
                    cv::Point2f point2;
                    math::Vec3f  pos2 = vertices[info.vertex_id];
                    math::Vec2f pixel2 = texture_view.get_pixel_coords_noshift(pos2);
                    if(pixel2[0] >= 0 && pixel2[0] <= (texture_view.get_width() - 1)
                            && pixel2[1] >= 0 && pixel2[1] <= (texture_view.get_height() - 1))
                    {
                        //                        cv::Point2f point2(pixel2[0], pixel2[1]);
                        //                        contours.push_back(point2);
                        point2.x = pixel2[0];
                        point2.y = pixel2[1];
                    }

                    //copy contours
                    contours.push_back(point2);
                    std::vector<cv::Point2f>  contours_inverse;
                    contours_inverse = contours;
                    std::reverse(contours_inverse.begin(), contours_inverse.end());
                    contours.insert(contours.begin(), contours_inverse.begin(), contours_inverse.end());
//                    if(contours.size() > 2)
//                    {
                        texture_view.contours.push_back(contours);

                        contours.clear();
                        contours.push_back(point2);
//                    }
                }
                else
                {
                    seg_order_verter_ids.push_back(info);

                    //countour
                    math::Vec3f  pos2 = vertices[info.vertex_id];
                    math::Vec2f pixel2 = texture_view.get_pixel_coords_noshift(pos2);
                    if(pixel2[0] >= 0 && pixel2[0] <= (texture_view.get_width() - 1)
                            && pixel2[1] >= 0 && pixel2[1] <= (texture_view.get_height() - 1))
                    {
                        cv::Point2f point2(pixel2[0], pixel2[1]);
                        contours.push_back(point2);
                    }
                }

            }

            if(seg_order_verter_ids.size() <3)
            {
                continue;
            }
            std::vector<VertexProjectionInfo>   seg_reverse_verter_ids;//为了构建轮廓复制一个副本形成闭环
            seg_reverse_verter_ids = seg_order_verter_ids;//拷贝一份
            std::reverse(seg_reverse_verter_ids.begin(), seg_reverse_verter_ids.end());//翻转
            seg_order_verter_ids.insert(seg_order_verter_ids.begin(), seg_reverse_verter_ids.begin(), seg_reverse_verter_ids.end());//连接两个向量

            texture_view.seamCounters.push_back(seg_order_verter_ids);


            //copy contours
            std::vector<cv::Point2f>  contours_inverse;
            contours_inverse = contours;
            std::reverse(contours_inverse.begin(), contours_inverse.end());
            contours.insert(contours.begin(), contours_inverse.begin(), contours_inverse.end());
            texture_view.contours.push_back(contours);

        }

//        char buf[256];
//        sprintf(buf,"contour%02d.png", i);
//        myDrawContours(texture_view.sourceImage, texture_view.contours, buf);
    }

//    drawViewCountor(texture_views);

    std::cout<<"----------generate pyramid"<<std::endl;
    int scaleNum = 3;
    std::vector<std::vector<TextureView> >  PyramidViews;
    PyramidViews.resize(texture_views->size());
    for(int level = 0; level < scaleNum; level++)
    {
        std::cout<<"-------level:"<<level<<std::endl;

        for(int i = 0; i<texture_views->size(); i++)
        {
            std::vector<TextureView> &myViews = PyramidViews[i];
            int img_level = myViews.size();
            TextureView &textureview = texture_views->at(i);
            int currentlabel = i + 1;

            if(level == 0)//
            {

//                textureview.seamInfoImg = cv::Mat(G2LTexConfig::get().IMAGE_HEIGHT, G2LTexConfig::get().IMAGE_WIDTH, CV_32FC3, cv::Scalar( G2LTexConfig::get().IMAGE_WIDTH,-1,-1));
                textureview.seamInfoImg = new PixelInfoMatrix(G2LTexConfig::get().IMAGE_HEIGHT, G2LTexConfig::get().IMAGE_WIDTH);

                if(textureview.seamCounters.size() != 0)
                {
#pragma omp parallel for
                    for(int p_y = 0; p_y < textureview.height; p_y++)
                    {
                        for(int p_x = 0; p_x < textureview.width; p_x ++)
                        {

                            cv::Point2f pt(p_x, p_y);

                            PixelContourDistStr  pixelinfo;

                            for(int s_idx = 0; s_idx < textureview.seamCounters.size(); s_idx++)
                            {
                                std::vector<cv::Point2f>  countour = textureview.contours[s_idx];
                                if(countour.size() < 10)
                                {
                                    continue;
                                }

                                std::vector<VertexProjectionInfo>  vert_infos = textureview.seamCounters[s_idx];//缝隙上的所有顶点信息

                                VertexProjectionInfo v_info1 = vert_infos[vert_infos.size()/2];
                                VertexProjectionInfo v_info2 = vert_infos[vert_infos.size()/2+1];

                                //                        int current_label = v_info.v_label;
                                //                        int tp_label = v_info.v_twopass_label;
                                std::vector<std::size_t>  edge_faces;
                                mesh_info.get_faces_for_edge(v_info1.vertex_id, v_info2.vertex_id, &edge_faces);//取得公共边两边的两个面
                                if(edge_faces.size() != 2)
                                {
                                    continue;
                                }
                                int label1 = graph.get_label(edge_faces[0]);
                                int label2 = graph.get_label(edge_faces[1]);
        //                            std::cout<<"current label:"<<currentlabel<<"------->label1:"<<label1<<" label2:"<<label2<<std::endl;

                                int adj_label = -1;
                                if(label1 == currentlabel)
                                {
                                    adj_label = label2;
                                }
                                else if(label2 == currentlabel)//
                                {
                                    adj_label = label1;
                                }

                                if(adj_label == -1)
                                {
                                    continue;
                                }


                                float kk = pointPolygonTest(countour, pt, true);

                                if(std::abs(kk) < 50)
                                {
                                    pixelinfo.adjLabel.push_back(adj_label);
                                    pixelinfo.adjContourDist.push_back(std::abs(kk));
                                }

                            }
                            //                        std::cout<<"-------------------2------------------------"<<std::endl;

                            //label
                            pixelinfo.num = pixelinfo.adjLabel.size();
                            pixelinfo.curLabel = currentlabel;

#pragma omp critical
                            {
                                textureview.seamInfoImg->set(p_y, p_x, pixelinfo);
                            }

                        }
                    }
                }

                myViews.push_back(textureview);

            }
            else
            {
                TextureView newview;
                //                newview.id = textureview.id;
                newview.level = img_level;

                int div = 1 << level;
                newview.width = int(textureview.width/div);
                newview.height = int(textureview.height/div);

                newview.sourceImage = cv::Mat(newview.height, newview.width, CV_8UC3);
                cv::resize(textureview.sourceImage, newview.sourceImage, newview.sourceImage.size(), 0, 0, CV_INTER_AREA);

                newview.targeImage = cv::Mat(newview.height, newview.width, CV_8UC3);
                cv::resize(textureview.targeImage, newview.targeImage, newview.targeImage.size(), 0, 0, CV_INTER_AREA);

                newview.referenceImage = cv::Mat(newview.height, newview.width, CV_8UC3);
                cv::resize(textureview.referenceImage, newview.referenceImage, newview.referenceImage.size(), 0, 0, CV_INTER_AREA);


                math::Matrix4f mat = textureview.getWorldToCamMatrix();
                newview.setWorldToCamMatrix(tex::mathToEigen(mat));
                tex::generateDepthImageByRayCasting(mesh, mesh_info, newview);//利用光线追踪生成深度图，防止深度图与彩色图像分辨率不一致

                myViews.push_back(newview);

            }
        }
    }

//    scaleNum=1;
    for(int scaleiter = scaleNum - 1; scaleiter >= 0; scaleiter--)
    {
        std::cout<<"--------------scale:"<<scaleiter<<std::endl;
        //计算权重图，并开始纹理合成

        int iternum = 15 - 5 * (scaleNum - scaleiter - 1);
//        int iternum = 25 - 5 * (scaleNum - scaleiter - 1);
//        int iternum = 50 - 10 * (scaleNum - scaleiter - 1);
        for(int innerIter = 0; innerIter < iternum; innerIter ++)
        {
            std::cout<<"------------iter:"<<innerIter<<std::endl;

            for(int view_id = 0; view_id < texture_views->size(); view_id++)
            {
//                std::cout<<"------view id:"<<view_id<<std::endl;
                TextureView  &texture_view =  texture_views->at(view_id);
                TextureView  &pyramidview = PyramidViews[view_id][scaleiter];

                if(texture_view.seamCounters.size() == 0)
                {
                    pyramidview.targeImage = pyramidview.sourceImage.clone();
                    pyramidview.referenceImage = pyramidview.sourceImage.clone();

                    continue;
                }

                int scale = pyramidview.level;
                float div = 1 << scale;

//                std::cout<<"1----------viewid:"<<view_id<<std::endl;
                //patchmatch
                cv::Mat ann;
                cv::Mat annd;

                ann = cv::Mat(pyramidview.height, pyramidview.width, CV_32SC1, cv::Scalar(-1));
                annd = cv::Mat(pyramidview.height, pyramidview.width, CV_32SC1, cv::Scalar(0));
                PatchMatch::PatchMatchWithGPU(pyramidview.targeImage, pyramidview.sourceImage, ann, annd);

                cv::Mat bnn;
                cv::Mat bnnd;
                bnn = cv::Mat(pyramidview.height, pyramidview.width, CV_32SC1, cv::Scalar(-1));
                bnnd = cv::Mat(pyramidview.height, pyramidview.width, CV_32SC1, cv::Scalar(0));
                PatchMatch::PatchMatchWithGPU(pyramidview.sourceImage, pyramidview.targeImage,  bnn, bnnd);

                //BDS completeness
                cv::Mat  compSum = cv::Mat(pyramidview.height, pyramidview.width, CV_32FC4, cv::Scalar(0.0, 0.0, 0.0, 0.0));
                PatchMatch::BDSCompleteness(pyramidview.sourceImage, ann, annd, compSum);

                //BDS Conherence
                cv::Mat  conhSum = cv::Mat(pyramidview.height, pyramidview.width, CV_32FC4, cv::Scalar(0.0, 0.0, 0.0, 0.0));
                PatchMatch::BDSCoherence(pyramidview.sourceImage, bnn, bnnd, conhSum);

                {

                    float L = float(patch_size*patch_size);

                    //进行纹理合成
//                    util::WallTimer sysconsime;
#pragma omp parallel for schedule (static, 16)
                    for(int pidx = 0; pidx < pyramidview.height*pyramidview.width; pidx++)
                    {
                        int p_y = pidx / pyramidview.width;
                        int p_x = pidx % pyramidview.width;
                        {
                            PixelContourDistStr pixinfo = texture_view.seamInfoImg->get(int(p_y*div), int(p_x*div));

                            if(pixinfo.num == 0)
                            {
                                pyramidview.targeImage.at<cv::Vec3b>(p_y, p_x)[0] = pyramidview.sourceImage.at<cv::Vec3b>(p_y, p_x)[0];
                                pyramidview.targeImage.at<cv::Vec3b>(p_y, p_x)[1] = pyramidview.sourceImage.at<cv::Vec3b>(p_y, p_x)[1];
                                pyramidview.targeImage.at<cv::Vec3b>(p_y, p_x)[2] = pyramidview.sourceImage.at<cv::Vec3b>(p_y, p_x)[2];
                                continue;
                            }
                            else
                            {
                                //for test

                                Eigen::Vector3f mcolor;
                                mcolor(0) = pyramidview.targeImage.at<cv::Vec3b>(p_y, p_x)[0]/255.0f;
                                mcolor(1) = pyramidview.targeImage.at<cv::Vec3b>(p_y, p_x)[1]/255.0f;
                                mcolor(2) = pyramidview.targeImage.at<cv::Vec3b>(p_y, p_x)[2]/255.0f;

                                Eigen::Vector3f  sumColor(0.0, 0.0, 0.0);
                                float factor = 0.0;
                                float weight = 2.0f;

                                //遍历所有的轮廓线

                                float view_cout = 0;
                                for(int cter_idx = 0; cter_idx < pixinfo.num; cter_idx++)
                                {
                                    float mindistance = pixinfo.adjContourDist.at(cter_idx);
                                    int adjlabel = pixinfo.adjLabel.at(cter_idx);

                                    int current_label = pixinfo.curLabel;

                                    float twopasslamda = std::exp(-((std::abs(mindistance))/15.0f)*((std::abs(mindistance))/15.0f));
//                                    float twopasslamda = std::exp(-((std::abs(mindistance))/5.0f)*((std::abs(mindistance))/5.0f));

                                    float lamda = (1.0 - twopasslamda);

                                    int colorcount = 0;
                                    float cweigth = 1.0;

//                                    Eigen::Vector3f ccolor = tex::generateRGBByRemapping(mesh, PyramidViews, current_label, adjlabel, p_x, p_y,scale, colorcount);

                                    Eigen::Vector3f ccolor = tex::generateRGBAndWeightByRemapping(mesh, PyramidViews, current_label, adjlabel, p_x, p_y,scale, colorcount, cweigth);

                                    if(colorcount == 1)
                                    {
                                        sumColor(0) += cweigth*twopasslamda*ccolor(0);
                                        sumColor(1) += cweigth*twopasslamda*ccolor(1);
                                        sumColor(2) += cweigth*twopasslamda*ccolor(2);
                                        //current target
                                        sumColor(0) += cweigth*lamda*mcolor(0);
                                        sumColor(1) += cweigth*lamda*mcolor(1);
                                        sumColor(2) += cweigth*lamda*mcolor(2);
                                        view_cout += cweigth;
                                    }
                                }

                                if(view_cout > 0)
                                {

                                    sumColor(0) = weight*sumColor(0)/view_cout;
                                    sumColor(1) = weight*sumColor(1)/view_cout;
                                    sumColor(2) = weight*sumColor(2)/view_cout;
                                    factor += weight;
                                }

                                //BDS
                                math::Vec3f sumCompleteness;
                                float comCount = 0;
                                sumCompleteness[0] = compSum.at<cv::Vec4f>(p_y, p_x)[0];
                                sumCompleteness[1] = compSum.at<cv::Vec4f>(p_y, p_x)[1];
                                sumCompleteness[2] = compSum.at<cv::Vec4f>(p_y, p_x)[2];
                                comCount = compSum.at<cv::Vec4f>(p_y, p_x)[3];

                                math::Vec3f sumCoherence;
                                float cohCount = 0;
                                sumCoherence[0] = conhSum.at<cv::Vec4f>(p_y, p_x)[0];
                                sumCoherence[1] = conhSum.at<cv::Vec4f>(p_y, p_x)[1];
                                sumCoherence[2] = conhSum.at<cv::Vec4f>(p_y, p_x)[2];
                                cohCount = conhSum.at<cv::Vec4f>(p_y, p_x)[3];

                                float wCom = 0.1f;

                                float wCoh = 2.0f;

                                if(comCount > 0)
                                {
                                    factor += wCom*comCount/L;
                                    sumColor(0) += (sumCompleteness[0]/L)*wCom;
                                    sumColor(1) += (sumCompleteness[1]/L)*wCom;
                                    sumColor(2) += (sumCompleteness[2]/L)*wCom;
                                }

                                if(cohCount > 0)
                                {
                                    factor += wCoh*cohCount/L;
                                    sumColor(0) += (sumCoherence[0]/L)*wCoh;
                                    sumColor(1) += (sumCoherence[1]/L)*wCoh;
                                    sumColor(2) += (sumCoherence[2]/L)*wCoh;

                                }

#pragma omp critical
                                {
                                    pyramidview.targeImage.at<cv::Vec3b>(p_y, p_x)[0] = static_cast<uchar>(sumColor(0)*255.0f/factor);
                                    pyramidview.targeImage.at<cv::Vec3b>(p_y, p_x)[1] = static_cast<uchar>(sumColor(1)*255.0f/factor);
                                    pyramidview.targeImage.at<cv::Vec3b>(p_y, p_x)[2] = static_cast<uchar>(sumColor(2)*255.0f/factor);

                                }
                            }
                        }
                    }

//                    std::cout << "sysconsimetime:" << sysconsime.get_elapsed() << " ms" << std::endl;

                }//end seams
            }//end iter

        }//end vew


#pragma omp parallel for
        for(int view_id = 0; view_id < PyramidViews.size(); view_id++)
        {

            TextureView &imginfo = PyramidViews[view_id][scaleiter];

            if(scaleiter > 0)//initialize the high-level
            {
                int h_level = scaleiter - 1;
                TextureView &h_imginfo = PyramidViews[view_id][h_level];
                cv::resize(imginfo.targeImage, h_imginfo.targeImage, h_imginfo.targeImage.size());

                cv::resize(imginfo.referenceImage, h_imginfo.referenceImage, h_imginfo.referenceImage.size());

            }
        }

    }//end scale



    for(int view_id = 0; view_id < texture_views->size(); view_id++)
    {
        TextureView  py_texture_view =  PyramidViews[view_id][0];
        TextureView  &texture_view =  texture_views->at(view_id);

        for(int h = 0; h < G2LTexConfig::get().IMAGE_HEIGHT; h++)
        {
            for(int w = 0; w < G2LTexConfig::get().IMAGE_WIDTH; w++)
            {
                    texture_view.image->at(w, h, 0) = py_texture_view.targeImage.at<cv::Vec3b>(h, w)[2];
                    texture_view.image->at(w, h, 1) = py_texture_view.targeImage.at<cv::Vec3b>(h, w)[1];
                    texture_view.image->at(w, h, 2) = py_texture_view.targeImage.at<cv::Vec3b>(h, w)[0];
            }
        }
    }
}


void generate_chart_vertexInfo(UniGraph const & graph,
                               mve::TriangleMesh::ConstPtr mesh,
                               mve::MeshInfo const & mesh_info,
                               std::vector<std::vector<VertexProjectionInfo> > &vertex_projection_infos,
                               std::vector<std::vector<VertexProjectionInfo> > &edge_vertex_infos,
                               std::vector<tex::myFaceInfo>   faceInfoList,
                               std::vector<std::vector<std::size_t> >   subgraphs)
{
    mve::TriangleMesh::FaceList const & mesh_faces = mesh->get_faces();//得到面的索引
    mve::TriangleMesh::VertexList const & vertices = mesh->get_vertices();//到到点的索引
    vertex_projection_infos.resize(subgraphs.size());//每个chart构建一个点信息列表，包含当前chart的所有顶点信息
    edge_vertex_infos.resize(subgraphs.size());

    for(int i = 0; i < subgraphs.size(); i++)//
    {
        //        std::cout<<"---->i:"<<i<<std::endl;
        std::vector<VertexProjectionInfo>   chartVertexinfo;
        std::vector<VertexProjectionInfo>   chartEdgeVertexinfo;

        std::vector<std::size_t>  chart = subgraphs[i];
        std::map<std::size_t, bool>  vertex_processed_flag;
        for (std::size_t  face_idx = 0; face_idx < chart.size(); face_idx++)
        {
            std::size_t const face_id = chart[face_idx];
            int    label = graph.get_label(face_id);
            if(label == 0)
            {
                break;
            }
            std::size_t const v_pos = face_id * 3;
            for(std::size_t v_idx = 0; v_idx < 3; v_idx++)
            {
                std::size_t const vertex_id = mesh_faces[v_pos  + v_idx];
                if(vertex_processed_flag.count(vertex_id) != 0)
                {
                    continue;
                }
                vertex_processed_flag[vertex_id] = true;
                mve::MeshInfo::VertexInfo  v_info = mesh_info[vertex_id];

                //处理邻接面
                std::vector<std::size_t>  face_ids;
                std::vector<std::size_t>  adj_faces = v_info.faces;
                for(int fs = 0; fs < adj_faces.size(); fs++)
                {
                    std::size_t  fs_idx = adj_faces[fs];
                    std::size_t  fs_label = faceInfoList[fs_idx].lable;
                    if(fs_label == label)
                    {
                        std::vector<std::size_t>::iterator f_iter = std::find(face_ids.begin(), face_ids.end(), fs_idx);
                        if(f_iter == face_ids.end())
                        {
                            face_ids.push_back(fs_idx);
                        }
                    }
                }//end fs
                //                std::cout<<"----same label:"<<face_ids.size()<<std::endl;

                std::vector<std::size_t>  adj_chart;
                std::vector<std::size_t>  edge_verts;
                std::vector<std::size_t>  adj_verts = v_info.verts;

                bool  isChartboudary = false;
                bool  isChartedge =false;
                for(int a_idx = 0; a_idx < adj_verts.size(); a_idx++)
                {
                    std::size_t  adj_id = adj_verts[a_idx];
                    std::vector<std::size_t>  edge_faces;
                    mesh_info.get_faces_for_edge(vertex_id, adj_id, &edge_faces);


                    if(edge_faces.size() == 1)
                    {
                        isChartedge = true;
                    }
                    else if(edge_faces.size() == 2)
                    {
                        //公共边两边的两个面
                        std::size_t  face_id1 = edge_faces[0];
                        std::size_t  face_id2 = edge_faces[1];
                        std::size_t  face_label1 = graph.get_label(face_id1);//label
                        std::size_t  face_label2 = graph.get_label(face_id2);//label

                        if(face_label1 != face_label2 && face_label1 != 0 && face_label2 != 0)
                        {
                            int face_id = -1;
                            if(face_label1 == label)
                            {
                                face_id = face_id2;
                            }
                            else if(face_label2 == label)
                            {
                                face_id = face_id1;
                            }

                            if(face_id != -1)
                            {
                                int chart_id = faceInfoList[face_id].chart_id;
                                std::vector<std::size_t>::iterator c_iter = std::find(adj_chart.begin(), adj_chart.end(), chart_id);
                                if(c_iter == adj_chart.end())
                                {
                                    adj_chart.push_back(chart_id);//
                                }

                                //                                std::vector<std::size_t>::iterator e_iter = std::find(edge_verts.begin(), edge_verts.end(), chart_id);
                                //                                if(e_iter == edge_verts.end())
                                //                                {
                                edge_verts.push_back(adj_id);
                                //                                }
                                isChartboudary = true;
                            }
                            else
                            {

                            }
                        }//end face1 != face2
                        else if(face_label1 == 0 && face_label2 == label)
                        {
                            isChartedge = true;
                        }
                        else if(face_label2 == 0 && face_label1 == label)
                        {
                            isChartedge = true;
                        }
                    }




                }//end a_idx

                math::Vec2f   projection(0.0f,0.0f);
                VertexProjectionInfo info(i, vertex_id, projection, face_ids, false, adj_chart, edge_verts, label, false);
                if(isChartboudary == true)
                {
                    //                    std::cout<<"---------------------dd"<<std::endl;
                    info.isChartBoundaryV = true;
                }

                if(isChartboudary == true && isChartedge == true)
                {
                    info.isModelBoundaryVer = true;

                }

                //                std::cout<<"-------------->size:"<<adj_chart.size()<<std::endl;
                if(isChartboudary == true)
                {
                    chartEdgeVertexinfo.push_back(info);

                }

                chartVertexinfo.push_back(info);

            }//end the v_idx on the face

        }//end face_idx of chart
        vertex_projection_infos.at(i) = chartVertexinfo;
        edge_vertex_infos.at(i) = chartEdgeVertexinfo;//只保存边界上的顶点信息
    }//end i for subgraphs
}

TEX_NAMESPACE_END
