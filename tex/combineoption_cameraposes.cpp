/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */


#include "texturing.h"
#include "util.h"
TEX_NAMESPACE_BEGIN

void generateGradImg(cv::Mat&  in, cv::Mat& outx, cv::Mat& outy)
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

void combineOptionCameraPoses(UniGraph const & graph, mve::TriangleMesh::ConstPtr mesh, mve::MeshInfo const & mesh_info,
                              std::vector<TextureView> &texture_views, std::string const & indir, std::vector<myFaceInfo>   &faceInfoList,
                              std::vector<myViewImageInfo>&  viewImageList, std::vector<std::vector<std::size_t> > &patch_graph,
                              std::vector<std::vector<std::size_t> > &subgraphs, Settings const & settings)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();//所有顶点
    std::size_t  const num_faces = faces.size()/3;//面的个数
    float   w_color = 1, w_depth = 100;


    for(int i = 0; i < texture_views.size(); i++)
    {
        myViewImageInfo  info;
        info.view_id = i;
        char colorbuf[256];
        char depthbuf[256];
        sprintf(colorbuf,"%s/color_%02d.jpg",indir.c_str(), i);
        sprintf(depthbuf,"%s/depth_%02d.png",indir.c_str(), i);
        info.img = cv::imread(colorbuf);
        info.depth = cv::imread(depthbuf, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

        cv::Mat gray;
        cv::cvtColor(info.img, gray, CV_BGR2GRAY);
        generateGradImg(gray, info.gradxImg, info.gradyImg);

        viewImageList.push_back(info);

        //为每个视口读取图像，防止访问为NULL
        TextureView  &texture_view =  texture_views.at(i);
        texture_view.load_image();
        texture_view.generate_gradient_magnitude();
    }


    for(int i = 0; i < num_faces; i++)//读取每个面的信息。
    {
        myFaceInfo  finfo;
        int lable = graph.get_label(i);
        if(lable != 0)
        {
            finfo.face_id = i;
            finfo.lable = lable;
            //        finfo.world_to_cam = ;
            TextureView  texture_view =  texture_views.at(lable - 1);
            math::Matrix4f  mat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  wtc;
            wtc<<mat(0,0),mat(0,1),mat(0,2),mat(0,3),
                    mat(1,0),mat(1,1),mat(1,2),mat(1,3),
                    mat(2,0),mat(2,1),mat(2,2),mat(2,3),
                    mat(3,0),mat(3,1),mat(3,2),mat(3,3);
            finfo.world_to_cam = wtc;

            math::Vec3f   v_1 = vertices[faces[i*3]];
            math::Vec3f   v_2 = vertices[faces[i*3 + 1]];
            math::Vec3f   v_3 = vertices[faces[i*3 + 2]];


            FaceProjectionInfo info = {i, 0.0f, 0.0f, math::Vec3f(0.0f, 0.0f, 0.0f)};//view_id;  quality; mean_color;

            texture_view.get_face_info(v_1, v_2, v_3, &info, settings);
            finfo.quality = info.quality;

            //            if(info.quality > 0.5)//不需要采样。
            //            {
            //                finfo.generateSample(v_1, v_2, v_3);
            //            }
        }
        else
        {
            finfo.face_id = i;
            finfo.world_to_cam = Eigen::Matrix4f::Identity();
            finfo.lable = 0;
            finfo.quality = 0.0f;
        }

        faceInfoList.push_back(finfo);
    }


    //构建1通chart块之间的连接关系
    std::vector<int>  labelPatchCount;//记录每个标签包含的chart数量，也就可以计算下一个标签所有chart开始的索引
    std::vector<int>  twoPassLabelChartCount;//记录每个标签包含的chart数量，也就可以计算下一个标签所有chart开始的索引
    std::vector<std::vector<std::size_t> > twoPassSubgraphs;
    int startsize, twoPassStartSize;
    int endsize, twoPassEndSize;
    for(int i = 0; i < texture_views.size(); i++)
    {
        int const label = i + 1;
        //one pass
        startsize = subgraphs.size();
        graph.get_subgraphs(label, &subgraphs);
        endsize = subgraphs.size();
        labelPatchCount.push_back(endsize - startsize);

        //two pass
        twoPassStartSize = twoPassSubgraphs.size();
        graph.get_twoPassSubgraphs(label, &twoPassSubgraphs);
        twoPassEndSize = twoPassSubgraphs.size();
        twoPassLabelChartCount.push_back(twoPassEndSize - twoPassStartSize);
    }

    build_patch_adjacency_graph(graph, subgraphs, labelPatchCount, patch_graph, faceInfoList);//构建chart的邻接图
    std::vector<std::vector<std::size_t> > twopass_adj_chart_graph;//构建2通chart的邻接图
    build_twopass_adjacency_graph(graph, twoPassSubgraphs, twoPassLabelChartCount, twopass_adj_chart_graph);

    const int nvariable = 6;	// 3 for rotation and 3 for translation


    for(int iter = 0; iter < 400; iter++)//迭代次数
    {
        std::cout<<"2----------------iter:"<<iter<<std::endl;
        for(int view_idx = 0; view_idx < texture_views.size(); view_idx++)//处理所有彩色图像的相机位姿
        {
            int current_label = view_idx + 1;
            Eigen::MatrixXd   JTJ(nvariable, nvariable);//总的优化方程中的A
            Eigen::VectorXd   JTr(nvariable);//总的优化方程中的b
            JTJ.setZero();
            JTr.setZero();

            //颜色项
            Eigen::MatrixXd    c_JTJ(nvariable, nvariable);
            Eigen::MatrixXd    c_JTr(nvariable, 1);
            Eigen::MatrixXd    c_J(nvariable, 1);
            c_JTJ.setZero();
            c_JTr.setZero();

            //深度项
            Eigen::MatrixXd    d_JTJ(nvariable, nvariable);
            Eigen::MatrixXd    d_JTr(nvariable, 1);
            Eigen::MatrixXd    d_J(nvariable, 1);
            d_JTJ.setZero();
            d_JTr.setZero();


            //当前标签对应的信息
            TextureView  &texture_view =  texture_views.at(current_label - 1);
            math::Matrix4f  cmat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  currentMat = mathToEigen(cmat);
//            Eigen::Matrix4f  currentMat = faceInfoList[c_f_x].world_to_cam;//当前块的相机变换

            cv::Mat  currentImg = viewImageList[current_label - 1].img;
            cv::Mat  currentDepth = viewImageList[current_label - 1].depth;
            cv::Mat currentGradXImg = viewImageList[current_label - 1].gradxImg;//得到当前面对应视口的梯度图应用颜色一致性追踪
            cv::Mat currentGradYImg = viewImageList[current_label - 1].gradyImg;//得到当前面对应视口的梯度图应用颜色一致性追踪


            int chartnum = labelPatchCount[view_idx];//当前标签在chart列表中个数
            int chart_start = getChartIndex(labelPatchCount, current_label);//当前标签在chart列表中的起始位置

//            std::cout<<"A----cur label:"<<current_label-1<<" cur chart num:"<<chartnum<<" start:"<<chart_start<<std::endl;

            for(int chart_idx = 0; chart_idx < chartnum; chart_idx++)//遍历当前视口所有的chart块
            {
                std::vector<std::size_t> cur_charts = subgraphs[chart_start+chart_idx];//取出当前chart块

                std::vector<std::size_t>   adj_patches = patch_graph[chart_start+chart_idx];//取出当前chart块相邻块
                if(adj_patches.size() == 0)//没有邻域不考虑
                {
                    continue;
                }

//                std::cout<<"-------adj size:"<<adj_patches.size()<<std::endl;
                for(int adj_idx = 0; adj_idx < adj_patches.size(); adj_idx++)//考虑当前chart的所有邻域chart
                {
                    std::size_t  adj_chartidx = adj_patches[adj_idx];//adj patch index
                    std::vector<std::size_t>  adj_faces = subgraphs[adj_chartidx];//邻接的块中所有的面（不存在与当前块相同标签的面）

                    int    f_i = adj_faces[0];//邻接面中的面索引
                    int    adj_chart_label = graph.get_label(f_i);//邻接面中的标签

                    TextureView  adj_texture_view =  texture_views.at(adj_chart_label - 1);//邻接块对应的视口
                    //取出邻接视口对应的图片信息
                    cv::Mat  adj_chart_img = viewImageList[adj_chart_label - 1].img;
                    cv::Mat  adj_chart_depth = viewImageList[adj_chart_label - 1].depth;
                    cv::Mat adj_chart__GradXImg = viewImageList[adj_chart_label - 1].gradxImg;//得到当前面对应视口的梯度图应用颜色一致性追踪
                    cv::Mat adj_chart_GradYImg = viewImageList[adj_chart_label - 1].gradyImg;//得到当前面对应视口的梯度图应用颜色一致性追踪

                    math::Matrix4f  amat =   adj_texture_view.getWorldToCamMatrix();
                    Eigen::Matrix4f adj_chart_world_to_cam = mathToEigen(amat);
                    std::map<int, int>  computeFlag;
                    for(int face_idx = 0; face_idx < cur_charts.size(); face_idx++)//遍历当前chart上所有的顶点
                    {
                        f_i = cur_charts[face_idx];//当前面面索引
                        for(int v_idx = 0; v_idx < 3; v_idx++)//取出面上的三个顶点
                        {
                            int   f_v_idx = faces[f_i*3 + v_idx];
                            if(computeFlag.count(f_v_idx) != 0)//已经计算过
                            {
                                continue;
                            }
                            computeFlag[f_v_idx] = 100;

                            math::Vec3f   v_pos = vertices[f_v_idx];
                            Eigen::Vector4f   q0;
                            q0<<v_pos(0), v_pos(1), v_pos(2), 1;//三维点
                            Eigen::Vector4f   q = currentMat * q0;//把顶点变换到相机坐标

                            //color
                            float v_0 = q(0) * G2LTexConfig::get().IMAGE_FX / q(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_0 = q(1) * G2LTexConfig::get().IMAGE_FY / q(2) + G2LTexConfig::get().IMAGE_CY;

                            Eigen::Vector4f    p = adj_chart_world_to_cam * q0;//投影到邻接平面上
                            float v_1 = p(0) * G2LTexConfig::get().IMAGE_FX / p(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_1 = p(1) * G2LTexConfig::get().IMAGE_FY / p(2) + G2LTexConfig::get().IMAGE_CY;
                            if( v_0 >= G2LTexConfig::get().BOARD_IGNORE && v_0 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                                    u_0 >= G2LTexConfig::get().BOARD_IGNORE && u_0 <= (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) &&
                                    v_1 >= G2LTexConfig::get().BOARD_IGNORE && v_1 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                                    u_1 >= G2LTexConfig::get().BOARD_IGNORE && u_1 <= (G2LTexConfig::get().IMAGE_HEIGHT-G2LTexConfig::get().BOARD_IGNORE) )//ablity
                            {
                                float  currentgray = getInterColorFromRGBImgV2(currentImg, v_0, u_0);
                                float  adjgray = getInterColorFromRGBImgV2(adj_chart_img, v_1, u_1);
                                float  c_r = currentgray - adjgray;

                                float invz = 1.0f / q(2);
                                float  gx = getInterColorFromGrayImgV2(currentGradXImg, v_0, u_0);
                                float  gy = getInterColorFromGrayImgV2(currentGradYImg, v_0, u_0);
                                if(view_idx == 0)
                                {
//                                std::cout<<"gx:"<<gx<<"  gy:"<<gy<<"  c_r:"<<c_r << " invz:"<<invz<<std::endl;
                                }
                                float  k0 = gx * G2LTexConfig::get().IMAGE_FX * invz;
                                float  k1 = gy * G2LTexConfig::get().IMAGE_FY * invz;
                                float  k2 = -(k0 * q(0) + k1 * q(1))*invz;

                                c_J.setZero();
                                c_J(0) = -q(2) * k1 + q(1) * k2;
                                c_J(1) =  q(2) * k0 - q(0) * k2;
                                c_J(2) = -q(1) * k0 + q(0) * k1;
                                c_J(3) = k0;
                                c_J(4) = k1;
                                c_J(5) = k2;
                                c_JTJ += c_J * c_J.transpose();
                                c_JTr += c_J * c_r;
                                if(view_idx == 0)
                                {
//                                    std::cout<<c_JTJ<<std::endl;
                                }
                            }//end color

                            //depth
                            Eigen::Vector4f    q_d = currentMat * q0;//投影当前平面上
                            float depth_v = q_d(0) * G2LTexConfig::get().DEPTH_FX / q_d(2) + G2LTexConfig::get().DEPTH_CX;
                            float depth_u =  q_d(1) * G2LTexConfig::get().DEPTH_FY / q_d(2) + G2LTexConfig::get().DEPTH_CY;
                            //                        if( depth_v < 0 || depth_v > (G2LTexConfig::get().DEPTH_WIDTH - 1) || depth_u < 0 || depth_u > (G2LTexConfig::get().DEPTH_HEIGHT - 1))
                            if( depth_v < G2LTexConfig::get().BOARD_IGNORE || depth_v > (G2LTexConfig::get().DEPTH_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                                depth_u < G2LTexConfig::get().BOARD_IGNORE || depth_u > (G2LTexConfig::get().DEPTH_HEIGHT - G2LTexConfig::get().BOARD_IGNORE))
                            {
                                continue;
                            }

                            ushort d = currentDepth.at<ushort>(int(depth_u), int(depth_v));
                            if(d > 0 )//
                            {
                                float z = d/1000.0f;
                                float x = (depth_v - G2LTexConfig::get().DEPTH_CX)*z/G2LTexConfig::get().DEPTH_FX;
                                float y = (depth_u - G2LTexConfig::get().DEPTH_CY)*z/G2LTexConfig::get().DEPTH_FY;
                                Eigen::Vector3f  p_d(x, y, z);//深度图得到的三维点

                                float r = 0.0;
                                Eigen::Vector3f   rpq;
    //                            rpq<<q_d(0) - p_d(0), q_d(1) - p_d(1), q_d(2) - p_d(2);//两点之间的向量
                                rpq<<0.0, 0.0, q_d(2) - p_d(2);//两点之间的向量
                                d_J.setZero();
                                d_J(1) = -q_d(2);
                                d_J(2) = q_d(1);
                                d_J(3) = -1;
                                r = rpq(0);
                                d_JTJ += d_J * d_J.transpose();
                                d_JTr += d_J * r ;

                                d_J.setZero();
                                d_J(2) = -q_d(0);
                                d_J(0) = q_d(2);
                                d_J(4) = -1;
                                r = rpq(1);
                                d_JTJ += d_J *d_J.transpose();
                                d_JTr += d_J *r;

                                d_J.setZero();
                                d_J(0) = -q_d(1);
                                d_J(1) = q_d(0);
                                d_J(5) = -1;
                                r = rpq(2);
                                d_JTJ += d_J * d_J.transpose();
                                d_JTr += d_J * r;
                            }//end depth

                        }//end vertex on the face

                    }//end adj face vertex loop

                    //邻接面,（根据论文这个应该不需要考虑：）
                    computeFlag.clear();
                    for(int face_idx = 0; face_idx < adj_faces.size(); face_idx++)//遍历所有的面
                    {
                        f_i = adj_faces[face_idx];//邻接面中的面索引
                        for(int v_idx = 0; v_idx < 3; v_idx++)
                        {
                            int   f_v_idx = faces[f_i*3 + v_idx];//邻接面上的顶点
                            if(computeFlag.count(f_v_idx) != 0)//已经计算过
                            {
                                continue;
                            }
                            computeFlag[f_v_idx] = 100;
                            math::Vec3f   v_pos = vertices[f_v_idx];

                            Eigen::Vector4f   q0;
                            q0<<v_pos(0), v_pos(1), v_pos(2), 1;//三维点
                            Eigen::Vector4f   q = currentMat * q0;//每次迭代改变结果

                            float v_0 = q(0) * G2LTexConfig::get().IMAGE_FX / q(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_0 = q(1) * G2LTexConfig::get().IMAGE_FY / q(2) + G2LTexConfig::get().IMAGE_CY;

                            Eigen::Vector4f    p = adj_chart_world_to_cam * q0;//投影到邻接平面上
                            float v_1 = p(0) * G2LTexConfig::get().IMAGE_FX / p(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_1 = p(1) * G2LTexConfig::get().IMAGE_FY / p(2) + G2LTexConfig::get().IMAGE_CY;
                            if( v_0 >= G2LTexConfig::get().BOARD_IGNORE && v_0 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                                    u_0 >= G2LTexConfig::get().BOARD_IGNORE && u_0 <= (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) &&
                                    v_1 >= G2LTexConfig::get().BOARD_IGNORE && v_1 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                                    u_1 >= G2LTexConfig::get().BOARD_IGNORE && u_1 <= (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE))//ablity
                            {
                                float  currentgray = getInterColorFromRGBImgV2(currentImg, v_0, u_0);
                                float  adjgray = getInterColorFromRGBImgV2(adj_chart_img, v_1, u_1);
                                float  c_r = currentgray - adjgray;

                                float invz = 1.0f / q(2);
                                float  gx = getInterColorFromGrayImgV2(currentGradXImg, v_0, u_0);
                                float  gy = getInterColorFromGrayImgV2(currentGradYImg, v_0, u_0);

                                float  k0 = gx * G2LTexConfig::get().IMAGE_FX * invz;
                                float  k1 = gy * G2LTexConfig::get().IMAGE_FY * invz;
                                float  k2 = -(k0 * q(0) + k1 * q(1))*invz;

                                c_J.setZero();
                                c_J(3) = k0;
                                c_J(4) = k1;
                                c_J(5) = k2;
                                c_J(0) = -q(2) * k1 + q(1) * k2;
                                c_J(1) =  q(2) * k0 - q(0) * k2;
                                c_J(2) = -q(1) * k0 + q(0) * k1;
                                c_JTJ += c_J * c_J.transpose();
                                c_JTr += c_J * c_r;
                            }

                            //depth
                            Eigen::Vector4f   q_d = currentMat * q0;//每次迭代改变结果
                            float depth_v = q_d(0)*G2LTexConfig::get().DEPTH_FX/q_d(2) + G2LTexConfig::get().DEPTH_CX;
                            float depth_u =  q_d(1)*G2LTexConfig::get().DEPTH_FY/q_d(2) + G2LTexConfig::get().DEPTH_CY;
                            if( depth_v < G2LTexConfig::get().BOARD_IGNORE || depth_v > (G2LTexConfig::get().DEPTH_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                                    depth_u < G2LTexConfig::get().BOARD_IGNORE || depth_u >= (G2LTexConfig::get().DEPTH_HEIGHT - G2LTexConfig::get().BOARD_IGNORE))
                            {
                                continue;
                            }

                            ushort d = currentDepth.at<ushort>(int(depth_u), int(depth_v));
                            if(d > 0 )//
                            {
                                float z = d/1000.0f;
                                float x = (depth_v - G2LTexConfig::get().DEPTH_CX)*z/G2LTexConfig::get().DEPTH_FX;
                                float y = (depth_u - G2LTexConfig::get().DEPTH_CY)*z/G2LTexConfig::get().DEPTH_FY;
                                Eigen::Vector3f  p_d(x, y, z);//深度图得到的三维点

                                float r = 0.0;
                                Eigen::Vector3f   rpq;
    //                            rpq<<q_d(0) - p_d(0),q_d(1) - p_d(1),q_d(2) - p_d(2);//两点之间的向量
                                rpq<<0.0, 0.0, q_d(2) - p_d(2);//两点之间的向量

                                d_J.setZero();
                                d_J(1) = -q_d(2);
                                d_J(2) = q_d(1);
                                d_J(3) = -1;
                                r = rpq(0);
                                d_JTJ += d_J * d_J.transpose() ;
                                d_JTr += d_J * r  ;

                                d_J.setZero();
                                d_J(2) = -q_d(0);
                                d_J(0) = q_d(2);
                                d_J(4) = -1;
                                r = rpq(1);
                                d_JTJ += d_J * d_J.transpose();
                                d_JTr += d_J * r;

                                d_J.setZero();
                                d_J(0) = -q_d(1);
                                d_J(1) = q_d(0);
                                d_J(5) = -1;
                                r = rpq(2);
                                d_JTJ += d_J * d_J.transpose();
                                d_JTr += d_J * r;
                            }
                        }

                    }


                }//end adj chart


            }//end current same label chart


            //twopass option
            int tp_chartnum = twoPassLabelChartCount[view_idx];//当前标签在chart列表中个数
            int tp_chart_start = getChartIndex(twoPassLabelChartCount, current_label);//当前标签在chart列表中的起始位置

            for(int chart_idx = 0; chart_idx < tp_chartnum; chart_idx++)//遍历当前视口对应的所有的chart块（相同标签）
            {
                std::vector<std::size_t> cur_chart = twoPassSubgraphs[tp_chart_start+chart_idx];//取出当前chart块
                std::vector<std::size_t>   adj_patches = twopass_adj_chart_graph[chart_start+chart_idx];//取出当前chart块相邻块

                if(adj_patches.size() == 0)//没有邻域不考虑
                {
                    continue;
                }

                for(int adj_idx = 0; adj_idx < adj_patches.size(); adj_idx++)//考虑当前chart的所有邻域chart
                {
                    std::size_t  adj_chartidx = adj_patches[adj_idx];//adj patch index
                    std::vector<std::size_t>  adj_chart = twoPassSubgraphs[adj_chartidx];//邻接的块中所有的面（不存在与当前块相同标签的面）
                    int    f_i = adj_chart[0];//邻接面中的面索引
                    int    adj_chart_label = graph.get_twoPassLabel(f_i);//邻接面中的标签
                    TextureView  adj_texture_view =  texture_views.at(adj_chart_label - 1);//邻接块对应的视口
                    //取出邻接视口对应的图片信息
                    cv::Mat  adj_chart_img = viewImageList[adj_chart_label - 1].img;

                    math::Matrix4f  amat =   adj_texture_view.getWorldToCamMatrix();
                    Eigen::Matrix4f adj_chart_world_to_cam = mathToEigen(amat);
                    std::map<int, int>  computeFlag;

                    for(int face_idx = 0; face_idx < cur_chart.size(); face_idx++)//遍历当前chart上所有的顶点
                    {
                        f_i = cur_chart[face_idx];//当前面面索引
                        for(int v_idx = 0; v_idx < 3; v_idx++)//取出面上的三个顶点
                        {
                            int   f_v_idx = faces[f_i*3 + v_idx];
                            if(computeFlag.count(f_v_idx) != 0)//已经计算过
                            {
                                continue;
                            }
                            computeFlag[f_v_idx] = 100;

                            math::Vec3f   v_pos = vertices[f_v_idx];
                            Eigen::Vector4f   q0;
                            q0<<v_pos(0), v_pos(1), v_pos(2), 1;//三维点
                            Eigen::Vector4f   q = currentMat * q0;//把顶点变换到相机坐标

                            //color
                            float v_0 = q(0) * G2LTexConfig::get().IMAGE_FX / q(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_0 = q(1) * G2LTexConfig::get().IMAGE_FY / q(2) + G2LTexConfig::get().IMAGE_CY;

                            Eigen::Vector4f    p = adj_chart_world_to_cam * q0;//投影到邻接平面上
                            float v_1 = p(0) * G2LTexConfig::get().IMAGE_FX / p(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_1 = p(1) * G2LTexConfig::get().IMAGE_FY / p(2) + G2LTexConfig::get().IMAGE_CY;
                            if( v_0 >= G2LTexConfig::get().BOARD_IGNORE && v_0 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                                    u_0 >= G2LTexConfig::get().BOARD_IGNORE && u_0 <= (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) &&
                                    v_1 >= G2LTexConfig::get().BOARD_IGNORE && v_1 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                                    u_1 >= G2LTexConfig::get().BOARD_IGNORE && u_1 <= (G2LTexConfig::get().IMAGE_HEIGHT-G2LTexConfig::get().BOARD_IGNORE) )//ablity
                            {
                                float  currentgray = getInterColorFromRGBImgV2(currentImg, v_0, u_0);
                                float  adjgray = getInterColorFromRGBImgV2(adj_chart_img, v_1, u_1);
                                float  c_r = currentgray - adjgray;

                                float invz = 1.0f / q(2);
                                float  gx = getInterColorFromGrayImgV2(currentGradXImg, v_0, u_0);
                                float  gy = getInterColorFromGrayImgV2(currentGradYImg, v_0, u_0);
                                if(view_idx == 0)
                                {
//                                std::cout<<"gx:"<<gx<<"  gy:"<<gy<<"  c_r:"<<c_r << " invz:"<<invz<<std::endl;
                                }
                                float  k0 = gx * G2LTexConfig::get().IMAGE_FX * invz;
                                float  k1 = gy * G2LTexConfig::get().IMAGE_FY * invz;
                                float  k2 = -(k0 * q(0) + k1 * q(1))*invz;

                                c_J.setZero();
                                c_J(0) = -q(2) * k1 + q(1) * k2;
                                c_J(1) =  q(2) * k0 - q(0) * k2;
                                c_J(2) = -q(1) * k0 + q(0) * k1;
                                c_J(3) = k0;
                                c_J(4) = k1;
                                c_J(5) = k2;
                                c_JTJ += c_J * c_J.transpose();
                                c_JTr += c_J * c_r;
                            }//end v0 u0 v1 u1 board check

                            //depth
                            Eigen::Vector4f    q_d = currentMat * q0;//投影当前平面上
                            float depth_v = q_d(0) * G2LTexConfig::get().DEPTH_FX / q_d(2) + G2LTexConfig::get().DEPTH_CX;
                            float depth_u =  q_d(1) * G2LTexConfig::get().DEPTH_FY / q_d(2) + G2LTexConfig::get().DEPTH_CY;

                            if( depth_v < G2LTexConfig::get().BOARD_IGNORE || depth_v > (G2LTexConfig::get().DEPTH_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                                depth_u < G2LTexConfig::get().BOARD_IGNORE || depth_u > (G2LTexConfig::get().DEPTH_HEIGHT - G2LTexConfig::get().BOARD_IGNORE))
                            {
                                continue;
                            }

                            ushort d = currentDepth.at<ushort>(int(depth_u), int(depth_v));
                            if(d > 0 )//
                            {
                                float z = d/1000.0f;
                                float x = (depth_v - G2LTexConfig::get().DEPTH_CX)*z/G2LTexConfig::get().DEPTH_FX;
                                float y = (depth_u - G2LTexConfig::get().DEPTH_CY)*z/G2LTexConfig::get().DEPTH_FY;
                                Eigen::Vector3f  p_d(x, y, z);//深度图得到的三维点

                                float r = 0.0;
                                Eigen::Vector3f   rpq;
    //                            rpq<<q_d(0) - p_d(0), q_d(1) - p_d(1), q_d(2) - p_d(2);//两点之间的向量
                                rpq<<0.0, 0.0, q_d(2) - p_d(2);//两点之间的向量
                                d_J.setZero();
                                d_J(1) = -q_d(2);
                                d_J(2) = q_d(1);
                                d_J(3) = -1;
                                r = rpq(0);
                                d_JTJ += d_J * d_J.transpose();
                                d_JTr += d_J * r ;

                                d_J.setZero();
                                d_J(2) = -q_d(0);
                                d_J(0) = q_d(2);
                                d_J(4) = -1;
                                r = rpq(1);
                                d_JTJ += d_J *d_J.transpose();
                                d_JTr += d_J *r;

                                d_J.setZero();
                                d_J(0) = -q_d(1);
                                d_J(1) = q_d(0);
                                d_J(5) = -1;
                                r = rpq(2);
                                d_JTJ += d_J * d_J.transpose();
                                d_JTr += d_J * r;
                            }//end depth


                        }//end v_idx of 3  vertices

                    }//end face_idx of cur_chart

                    //邻接面,（根据论文这个应该不需要考虑：）
                    computeFlag.clear();
                    for(int face_idx = 0; face_idx < adj_chart.size(); face_idx++)//遍历所有的面
                    {
                        f_i = adj_chart[face_idx];//邻接面中的面索引
                        for(int v_idx = 0; v_idx < 3; v_idx++)
                        {
                            int   f_v_idx = faces[f_i*3 + v_idx];//邻接面上的顶点
                            if(computeFlag.count(f_v_idx) != 0)//已经计算过
                            {
                                continue;
                            }
                            computeFlag[f_v_idx] = 100;

                            math::Vec3f   v_pos = vertices[f_v_idx];

                            Eigen::Vector4f   q0;
                            q0<<v_pos(0), v_pos(1), v_pos(2), 1;//三维点
                            Eigen::Vector4f   q = currentMat * q0;//每次迭代改变结果

                            float v_0 = q(0) * G2LTexConfig::get().IMAGE_FX / q(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_0 = q(1) * G2LTexConfig::get().IMAGE_FY / q(2) + G2LTexConfig::get().IMAGE_CY;

                            Eigen::Vector4f    p = adj_chart_world_to_cam * q0;//投影到邻接平面上
                            float v_1 = p(0) * G2LTexConfig::get().IMAGE_FX / p(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_1 = p(1) * G2LTexConfig::get().IMAGE_FY / p(2) + G2LTexConfig::get().IMAGE_CY;
                            if( v_0 >= G2LTexConfig::get().BOARD_IGNORE && v_0 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                                    u_0 >= G2LTexConfig::get().BOARD_IGNORE && u_0 <= (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) &&
                                    v_1 >= G2LTexConfig::get().BOARD_IGNORE && v_1 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                                    u_1 >= G2LTexConfig::get().BOARD_IGNORE && u_1 <= (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE))//ablity
                            {
                                float  currentgray = getInterColorFromRGBImgV2(currentImg, v_0, u_0);
                                float  adjgray = getInterColorFromRGBImgV2(adj_chart_img, v_1, u_1);
                                float  c_r = currentgray - adjgray;

                                float invz = 1.0f / q(2);
                                float  gx = getInterColorFromGrayImgV2(currentGradXImg, v_0, u_0);
                                float  gy = getInterColorFromGrayImgV2(currentGradYImg, v_0, u_0);

                                float  k0 = gx * G2LTexConfig::get().IMAGE_FX * invz;
                                float  k1 = gy * G2LTexConfig::get().IMAGE_FY * invz;
                                float  k2 = -(k0 * q(0) + k1 * q(1))*invz;

                                c_J.setZero();
                                c_J(3) = k0;
                                c_J(4) = k1;
                                c_J(5) = k2;
                                c_J(0) = -q(2) * k1 + q(1) * k2;
                                c_J(1) =  q(2) * k0 - q(0) * k2;
                                c_J(2) = -q(1) * k0 + q(0) * k1;
                                c_JTJ += c_J * c_J.transpose();
                                c_JTr += c_J * c_r;
                            }//end v0 u0 v1 u1 board checking

                            //depth
                            Eigen::Vector4f   q_d = currentMat * q0;//每次迭代改变结果
                            float depth_v = q_d(0)*G2LTexConfig::get().DEPTH_FX/q_d(2) + G2LTexConfig::get().DEPTH_CX;
                            float depth_u =  q_d(1)*G2LTexConfig::get().DEPTH_FY/q_d(2) + G2LTexConfig::get().DEPTH_CY;
                            if( depth_v < G2LTexConfig::get().BOARD_IGNORE || depth_v > (G2LTexConfig::get().DEPTH_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                                    depth_u < G2LTexConfig::get().BOARD_IGNORE || depth_u >= (G2LTexConfig::get().DEPTH_HEIGHT - G2LTexConfig::get().BOARD_IGNORE))
                            {
                                continue;
                            }

                            ushort d = currentDepth.at<ushort>(int(depth_u), int(depth_v));
                            if(d > 0 )//
                            {
                                float z = d/1000.0f;
                                float x = (depth_v - G2LTexConfig::get().DEPTH_CX)*z/G2LTexConfig::get().DEPTH_FX;
                                float y = (depth_u - G2LTexConfig::get().DEPTH_CY)*z/G2LTexConfig::get().DEPTH_FY;
                                Eigen::Vector3f  p_d(x, y, z);//深度图得到的三维点

                                float r = 0.0;
                                Eigen::Vector3f   rpq;
    //                            rpq<<q_d(0) - p_d(0),q_d(1) - p_d(1),q_d(2) - p_d(2);//两点之间的向量
                                rpq<<0.0, 0.0, q_d(2) - p_d(2);//两点之间的向量

                                d_J.setZero();
                                d_J(1) = -q_d(2);
                                d_J(2) = q_d(1);
                                d_J(3) = -1;
                                r = rpq(0);
                                d_JTJ += d_J * d_J.transpose() ;
                                d_JTr += d_J * r  ;

                                d_J.setZero();
                                d_J(2) = -q_d(0);
                                d_J(0) = q_d(2);
                                d_J(4) = -1;
                                r = rpq(1);
                                d_JTJ += d_J * d_J.transpose();
                                d_JTr += d_J * r;

                                d_J.setZero();
                                d_J(0) = -q_d(1);
                                d_J(1) = q_d(0);
                                d_J(5) = -1;
                                r = rpq(2);
                                d_JTJ += d_J * d_J.transpose();
                                d_JTr += d_J * r;
                            }
                        }//end two pass v_idx of 3  vertices

                    }//end face_idx of adj_chart


                }//end ajd_idx of adj_patches


            }//end chart_idx of tp_chartnum

            Eigen::MatrixXd  A_JTJ(nvariable, nvariable);
            Eigen::VectorXd  b_JTr(nvariable, 1);
            A_JTJ.setZero();
            b_JTr.setZero();
            A_JTJ = c_JTJ*w_color*w_color + d_JTJ*w_depth*w_depth;
            b_JTr = c_JTr*w_color + d_JTr*w_depth;
//            A_JTJ = c_JTJ*w_color*w_color ;
//            b_JTr = c_JTr*w_color;

//            A_JTJ = d_JTJ;
//            b_JTr = d_JTr;
            Eigen::VectorXd  x(6);

            x = -A_JTJ.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_JTr);
            Eigen::Affine3d aff_mat;
            aff_mat.linear() = (Eigen::Matrix3d) Eigen::AngleAxisd(x(2), Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(x(1), Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(x(0), Eigen::Vector3d::UnitX());
            aff_mat.translation() = Eigen::Vector3d(x(3), x(4), x(5));

            Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();
//            if(checkNAN(delta))//防止解错误
//            {
//                continue;
//            }


            Eigen::Matrix4f trans = delta* currentMat;
            texture_view.setWorldToCamMatrix(trans);
            if(view_idx == 0)
            {
//                std::cout<<"1---------------------A----------------------"<<std::endl;
//                std::cout<<A_JTJ<<std::endl;
//                std::cout<<"-------------------b----------------"<<std::endl;
//                std::cout<<b_JTr<<std::endl;
//                std::cout<<"currentMat:"<<std::endl<<currentMat<<std::endl;

//                std::cout<<"trans:"<<std::endl<<trans<<std::endl;
//                std::cout<<"delta:"<<delta<<std::endl;
//               math::Matrix4f  mat = texture_view.getWorldToCamMatrix();
//               Eigen::Matrix4f ccmat = mathToEigen(mat);
//               std::cout<<"-new:"<<std::endl<<ccmat<<std::endl;
            }

        }//end view
    }//end iter

    for(int i = 0; i < num_faces; i++)//更新每个面的信息。
    {
        myFaceInfo face_info = faceInfoList[i];
        int face_label = face_info.lable;
        if(face_label != 0)
        {
            TextureView  texture_view =  texture_views.at(face_label - 1);
            math::Matrix4f  mat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  trans = mathToEigen(mat);
            faceInfoList[i].world_to_cam = trans;//
        }
    }

}

void rgb2lab(cv::Mat src, cv::Mat &tagt)
{
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            uchar r = src.at<cv::Vec3b>(i, j)[0];
            uchar g = src.at<cv::Vec3b>(i, j)[1];
            uchar b = src.at<cv::Vec3b>(i, j)[2];

            float ll,la,lb;
            Pix_RGB2LAB(r, g, b, ll, la, lb);
            tagt.at<cv::Vec3f>(i, j)[0] = ll;
            tagt.at<cv::Vec3f>(i, j)[1] = la;
            tagt.at<cv::Vec3f>(i, j)[2] = lb;
        }
    }
}

void combineOptionCameraPosesV2(UniGraph const & graph, mve::TriangleMesh::ConstPtr mesh,
                                mve::MeshInfo const & mesh_info, std::vector<TextureView> &texture_views,
                                std::string const & indir, std::vector<myFaceInfo>   &faceInfoList,
                                std::vector<myViewImageInfo>&  viewImageList,
                                std::vector<std::vector<std::size_t> > &patch_graph,
                                std::vector<std::vector<std::size_t> > &subgraphs, std::vector<int>  &labelPatchCount,
                                Settings const & settings)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();//所有顶点
    std::size_t  const num_faces = faces.size()/3;//面的个数
    float   w_color = 1, w_depth = 100;


    for(int i = 0; i < texture_views.size(); i++)
    {
        myViewImageInfo  info;
        info.view_id = i;
        char colorbuf[256];
        char depthbuf[256];
        sprintf(colorbuf,"%s/color_%02d.jpg",indir.c_str(), i);
        sprintf(depthbuf,"%s/depth_%02d.png",indir.c_str(), i);
        info.img = cv::imread(colorbuf);
        info.depth = cv::imread(depthbuf, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

        cv::Mat gray;
        cv::cvtColor(info.img, gray, CV_BGR2GRAY);
        generateGradImg(gray, info.gradxImg, info.gradyImg);

        info.L_alpha = 1.0;
        info.L_beta = 0.0;

        info.A_alpha = 1.0;
        info.A_beta = 0.0;

        info.B_alpha = 1.0;
        info.B_beta = 0.0;

        info.LabImg = cv::Mat(info.img.rows, info.img.cols, CV_32FC3, cv::Scalar(0.0,0.0,0.0));
//        rgb2lab(info.img, info.LabImg);//彩色图转化为lab图

        cv::Mat floatmat;
        info.img.convertTo(floatmat,CV_32FC3, 1.0f/255.0f);
        cv::cvtColor(floatmat, info.LabImg, CV_RGB2Lab);//

        double maxVal, minVal;
        cv::minMaxLoc(info.LabImg, &minVal, &maxVal);
        std::cout<<" image_l:"<<i<<"  original min, max: "<<minVal<<", "<<maxVal<<std::endl;

        viewImageList.push_back(info);

        //为每个视口读取图像，防止访问为NULL
        TextureView  &texture_view =  texture_views.at(i);
        texture_view.load_image();
        texture_view.generate_gradient_magnitude();
    }


    for(int i = 0; i < num_faces; i++)//读取每个面的信息。
    {
        myFaceInfo  finfo;
        int lable = graph.get_label(i);
        if(lable != 0)
        {
            finfo.face_id = i;
            finfo.lable = lable;
            //        finfo.world_to_cam = ;
            TextureView  texture_view =  texture_views.at(lable - 1);
            math::Matrix4f  mat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  wtc;
            wtc<<mat(0,0),mat(0,1),mat(0,2),mat(0,3),
                    mat(1,0),mat(1,1),mat(1,2),mat(1,3),
                    mat(2,0),mat(2,1),mat(2,2),mat(2,3),
                    mat(3,0),mat(3,1),mat(3,2),mat(3,3);
            finfo.world_to_cam = wtc;

            math::Vec3f   v_1 = vertices[faces[i*3]];
            math::Vec3f   v_2 = vertices[faces[i*3 + 1]];
            math::Vec3f   v_3 = vertices[faces[i*3 + 2]];


            FaceProjectionInfo info = {i, 0.0f, 0.0f, math::Vec3f(0.0f, 0.0f, 0.0f)};//view_id;  quality; mean_color;

            texture_view.get_face_info(v_1, v_2, v_3, &info, settings);
            finfo.quality = info.quality;

            //            if(info.quality > 0.5)//不需要采样。
            //            {
            //                finfo.generateSample(v_1, v_2, v_3);
            //            }
        }
        else
        {
            finfo.face_id = i;
            finfo.world_to_cam = Eigen::Matrix4f::Identity();
            finfo.lable = 0;
            finfo.quality = 0.0f;
        }

        faceInfoList.push_back(finfo);
    }


    //构建1通chart块之间的连接关系
//    std::vector<int>  labelPatchCount;//记录每个标签包含的chart数量，也就可以计算下一个标签所有chart开始的索引
    std::vector<int>  twoPassLabelChartCount;//记录每个标签包含的chart数量，也就可以计算下一个标签所有chart开始的索引
    std::vector<std::vector<std::size_t> > twoPassSubgraphs;
    int startsize, twoPassStartSize;
    int endsize, twoPassEndSize;
    for(int i = 0; i < texture_views.size(); i++)
    {
        int const label = i + 1;
        //one pass
        startsize = subgraphs.size();
        graph.get_subgraphs(label, &subgraphs);
        endsize = subgraphs.size();
        labelPatchCount.push_back(endsize - startsize);//当前标签chart的个数

        //two pass
//        twoPassStartSize = twoPassSubgraphs.size();
//        graph.get_twoPassSubgraphs(label, &twoPassSubgraphs);
//        twoPassEndSize = twoPassSubgraphs.size();
//        twoPassLabelChartCount.push_back(twoPassEndSize - twoPassStartSize);
    }

    build_patch_adjacency_graph(graph, subgraphs, labelPatchCount, patch_graph, faceInfoList);//构建chart的邻接图
//    std::vector<std::vector<std::size_t> > twopass_adj_chart_graph;//构建2通chart的邻接图
//    build_twopass_adjacency_graph(graph, twoPassSubgraphs, twoPassLabelChartCount, twopass_adj_chart_graph);

    int nvariable = 6;
    for(int iter = 0; iter < 50; iter++)//迭代次数
    {
        std::cout<<"22----------------iter:"<<iter<<std::endl;
#pragma omp parallel for schedule(static, 8)
        for(int view_idx = 0; view_idx < texture_views.size(); view_idx++)
        {
            int current_label = view_idx + 1;
            TextureView  &texture_view =  texture_views.at(current_label - 1);
            math::Matrix4f  cmat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  currentMat = mathToEigen(cmat);

            Eigen::MatrixXd   JTJ(nvariable, nvariable);//总的优化方程中的A
            Eigen::VectorXd   JTr(nvariable);//总的优化方程中的b
            JTJ.setZero();
            JTr.setZero();

            //颜色项
            Eigen::MatrixXd    c_JTJ(nvariable, nvariable);
            Eigen::MatrixXd    c_JTr(nvariable, 1);
            Eigen::MatrixXd    c_J(nvariable, 1);
            c_JTJ.setZero();
            c_JTr.setZero();

            //深度项
            Eigen::MatrixXd    d_JTJ(nvariable, nvariable);
            Eigen::MatrixXd    d_JTr(nvariable, 1);
            Eigen::MatrixXd    d_J(nvariable, 1);
            d_JTJ.setZero();
            d_JTr.setZero();

            int chartnum = labelPatchCount[view_idx];//当前标签在chart列表中个数
            int chart_start = getChartIndex(labelPatchCount, current_label);//当前标签在chart列表中的起始位置
            for(int chart_idx = 0; chart_idx < chartnum; chart_idx++)//遍历当前视口所有的chart块
            {
                std::vector<std::size_t> cur_charts = subgraphs[chart_start+chart_idx];//取出当前chart块

                std::vector<std::size_t>   adj_patches = patch_graph[chart_start+chart_idx];//取出当前chart块相邻块
                if(adj_patches.size() == 0)//没有邻域不考虑
                {
                    continue;
                }

                for(int adj_idx = 0; adj_idx < adj_patches.size(); adj_idx++)//考虑当前chart的所有邻域chart
                {
                    std::size_t  adj_chartidx = adj_patches[adj_idx];//adj patch index
                    std::vector<std::size_t>  adj_chart = subgraphs[adj_chartidx];//邻接的块中所有的面（不存在与当前块相同标签的面）
                    int    f_i = adj_chart[0];//邻接面中的面索引
                    int    adj_chart_label = graph.get_label(f_i);//邻接面中的标签

                    Eigen::MatrixXd JTc(nvariable, nvariable);
                    JTc.setZero();
                    Eigen::VectorXd JRc(nvariable, 1);
                    JRc.setZero();
                    Eigen::MatrixXd JTd(nvariable, nvariable);
                    JTd.setZero();
                    Eigen::VectorXd JRd(nvariable, 1);
                    JRd.setZero();
                    camera_pose_option(texture_views, current_label, adj_chart_label, faces, vertices, cur_charts,
                                       adj_chart, viewImageList, JTc, JRc, JTd, JRd);
                    c_JTJ += JTc; c_JTr += JRc;
                    d_JTJ += JTd; d_JTr += JRd;
//                    std::cout<<"---------JRc:"<<std::endl<<JRc<<std::endl;
//                    std::cout<<"---------JRd:"<<std::endl<<JRd<<std::endl;

                }//end adj chart

            }//end current same label chart

//            //twopass option
//            int tp_chartnum = twoPassLabelChartCount[view_idx];//当前标签在chart列表中个数
//            int tp_chart_start = getChartIndex(twoPassLabelChartCount, current_label);//当前标签在chart列表中的起始位置

//            for(int chart_idx = 0; chart_idx < tp_chartnum; chart_idx++)//遍历当前视口对应的所有的chart块（相同标签）
//            {
//                std::vector<std::size_t> cur_chart = twoPassSubgraphs[tp_chart_start+chart_idx];//取出当前chart块
//                std::vector<std::size_t>   adj_patches = twopass_adj_chart_graph[chart_start+chart_idx];//取出当前chart块相邻块
//                if(adj_patches.size() == 0)//没有邻域不考虑
//                {
//                    continue;
//                }

//                for(int adj_idx = 0; adj_idx < adj_patches.size(); adj_idx++)//考虑当前chart的所有邻域chart
//                {
//                    std::size_t  adj_chartidx = adj_patches[adj_idx];//adj patch index
//                    std::vector<std::size_t>  adj_chart = twoPassSubgraphs[adj_chartidx];//邻接的块中所有的面（不存在与当前块相同标签的面）
//                    int    f_i = adj_chart[0];//邻接面中的面索引
//                    int    adj_chart_label = graph.get_twoPassLabel(f_i);//邻接面中的标签

//                    Eigen::MatrixXd JTc(nvariable, nvariable);
//                    JTc.setZero();
//                    Eigen::VectorXd JRc(nvariable, 1);
//                    JRc.setZero();
//                    Eigen::MatrixXd JTd(nvariable, nvariable);
//                    JTd.setZero();
//                    Eigen::VectorXd JRd(nvariable, 1);
//                    JRd.setZero();
//                    camera_pose_option(texture_views, current_label, adj_chart_label, faces, vertices, cur_chart,
//                                       adj_chart, viewImageList, JTc, JRc, JTd, JRd);
//                    c_JTJ += JTc; c_JTr += JRc;
//                    d_JTJ += JTd; d_JTr += JRd;
//                }//end ajd_idx of adj_patches
//            } //end chart_idx of tp_chartnum


            Eigen::MatrixXd  A_JTJ(nvariable, nvariable);
            Eigen::VectorXd  b_JTr(nvariable, 1);
            A_JTJ.setZero();
            b_JTr.setZero();
            A_JTJ = c_JTJ*w_color*w_color + d_JTJ*w_depth*w_depth;
            b_JTr = c_JTr*w_color + d_JTr*w_depth;

            //线不考虑深度一致
//            A_JTJ = c_JTJ*w_color*w_color ;
//            b_JTr = c_JTr*w_color;

//            A_JTJ = d_JTJ;
//            b_JTr = d_JTr;
            Eigen::VectorXd  x(6);

            x = -A_JTJ.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_JTr);
            Eigen::Affine3d aff_mat;
            aff_mat.linear() = (Eigen::Matrix3d) Eigen::AngleAxisd(x(2), Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(x(1), Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(x(0), Eigen::Vector3d::UnitX());
            aff_mat.translation() = Eigen::Vector3d(x(3), x(4), x(5));

            Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();
//            if(checkNAN(delta))//防止解错误
//            {
//                continue;
//            }


            Eigen::Matrix4f trans = delta* currentMat;
            texture_view.setWorldToCamMatrix(trans);

        }//end view

    }//end iter

    for(int i = 0; i < num_faces; i++)//更新每个面的信息。
    {
        myFaceInfo face_info = faceInfoList[i];
        int face_label = face_info.lable;
        if(face_label != 0)
        {
            TextureView  texture_view =  texture_views.at(face_label - 1);
            math::Matrix4f  mat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  trans = mathToEigen(mat);
            faceInfoList[i].world_to_cam = trans;//
        }
    }
}


void combineOptionCameraPosesWithDetail(UniGraph const & graph, mve::TriangleMesh::ConstPtr mesh, mve::MeshInfo const & mesh_info,
                              std::vector<TextureView> &texture_views, std::string const & indir, std::vector<myFaceInfo>   &faceInfoList,
                              std::vector<myViewImageInfo>&  viewImageList, std::vector<std::vector<std::size_t> > &patch_graph,
                              std::vector<std::vector<std::size_t> > &subgraphs, std::vector<int>  &labelPatchCount, Settings const & settings)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();
    std::size_t  const num_faces = faces.size()/3;

    float   w_color = 1, w_detail = 1;
    float  w_depth = 10;


    for(int i = 0; i < texture_views.size(); i++)
    {
        TextureView  &texture_view =  texture_views.at(i);

        myViewImageInfo  info;
        info.view_id = i;
        char colorbuf[256];
        char depthbuf[256];
        sprintf(colorbuf,"%s/color_%02d.jpg",indir.c_str(), i);
        sprintf(depthbuf,"%s/depth_%02d.png",indir.c_str(), i);
        info.img = cv::imread(colorbuf);
        info.depth = cv::imread(depthbuf, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

        cv::Mat gray;
        cv::cvtColor(info.img, gray, CV_BGR2GRAY);
        generateGradImg(gray, info.gradxImg, info.gradyImg);

        generateGradImg(texture_view.detailMap, info.detailgradxImg, info.detailgradyImg);

        info.L_alpha = 1.0;
        info.L_beta = 0.0;

        info.A_alpha = 1.0;
        info.A_beta = 0.0;

        info.B_alpha = 1.0;
        info.B_beta = 0.0;


        viewImageList.push_back(info);

//        TextureView  &texture_view =  texture_views.at(i);
        texture_view.load_image();
        texture_view.generate_gradient_magnitude();
    }


    for(int i = 0; i < num_faces; i++)//
    {
        myFaceInfo  finfo;
        int lable = graph.get_label(i);
        if(lable != 0)
        {
            finfo.face_id = i;
            finfo.lable = lable;

            TextureView  texture_view =  texture_views.at(lable - 1);
            math::Matrix4f  mat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  wtc;
            wtc<<mat(0,0),mat(0,1),mat(0,2),mat(0,3),
                    mat(1,0),mat(1,1),mat(1,2),mat(1,3),
                    mat(2,0),mat(2,1),mat(2,2),mat(2,3),
                    mat(3,0),mat(3,1),mat(3,2),mat(3,3);
            finfo.world_to_cam = wtc;

            math::Vec3f   v_1 = vertices[faces[i*3]];
            math::Vec3f   v_2 = vertices[faces[i*3 + 1]];
            math::Vec3f   v_3 = vertices[faces[i*3 + 2]];


            FaceProjectionInfo info = {i, 0.0f, 0.0f, math::Vec3f(0.0f, 0.0f, 0.0f)};//view_id;  quality; mean_color;

            texture_view.get_face_info(v_1, v_2, v_3, &info, settings);
            finfo.quality = info.quality;
        }
        else
        {
            finfo.face_id = i;
            finfo.world_to_cam = Eigen::Matrix4f::Identity();
            finfo.lable = 0;
            finfo.quality = 0.0f;
        }

        faceInfoList.push_back(finfo);
    }


    std::vector<int>  twoPassLabelChartCount;//记录每个标签包含的chart数量，也就可以计算下一个标签所有chart开始的索引
    std::vector<std::vector<std::size_t> > twoPassSubgraphs;
    int startsize, twoPassStartSize;
    int endsize, twoPassEndSize;
    for(int i = 0; i < texture_views.size(); i++)
    {
        int const label = i + 1;
        //one pass
        startsize = subgraphs.size();
        graph.get_subgraphs(label, &subgraphs);
        endsize = subgraphs.size();
        labelPatchCount.push_back(endsize - startsize);//当前标签chart的个数
    }

    build_patch_adjacency_graph(graph, subgraphs, labelPatchCount, patch_graph, faceInfoList);//构建chart的邻接图

    int nvariable = 6;
    for(int iter = 0; iter < 200; iter++)//迭代次数
    {
        if(iter % 50 == 0)
        {
            std::cout<<"-----------iter:"<<iter<<std::endl;
        }

#pragma omp parallel for
        for(int view_idx = 0; view_idx < texture_views.size(); view_idx++)
        {

            int current_label = view_idx + 1;
            TextureView  &texture_view =  texture_views.at(view_idx);
            math::Matrix4f  cmat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  currentMat = mathToEigen(cmat);

            Eigen::MatrixXd   JTJ(nvariable, nvariable);
            Eigen::VectorXd   JTr(nvariable);
            JTJ.setZero();
            JTr.setZero();

            //颜色项
            Eigen::MatrixXd    c_JTJ(nvariable, nvariable);
            Eigen::MatrixXd    c_JTr(nvariable, 1);
            Eigen::MatrixXd    c_J(nvariable, 1);
            c_JTJ.setZero();
            c_JTr.setZero();
            c_J.setZero();
            //深度项
            Eigen::MatrixXd    d_JTJ(nvariable, nvariable);
            Eigen::MatrixXd    d_JTr(nvariable, 1);
            Eigen::MatrixXd    d_J(nvariable, 1);
            d_JTJ.setZero();
            d_JTr.setZero();
            d_J.setZero();
            //细节项
            Eigen::MatrixXd    t_JTJ(nvariable, nvariable);
            Eigen::MatrixXd    t_JTr(nvariable, 1);
            Eigen::MatrixXd    t_J(nvariable, 1);
            t_JTJ.setZero();
            t_JTr.setZero();
            t_J.setZero();

            int chartnum = labelPatchCount[view_idx];
            int chart_start = getChartIndex(labelPatchCount, current_label);//当前标签在chart列表中的起始位置
//#pragma omp parallel for
            for(int chart_idx = 0; chart_idx < chartnum; chart_idx++)//遍历当前视口所有的chart块
            {
                std::vector<std::size_t>    cur_charts = subgraphs[chart_start+chart_idx];//取出当前chart块
                std::vector<std::size_t>    adj_patches = patch_graph[chart_start+chart_idx];//取出当前chart块相邻块
                if(adj_patches.size() == 0)
                {
                    continue;
                }

                for(int adj_idx = 0; adj_idx < adj_patches.size(); adj_idx++)//考虑当前chart的所有邻域chart
                {
                    std::size_t  adj_chartidx = adj_patches[adj_idx];//adj patch index
                    std::vector<std::size_t>  adj_chart = subgraphs[adj_chartidx];//邻接的块中所有的面（不存在与当前块相同标签的面）
                    int    f_i = adj_chart[0];
                    int    adj_chart_label = graph.get_label(f_i);//邻接面中的标签

                    Eigen::MatrixXd JTc(nvariable, nvariable);
                    JTc.setZero();
                    Eigen::VectorXd JRc(nvariable, 1);
                    JRc.setZero();
                    Eigen::MatrixXd JTd(nvariable, nvariable);
                    JTd.setZero();
                    Eigen::VectorXd JRd(nvariable, 1);
                    JRd.setZero();

                    Eigen::MatrixXd JTt(nvariable, nvariable);
                    JTt.setZero();
                    Eigen::VectorXd JRt(nvariable, 1);
                    JRt.setZero();

                    bool sflag = camera_pose_option_with_detail(texture_views, current_label, adj_chart_label, faces, vertices, cur_charts,
                                       adj_chart, viewImageList, JTc, JRc, JTd, JRd, JTt, JRt);

 #pragma omp critical
                    {

                        c_JTJ += JTc; c_JTr += JRc;
                        d_JTJ += JTd; d_JTr += JRd;
                        t_JTJ += JTt; t_JTr += JRt;
                    }
                }//end adj chart
            }//end current same label chart

            Eigen::MatrixXd  A_JTJ(nvariable, nvariable);
            Eigen::VectorXd  b_JTr(nvariable, 1);
            A_JTJ.setZero();
            b_JTr.setZero();
            A_JTJ = c_JTJ*w_color*w_color + d_JTJ*w_depth*w_depth + t_JTJ*w_detail*w_detail;
            b_JTr = c_JTr*w_color + d_JTr*w_depth + t_JTr*w_detail;

            Eigen::VectorXd  x(6);

            x = -A_JTJ.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_JTr);
            Eigen::Affine3d aff_mat;
            aff_mat.linear() = (Eigen::Matrix3d) Eigen::AngleAxisd(x(2), Eigen::Vector3d::UnitZ())
                    * Eigen::AngleAxisd(x(1), Eigen::Vector3d::UnitY())
                    * Eigen::AngleAxisd(x(0), Eigen::Vector3d::UnitX());
            aff_mat.translation() = Eigen::Vector3d(x(3), x(4), x(5));

            Eigen::Matrix4f delta = aff_mat.matrix().cast<float>();

            Eigen::Matrix4f trans = delta* currentMat;
            texture_view.setWorldToCamMatrix(trans);
//            std::cout<<"------trans:"<<std::endl<<delta<<std::endl;
        }//end view

    }//end iter


#pragma omp parallel for
    for(int i = 0; i < num_faces; i++)//更新每个面的信息。
    {
        myFaceInfo face_info = faceInfoList[i];
        int face_label = face_info.lable;
        if(face_label != 0)
        {
            TextureView  texture_view =  texture_views.at(face_label - 1);
            math::Matrix4f  mat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  trans = mathToEigen(mat);
            faceInfoList[i].world_to_cam = trans;//
        }
    }
}

TEX_NAMESPACE_END
