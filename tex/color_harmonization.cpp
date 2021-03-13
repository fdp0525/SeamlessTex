/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */


#include "texturing.h"
#include "ll.h"
#include "mve/image_io.h"
#include "load_EXR.h"
#include "linear_bf.h"

TEX_NAMESPACE_BEGIN

typedef Image_file::EXR::image_type image_type;
typedef image_type::channel_type    channel_type;

/**
 * @brief Pix_RGB2LAB  像素点从rgb转化为LAB
 * @param r
 * @param g
 * @param b
 * @param L
 * @param a
 * @param b
 */
void Pix_RGB2LAB(uchar r, uchar g, uchar b, float &LL ,float &La, float &Lb)
{

    //rgb--->lms
    double L = 0.3844*r + 0.5783*g + 0.0402*b;
    double M = 0.1967*r + 0.7244*g + 0.0782*b;
    double S = 0.0241*r + 0.1288*g + 0.8444*b;

    if(L!=0)
    {
        L = std::log(L)/std::log(10);
    }

    if(M!=0)
    {
        M = std::log(M)/std::log(10);
    }

    if(S!=0)
    {
        S = std::log(S)/std::log(10);
    }

    LL = (L + M + S)/std::sqrt(3.0);
    La = (L + M - 2*S)/std::sqrt(6.0);
    Lb = (L - M)/std::sqrt(2.0);
    //    std::cout<<"LL:"<<LL<<" La:"<<La<<" Lb:"<<Lb<<std::endl;
}

/**
 * @brief Pix_RGB2LAB  像素点从LAB转化为rgb
 * @param r
 * @param g
 * @param b
 * @param L
 * @param a
 * @param b
 */
void Pix_LAB2RGB(float LL ,float La, float Lb, uchar &R, uchar &G, uchar &B)
{

    float l = LL/ std::sqrt(3);
    float a =La/ std::sqrt(6);
    float b =Lb/ std::sqrt(2);
    double L = l + a + b;
    double M = l + a - b;
    double S = l - 2*a;

    L = std::pow(10,L);
    M = std::pow(10,M);
    S = std::pow(10,S);

    double dR = 4.4679*L - 3.5873*M + 0.1193*S;
    double dG = -1.2186*L + 2.3809*M - 0.1624*S;
    double dB = 0.0497*L - 0.2439*M + 1.2045*S;

    //防止溢出，若求得RGB值大于255则置为255，若小于0则置为0
    if (dR>255)
        R=255;
    else if (dR<0)
        R=0;
    else
        R = uchar(dR);

    if (dG >255)
        G=255;
    else if (dG<0)
        G=0;
    else
        G = uchar(dG);

    if (dB>255)
        B=255;
    else if (dB<0)
        B=0;
    else
        B = uchar(dB);
}

void getInterLabFromImg(cv::Mat  img, float u, float v, float &Ll, float &La, float& Lb)
{
    int x = floor(u);
    int y = floor(v);
    float offsetx = u - x;
    float offsety = v - y;
    Ll = (1-offsetx)*(1-offsety)*img.at<cv::Vec3f>(y, x)[0]
            +(1-offsetx)*offsety*img.at<cv::Vec3f>(y+1, x)[0]
            +(1-offsety)*offsetx*img.at<cv::Vec3f>(y, x+1)[0]
            +offsetx*offsety*img.at<cv::Vec3f>(y + 1, x + 1)[0];

    La = (1-offsetx)*(1-offsety)*img.at<cv::Vec3f>(y, x)[1]
            +(1-offsetx)*offsety*img.at<cv::Vec3f>(y+1, x)[1]
            +(1-offsety)*offsetx*img.at<cv::Vec3f>(y, x+1)[1]
            +offsetx*offsety*img.at<cv::Vec3f>(y + 1, x + 1)[1];

    Lb = (1-offsetx)*(1-offsety)*img.at<cv::Vec3f>(y, x)[2]
            +(1-offsetx)*offsety*img.at<cv::Vec3f>(y+1, x)[2]
            +(1-offsety)*offsetx*img.at<cv::Vec3f>(y, x+1)[2]
            +offsetx*offsety*img.at<cv::Vec3f>(y + 1, x + 1)[2];
}


/**
 * @brief color_harmonization  利用优化的方法来进行光照一致性处理
 * @param graph
 * @param mesh
 * @param mesh_info
 * @param vertex_projection_infos
 * @param edge_vertex_infos
 * @param faceInfoList
 * @param subgraphs
 */
void color_harmonization(UniGraph const  & graph, mve::TriangleMesh::ConstPtr mesh,
                         std::vector<TextureView>  &texture_views,
                         std::vector<std::vector<std::size_t> > &subgraphs,
                         std::vector<int>  &labelPatchCount,
                         std::vector<std::vector<std::size_t> > &patch_graph,
                         std::vector<myViewImageInfo>&  viewImageList)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();//所有顶点
    std::size_t  const num_faces = faces.size()/3;//面的个数

    int iternum = 10;
    for(int iter = 0; iter < iternum; iter++)
    {
        //优化所有视口的光照参数
        for(int view_idx = 0; view_idx < texture_views.size(); view_idx++)
            //        int view_idx = 12;
        {
            std::cout<<"-------------viewid:"<<view_idx<<std::endl;

            int  nvariable = 2;

            Eigen::MatrixXd   JT_LL(nvariable, nvariable);//总的优化方程中的A
            Eigen::VectorXd   Jr_LL(nvariable);//总的优化方程中的b
            JT_LL.setZero();
            Jr_LL.setZero();

            Eigen::MatrixXd   JT_La(nvariable, nvariable);//总的优化方程中的A
            Eigen::VectorXd   Jr_La(nvariable);//总的优化方程中的b
            JT_La.setZero();
            Jr_La.setZero();

            Eigen::MatrixXd   JT_Lb(nvariable, nvariable);//总的优化方程中的A
            Eigen::VectorXd   Jr_Lb(nvariable);//总的优化方程中的b
            JT_Lb.setZero();
            Jr_Lb.setZero();

            int current_label = view_idx + 1;//当前视口的label
            TextureView  &texture_view =  texture_views.at(current_label - 1);//当前视口信息
            myViewImageInfo  &cur_view_infos = viewImageList.at(current_label - 1);
            math::Matrix4f  cmat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  currentMat = mathToEigen(cmat);//模型到当前视口的相机变换矩阵
            cv::Mat  currentLab = cur_view_infos.LabImg;

            int chartnum = labelPatchCount[view_idx];//当前标签在chart列表中个数

            std::cout<<"----------------chart num"<<chartnum<<std::endl;
            if(chartnum == 0)//当前视口没有被选为纹理，不需要处理
            {
                continue;
            }
            int chart_start = getChartIndex(labelPatchCount, current_label);//当前标签在chart列表中的起始位置

            for(int chart_idx = 0; chart_idx < chartnum; chart_idx++)//遍历当前视口所有的chart块
            {
                std::vector<std::size_t> cur_chart = subgraphs[chart_start+chart_idx];//取出当前chart块
                std::vector<std::size_t>   adj_patches = patch_graph[chart_start+chart_idx];//取出当前chart块相邻块索引
                if(adj_patches.size() == 0)//没有邻域不考虑
                {
                    continue;
                }

                std::cout<<"chart idx:"<<chart_idx<<" num:"<<cur_chart.size()<<std::endl;




                //当前patch投影到所有的相邻patch上应该光照一致
                for(int adj_chart_idx = 0; adj_chart_idx < adj_patches.size(); adj_chart_idx++)//遍历所有的邻接chart
                {
                    std::size_t  adj_chartidx = adj_patches[adj_chart_idx];//adj patch index
                    std::vector<std::size_t>  adj_chart = subgraphs[adj_chartidx];//邻接的块中所有的面（不存在与当前块相同标签的面）
                    int adj_label = getChartLabel(labelPatchCount, adj_chartidx, texture_views.size());//邻接chart的标签
//                    std::cout<<"---------------->adj_lable:"<<adj_label - 1<<std::endl;
                    myViewImageInfo  &adj_view_infos = viewImageList.at(adj_label - 1);

                    TextureView       adj_texture_view =  texture_views.at(adj_label - 1);//邻接块对应的视口
                    math::Matrix4f   amat =   adj_texture_view.getWorldToCamMatrix();
                    Eigen::Matrix4f   adj_chart_world2cam = mathToEigen(amat);

                    std::map<int, int>  computeFlag;
                    //存储每个相邻chart的残差
                    Eigen::MatrixXd   LJT(nvariable, nvariable);//
                    Eigen::VectorXd   LJr(nvariable);//
                    LJT.setZero();
                    LJr.setZero();

                    Eigen::MatrixXd   AJT(nvariable, nvariable);//
                    Eigen::VectorXd   AJr(nvariable);//
                    AJT.setZero();
                    AJr.setZero();

                    Eigen::MatrixXd   BJT(nvariable, nvariable);//
                    Eigen::VectorXd   BJr(nvariable);//
                    BJT.setZero();
                    BJr.setZero();

                    int count = 0;
                    for(int face_idx = 0; face_idx < cur_chart.size(); face_idx++)//遍历当前chart上所有的顶点
                    {
                        int f_i = cur_chart[face_idx];//当前面面索引
                        for(int v_idx = 0; v_idx < 3; v_idx++)//取出面上的三个顶点
                        {
                            int   f_v_idx = faces[f_i*3 + v_idx];//面上顶点索引
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


                            //投影到邻接平面上
                            Eigen::Vector4f    p = adj_chart_world2cam * q0;
                            //                            Eigen::Vector4f    p = currentMat * q0;

                            float v_1 = p(0) * G2LTexConfig::get().IMAGE_FX / p(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_1 = p(1) * G2LTexConfig::get().IMAGE_FY / p(2) + G2LTexConfig::get().IMAGE_CY;
                            if( v_0 > 0 && v_0 < G2LTexConfig::get().IMAGE_WIDTH-1 &&
                                    u_0 > 0 && u_0 < G2LTexConfig::get().IMAGE_HEIGHT-1 &&
                                    v_1 > 0 && v_1 < G2LTexConfig::get().IMAGE_WIDTH-1 &&
                                    u_1 > 0 && u_1 < G2LTexConfig::get().IMAGE_HEIGHT-1)//ablity防止插值越界
                            {

                                //判断是否可见
                                float dv = p(0) * G2LTexConfig::get().DEPTH_FX / p(2) + G2LTexConfig::get().DEPTH_CX;
                                float du = p(1) * G2LTexConfig::get().DEPTH_FY / p(2) + G2LTexConfig::get().DEPTH_CY;
                                //                                if(dv < 0 || dv > G2LTexConfig::get().DEPTH_WIDTH - 1 ||
                                //                                        du < 0 || du > G2LTexConfig::get().DEPTH_HEIGHT - 1)
                                //                                {
                                //                                    continue;
                                //                                }

                                //                                std::cout<<"p2 :"<<p(2)<<"---------->"<<adj_view_infos.depth.at<ushort>(du, dv)/1000.0f<<std::endl;
                                //                                if(std::abs(p(2) - adj_view_infos.depth.at<ushort>(du, dv)/1000.0f) > 0.08)//不可见
                                //                                {
                                //                                    continue;
                                //                                }
                                count++;
                                float cur_L, cur_a, cur_b;
                                //                                getInterLabFromImg(currentLab, v_0, u_0, cur_L, cur_a, cur_b);
                                //                                cur_L = currentLab.at<cv::Vec3b>(u_0, v_0)[0]/255.0f;
                                //                                cur_a = currentLab.at<cv::Vec3b>(u_0, v_0)[1]/255.0f;
                                //                                cur_b = currentLab.at<cv::Vec3b>(u_0, v_0)[2]/255.0f;
                                cur_L = currentLab.at<cv::Vec3f>(u_0, v_0)[0];
                                cur_a = currentLab.at<cv::Vec3f>(u_0, v_0)[1];
                                cur_b = currentLab.at<cv::Vec3f>(u_0, v_0)[2];

                                //                                std::cout<<"curr r:"<<(int)cur_view_infos.img.at<cv::Vec3b>(u_0, v_0)[0]<<" g:"<<(int)cur_view_infos.img.at<cv::Vec3b>(u_0, v_0)[1]<<" b:"<<(int)cur_view_infos.img.at<cv::Vec3b>(u_0, v_0)[2]<<std::endl;
                                //                                std::cout<<"adj r:"<<(int)adj_view_infos.img.at<cv::Vec3b>(u_1, v_1)[0]<<" g:"<<(int)adj_view_infos.img.at<cv::Vec3b>(u_1, v_1)[1]<<" b:"<<(int)adj_view_infos.img.at<cv::Vec3b>(u_1, v_1)[2]<<std::endl;
                                //                                std::cout<<"1 adj r:"<<(int)adj_view_infos.img.at<cv::Vec3b>(u_1, v_1)[0]<<" g:"<<(int)adj_view_infos.img.at<cv::Vec3b>(u_1, v_1)[1]<<" b:"<<(int)adj_view_infos.img.at<cv::Vec3b>(u_1, v_1)[2]<<std::endl;

                                float adj_L, adj_a, adj_b;
                                //                                getInterLabFromImg(adj_view_infos.LabImg, v_1, u_1, adj_L, adj_a, adj_b);
                                //                                adj_L = adj_view_infos.LabImg.at<cv::Vec3b>(u_1, v_1)[0]/255.0f;
                                //                                adj_a = adj_view_infos.LabImg.at<cv::Vec3b>(u_1, v_1)[1]/255.0f;
                                //                                adj_b = adj_view_infos.LabImg.at<cv::Vec3b>(u_1, v_1)[2]/255.0f;
                                adj_L = adj_view_infos.LabImg.at<cv::Vec3f>(u_1, v_1)[0];
                                adj_a = adj_view_infos.LabImg.at<cv::Vec3f>(u_1, v_1)[1];
                                adj_b = adj_view_infos.LabImg.at<cv::Vec3f>(u_1, v_1)[2];

                                Eigen::VectorXd   J(nvariable);//
                                J.setZero();
                                //L
                                float Lr = cur_L*cur_view_infos.L_alpha + cur_view_infos.L_beta - adj_L*adj_view_infos.L_alpha - adj_view_infos.L_beta;
                                J(0) = cur_L;
                                J(1) = 1;
                                LJT += J*J.transpose();
                                LJr += J*Lr;

                                //alpha
                                J.setZero();
                                float Ar = cur_a*cur_view_infos.A_alpha + cur_view_infos.A_beta - adj_a*adj_view_infos.A_alpha - adj_view_infos.A_beta;
                                J(0) = cur_a;
                                J(1) = 1;
                                AJT += J*J.transpose();
                                AJr += J*Ar;

                                //beta
                                J.setZero();
                                float Br = cur_b*cur_view_infos.B_alpha + cur_view_infos.B_beta - adj_b*adj_view_infos.B_alpha - adj_view_infos.B_beta;
                                J(0) = cur_b;
                                J(1) = 1;
                                BJT += J*J.transpose();
                                BJr += J*Br;
                            }
                        }
                    }
                    if(count == 0)
                    {
                        continue;
                    }
                    //L reg
                    Eigen::MatrixXd   LReg(nvariable, nvariable);//
                    Eigen::VectorXd   Lr(nvariable);//
                    LReg.setZero();
                    Lr.setZero();
                    Eigen::VectorXd   Jreg(nvariable);//
                    Jreg.setZero();
                    float l_r1 = cur_view_infos.L_alpha;
                    Jreg(0) = 1;
                    Jreg(1) = 0;
                    LReg = Jreg*Jreg.transpose() ;
                    Lr = Jreg*l_r1;
                    Jreg.setZero();
                    Jreg(0) = 0;
                    Jreg(1) = 1;
                    float l_r2 = cur_view_infos.L_beta;
                    LReg += Jreg*Jreg.transpose();
                    Lr += Jreg*l_r2;

                    //alpha reg
                    Eigen::MatrixXd   AReg(nvariable, nvariable);//
                    Eigen::VectorXd   Ar(nvariable);//
                    AReg.setZero();
                    Ar.setZero();
                    Jreg.setZero();
                    float A_r1 = cur_view_infos.A_alpha;
                    Jreg(0) = 1;
                    Jreg(1) = 0;
                    AReg = Jreg*Jreg.transpose() ;
                    Ar = Jreg*A_r1;
                    Jreg.setZero();
                    Jreg(0) = 0;
                    Jreg(1) = 1;
                    float A_r2 = cur_view_infos.A_beta;
                    AReg += Jreg*Jreg.transpose();
                    Ar += Jreg*A_r2;

                    //beta reg
                    Eigen::MatrixXd   BReg(nvariable, nvariable);//
                    Eigen::VectorXd   Br(nvariable);//
                    BReg.setZero();
                    Br.setZero();
                    Jreg.setZero();
                    float B_r1 = cur_view_infos.B_alpha;
                    Jreg(0) = 1;
                    Jreg(1) = 0;
                    BReg = Jreg*Jreg.transpose() ;
                    Br = Jreg*B_r1;
                    Jreg.setZero();
                    Jreg(0) = 0;
                    Jreg(1) = 1;
                    float B_r2 = cur_view_infos.B_beta;
                    BReg += Jreg*Jreg.transpose();
                    Br += Jreg*B_r2;

                    std::cout<<"count:"<<count<<std::endl;

                    if(view_idx == 12)
                    {
                        //                    std::cout<<"--------------------LJT"<<std::endl;
                        //                    std::cout<<LJT<<std::endl;
                        //                    std::cout<<LReg<<std::endl;
                        //                    std::cout<<LJr<<std::endl;
                        //                    std::cout<<Lr<<std::endl;
                    }


                    //total
                    float lamd1 = 1.0f/count;
                    float lamd2 = 10;
                    JT_LL += lamd1*lamd1*LJT + lamd2*lamd2*LReg;
                    Jr_LL += lamd1*LJr + lamd2*Lr;

                    if(view_idx == 12)
                    {
                        //                    std::cout<<"--------------------JT_La"<<std::endl;
                        //                    std::cout<<AJT<<std::endl;
                        //                    std::cout<<AReg<<std::endl;
                        //                    std::cout<<AJr<<std::endl;
                        //                    std::cout<<Ar<<std::endl;
                    }

                    JT_La += lamd1*lamd1*AJT + lamd2*lamd2*AReg;
                    Jr_La += lamd1*AJr + lamd2*Ar;

                    if(view_idx == 12)
                    {
                        //                    std::cout<<"--------------------JT_Lb"<<std::endl;
                        //                    std::cout<<BJT<<std::endl;
                        //                    std::cout<<BReg<<std::endl;
                        //                    std::cout<<BJr<<std::endl;
                        //                    std::cout<<Br<<std::endl<<std::endl;
                    }

                    JT_Lb += lamd1*lamd1*BJT + lamd2*lamd2*BReg;
                    Jr_Lb += lamd1*BJr + lamd2*Br;
                }
            }

            //solve
            Eigen::VectorXd  Lx(2);
            Lx = -JT_LL.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Jr_LL);
            Eigen::VectorXd  Ax(2);
            Ax = -JT_La.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Jr_La);
            Eigen::VectorXd  Bx(2);
            Bx = -JT_Lb.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(Jr_Lb);

            //update
            cur_view_infos.L_alpha += Lx(0);
            cur_view_infos.L_beta += Lx(1);

            cur_view_infos.A_alpha += Ax(0);
            cur_view_infos.A_beta += Ax(1);

            cur_view_infos.B_alpha += Bx(0);
            cur_view_infos.B_beta += Bx(1);
        }

    }

    std::cout<<"-------------after optimization-------------"<<std::endl;
    for(int view_idx = 0; view_idx < texture_views.size(); view_idx++)
    {
        myViewImageInfo  &cur_view_infos = viewImageList.at(view_idx);
        std::cout<<"view id:"<<view_idx<<"L a:"<<cur_view_infos.L_alpha<<" L b:"<<cur_view_infos.L_beta<<
                   "A a:"<<cur_view_infos.A_alpha<<" A b:"<<cur_view_infos.A_beta<<
                   "B a:"<<cur_view_infos.B_alpha<<" B b:"<<cur_view_infos.B_beta<<std::endl;

        cv::Mat  img = cv::Mat(cur_view_infos.LabImg.rows, cur_view_infos.LabImg.cols, CV_8UC3);

        for(int i = 0; i < cur_view_infos.LabImg.rows; i++)
        {
            for(int j = 0; j<cur_view_infos.LabImg.cols; j++)
            {
                float LL = cur_view_infos.LabImg.at<cv::Vec3f>(i, j)[0];
                LL = LL*cur_view_infos.L_alpha +cur_view_infos.L_beta;
                float La = cur_view_infos.LabImg.at<cv::Vec3f>(i, j)[1];
                La = La*cur_view_infos.A_alpha + cur_view_infos.A_beta;
                float Lb = cur_view_infos.LabImg.at<cv::Vec3f>(i, j)[2];
                Lb = Lb*cur_view_infos.B_alpha + cur_view_infos.B_beta;
                uchar r, g, b;
                Pix_LAB2RGB(LL,La,Lb, r, g, b);
                img.at<cv::Vec3b>(i,j)[0] = r;
                img.at<cv::Vec3b>(i,j)[1] = g;
                img.at<cv::Vec3b>(i,j)[2] = b;
            }
        }
        char buf[256];
        sprintf(buf,"%02d.png",view_idx);
        cv::imwrite(buf, img);


    }

}


void color_harmonization_ceres(UniGraph const  & graph, mve::TriangleMesh::ConstPtr mesh,
                               std::vector<TextureView>  &texture_views,
                               std::vector<std::vector<std::size_t> > &subgraphs,
                               std::vector<int>  &labelPatchCount,
                               std::vector<std::vector<std::size_t> > &patch_graph,
                               std::vector<myViewImageInfo>&  viewImageList)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();//所有顶点
    std::size_t  const num_faces = faces.size()/3;//面的个数

    int iternum = 10;
    for(int iter = 0; iter < iternum; iter++)
    {
        for(int view_idx = 0; view_idx < texture_views.size(); view_idx++)
        {
            //
//            ceres::Problem  problemLL;

            ColorHarmonValue   colorharmLL;
            ColorHarmonValue   colorharmLa;
            ColorHarmonValue   colorharmLb;

            std::cout<<"-------------viewid:"<<view_idx<<std::endl;
            int current_label = view_idx + 1;//当前视口的label
            TextureView  &texture_view =  texture_views.at(current_label - 1);//当前视口信息
            myViewImageInfo  &cur_view_infos = viewImageList.at(current_label - 1);
            math::Matrix4f  cmat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  currentMat = mathToEigen(cmat);//模型到当前视口的相机变换矩阵
            cv::Mat  currentLab = cur_view_infos.LabImg;

            int chartnum = labelPatchCount[view_idx];//当前标签在chart列表中个数
            std::cout<<"----------------chart num"<<chartnum<<std::endl;
            if(chartnum == 0)//当前视口没有被选为纹理，不需要处理
            {
                continue;
            }
            int chart_start = getChartIndex(labelPatchCount, current_label);//当前标签在chart列表中的起始位置

            for(int chart_idx = 0; chart_idx < chartnum; chart_idx++)//遍历当前视口所有的chart块
            {
                std::vector<std::size_t> cur_chart = subgraphs[chart_start+chart_idx];//取出当前chart块
                std::vector<std::size_t>   adj_patches = patch_graph[chart_start+chart_idx];//取出当前chart块相邻块索引
                if(adj_patches.size() == 0)//没有邻域不考虑
                {
                    continue;
                }

                //当前patch投影到所有的相邻patch上应该光照一致
                for(int adj_chart_idx = 0; adj_chart_idx < adj_patches.size(); adj_chart_idx++)//遍历所有的邻接chart
                {
                    std::size_t  adj_chartidx = adj_patches[adj_chart_idx];//adj patch index
                    std::vector<std::size_t>  adj_chart = subgraphs[adj_chartidx];//邻接的块中所有的面（不存在与当前块相同标签的面）
                    int adj_label = getChartLabel(labelPatchCount, adj_chartidx, texture_views.size());//邻接chart的标签
//                    std::cout<<"---------------->adj_lable:"<<adj_label - 1<<std::endl;
                    myViewImageInfo  &adj_view_infos = viewImageList.at(adj_label - 1);

                    TextureView       adj_texture_view =  texture_views.at(adj_label - 1);//邻接块对应的视口
                    math::Matrix4f   amat =   adj_texture_view.getWorldToCamMatrix();
                    Eigen::Matrix4f   adj_chart_world2cam = mathToEigen(amat);

                    std::map<int, int>  computeFlag;

                    int count = 0;
                    for(int face_idx = 0; face_idx < cur_chart.size(); face_idx++)//遍历当前chart上所有的顶点
                    {
                        int f_i = cur_chart[face_idx];//当前面面索引
                        for(int v_idx = 0; v_idx < 3; v_idx++)//取出面上的三个顶点
                        {
                            int   f_v_idx = faces[f_i*3 + v_idx];//面上顶点索引
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


                            //投影到邻接平面上
                            Eigen::Vector4f    p = adj_chart_world2cam * q0;
                            //                            Eigen::Vector4f    p = currentMat * q0;

                            float v_1 = p(0) * G2LTexConfig::get().IMAGE_FX / p(2) + G2LTexConfig::get().IMAGE_CX;
                            float u_1 = p(1) * G2LTexConfig::get().IMAGE_FY / p(2) + G2LTexConfig::get().IMAGE_CY;
                            if( v_0 > 0 && v_0 < G2LTexConfig::get().IMAGE_WIDTH-1 &&
                                    u_0 > 0 && u_0 < G2LTexConfig::get().IMAGE_HEIGHT-1 &&
                                    v_1 > 0 && v_1 < G2LTexConfig::get().IMAGE_WIDTH-1 &&
                                    u_1 > 0 && u_1 < G2LTexConfig::get().IMAGE_HEIGHT-1)//ablity防止插值越界
                            {

                                //判断是否可见
                                float dv = p(0) * G2LTexConfig::get().DEPTH_FX / p(2) + G2LTexConfig::get().DEPTH_CX;
                                float du = p(1) * G2LTexConfig::get().DEPTH_FY / p(2) + G2LTexConfig::get().DEPTH_CY;

                                float cur_L, cur_a, cur_b;
                                cur_L = currentLab.at<cv::Vec3f>(u_0, v_0)[0];
                                cur_a = currentLab.at<cv::Vec3f>(u_0, v_0)[1];
                                cur_b = currentLab.at<cv::Vec3f>(u_0, v_0)[2];

                                float adj_L, adj_a, adj_b;
                                adj_L = adj_view_infos.LabImg.at<cv::Vec3f>(u_1, v_1)[0];
                                adj_a = adj_view_infos.LabImg.at<cv::Vec3f>(u_1, v_1)[1];
                                adj_b = adj_view_infos.LabImg.at<cv::Vec3f>(u_1, v_1)[2];

                                if(cur_L < 98.0f && adj_L < 98.0f)
                                {
//                                    count++;
//                                    ceres::LossFunction* lossfunction = new ceres::CauchyLoss(1.0);//new ceres::CauchyLoss(1.0);
//                                    ceres::CostFunction *cost_function1_l = colorCostFunctor1::Create(l1, cur_L, adj_L);
//                                    problem.AddResidualBlock(cost_function1_l, lossfunction, scalesi, offsetsi, scalesj, offsetsj );//三通道
                                    Lab_L2  llv;
                                    llv.A_v = cur_L;
                                    llv.B_v = adj_L;
                                    llv.A_a = adj_view_infos.L_alpha;
                                    llv.A_b = adj_view_infos.L_beta;
                                    colorharmLL.mypush_back(llv);

                                    Lab_L2  lav;
                                    lav.A_v = cur_a;
                                    lav.B_v = adj_a;
                                    lav.A_a = adj_view_infos.A_alpha;
                                    lav.A_b = adj_view_infos.A_beta;
                                    colorharmLa.mypush_back(lav);

                                    Lab_L2  lbv;
                                    lbv.A_v = cur_b;
                                    lbv.B_v = adj_b;
                                    lbv.A_a = adj_view_infos.B_alpha;
                                    lbv.A_b = adj_view_infos.B_beta;
                                    colorharmLb.mypush_back(lbv);
                                }
                            }
                        }
                    }
                }

            }


            //optimization
            //ll
            ceres::Problem  Lproblem;
            ceres::Problem  Aproblem;
            ceres::Problem  Bproblem;

            double lweight = 1.0 / colorharmLL.nums;
//            lweight = lweight/100.0;
            double Lalpha = cur_view_infos.L_alpha;
            double Lbeta = cur_view_infos.L_beta;
            std::cout<<"before L a:"<<Lalpha<<" b:"<<Lbeta<<"  nums:"<<colorharmLL.nums<<std::endl;

            double aweight = 1.0 / colorharmLa.nums;
//            aweight = aweight/100.0f;
            double Aalpha = cur_view_infos.A_alpha;
            double Abeta = cur_view_infos.A_beta;
            std::cout<<"before A a:"<<Aalpha<<" b:"<<Abeta<<"  nums:"<<colorharmLa.nums<<std::endl;

            double bweight = 1.0 / colorharmLb.nums;
//            bweight = bweight/100.0f;
            double Balpha = cur_view_infos.B_alpha;
            double Bbeta = cur_view_infos.B_beta;
            std::cout<<"before B a:"<<Balpha<<" b:"<<Bbeta<<"  nums:"<<colorharmLb.nums<<std::endl;

            //LL
           for(int m_i = 0; m_i<colorharmLL.nums; m_i++)
           {
               Lab_L2  lab_l = colorharmLL.at(m_i);
               ceres::LossFunction* lossfunction = new ceres::CauchyLoss(1.0);//new ceres::CauchyLoss(1.0);
               ceres::CostFunction *cost_function1_l = colorCostFunctor1::create(lweight, lab_l.A_v, lab_l.B_v, lab_l.A_a, lab_l.A_b);
               Lproblem.AddResidualBlock(cost_function1_l, lossfunction, &Lalpha, &Lbeta);//三通道
           }

           //La
           for(int m_i = 0; m_i<colorharmLa.nums; m_i++)
           {
               Lab_L2  lab_l = colorharmLa.at(m_i);
               ceres::LossFunction* lossfunction = new ceres::CauchyLoss(1.0);//new ceres::CauchyLoss(1.0);
               ceres::CostFunction *cost_function1_l = colorCostFunctor1::create(aweight, lab_l.A_v, lab_l.B_v, lab_l.A_a, lab_l.A_b);
               Aproblem.AddResidualBlock(cost_function1_l, lossfunction, &Aalpha, &Abeta);//三通道
           }

           //Lb
           for(int m_i = 0; m_i<colorharmLb.nums; m_i++)
           {
               Lab_L2  lab_l = colorharmLb.at(m_i);
               ceres::LossFunction* lossfunction = new ceres::CauchyLoss(1.0);//new ceres::CauchyLoss(1.0);
               ceres::CostFunction *cost_function1_l = colorCostFunctor1::create(bweight, lab_l.A_v, lab_l.B_v, lab_l.A_a, lab_l.A_b);
               Bproblem.AddResidualBlock(cost_function1_l, lossfunction, &Balpha, &Bbeta);//三通道
           }

           double l2 = 0.33;
           ceres::CostFunction *cost_function2 = colorCostFunctor2::Create(l2);
           Lproblem.AddResidualBlock(cost_function2, NULL, &Lalpha, &Lbeta);

           ceres::CostFunction *cost_function3 = colorCostFunctor2::Create(l2);
           Aproblem.AddResidualBlock(cost_function3, NULL, &Aalpha, &Abeta);

           ceres::CostFunction *cost_function4 = colorCostFunctor2::Create(l2);
           Bproblem.AddResidualBlock(cost_function4, NULL, &Balpha, &Bbeta);

           //LL
           {
               ///////////////////////////Solver/////////////////////////////////////////////////////////
               ceres::Solver::Options options;
               options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
               options.linear_solver_type = ceres::ITERATIVE_SCHUR;//ceres::DENSE_SCHUR; //ceres::LEVENBERG_MARQUARDT;
               //options.linear_solver_type = ceres::ITERATIVE_SCHUR;

               options.minimizer_progress_to_stdout = false;
               options.max_num_iterations = 100;//100;
               options.num_threads = 2;
               options.num_linear_solver_threads = 2;

               ceres::Solver::Summary summary;
               ceres::Solve(options, &Lproblem, &summary);
               //             std::cout << summary.FullReport() << "\n";
               cur_view_infos.L_alpha = Lalpha;
               cur_view_infos.L_beta = Lbeta;
               std::cout<<"after La:"<<Lalpha<<" Lb:"<<Lbeta<<std::endl;
           }

           //La
           {
               ///////////////////////////Solver/////////////////////////////////////////////////////////
               ceres::Solver::Options options;
               options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
               options.linear_solver_type = ceres::ITERATIVE_SCHUR;//ceres::DENSE_SCHUR; //ceres::LEVENBERG_MARQUARDT;
               //options.linear_solver_type = ceres::ITERATIVE_SCHUR;

               options.minimizer_progress_to_stdout = false;
               options.max_num_iterations = 100;//100;
               options.num_threads = 2;
               options.num_linear_solver_threads = 2;

               ceres::Solver::Summary summary;
               ceres::Solve(options, &Aproblem, &summary);
               //             std::cout << summary.FullReport() << "\n";
               cur_view_infos.A_alpha = Aalpha;
               cur_view_infos.A_beta = Abeta;
               std::cout<<"after Aa:"<<Aalpha<<" Ab:"<<Abeta<<std::endl;
           }

           //Lb
           {
               ///////////////////////////Solver/////////////////////////////////////////////////////////
               ceres::Solver::Options options;
               options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
               options.linear_solver_type = ceres::ITERATIVE_SCHUR;//ceres::DENSE_SCHUR; //ceres::LEVENBERG_MARQUARDT;
               //options.linear_solver_type = ceres::ITERATIVE_SCHUR;

               options.minimizer_progress_to_stdout = false;
               options.max_num_iterations = 100;//100;
               options.num_threads = 2;
               options.num_linear_solver_threads = 2;

               ceres::Solver::Summary summary;
               ceres::Solve(options, &Bproblem, &summary);
               //             std::cout << summary.FullReport() << "\n";
               cur_view_infos.B_alpha = Balpha;
               cur_view_infos.B_beta = Bbeta;
               std::cout<<"after a:"<<Balpha<<" b:"<<Bbeta<<std::endl;
           }


        }//end view

    }//end iter

    for(int i = 0;i<texture_views.size();i++)
    {
        TextureView  &texture_view =  texture_views.at(i);//当前视口信息
        myViewImageInfo  &cur_view_infos = viewImageList.at(i);
        cv::Mat  labimg = cv::Mat(texture_view.height, texture_view.width, CV_32FC3);
        std::cout<<"La:"<<cur_view_infos.L_alpha<<" Lb:"<<cur_view_infos.L_beta<<std::endl;
        std::cout<<"Aa:"<<cur_view_infos.A_alpha<<" Ab:"<<cur_view_infos.A_beta<<std::endl;
        std::cout<<"Ba:"<<cur_view_infos.B_alpha<<" Bb:"<<cur_view_infos.B_beta<<std::endl;

        for(int h = 0; h<texture_view.height;h++)
        {
            for(int w = 0; w< texture_view.width; w++)
            {
                float Lvalue = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[0];
                if(Lvalue < 98.0f)
                {
                    labimg.at<cv::Vec3f>(h,w)[0] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[0]*cur_view_infos.L_alpha+cur_view_infos.L_beta;
//                    labimg.at<cv::Vec3f>(h,w)[1] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[1]*cur_view_infos.A_alpha+cur_view_infos.A_beta;
//                    labimg.at<cv::Vec3f>(h,w)[2] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[2]*cur_view_infos.B_alpha+cur_view_infos.B_beta;
                    labimg.at<cv::Vec3f>(h,w)[1] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[1];
                    labimg.at<cv::Vec3f>(h,w)[2] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[2];

                }
                else
                {
                    labimg.at<cv::Vec3f>(h,w)[0] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[0];
//                    labimg.at<cv::Vec3f>(h,w)[1] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[1]*cur_view_infos.A_alpha+cur_view_infos.A_beta;
//                    labimg.at<cv::Vec3f>(h,w)[2] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[2]*cur_view_infos.B_alpha+cur_view_infos.B_beta;
                    labimg.at<cv::Vec3f>(h,w)[1] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[1];
                    labimg.at<cv::Vec3f>(h,w)[2] = cur_view_infos.LabImg.at<cv::Vec3f>(h,w)[2];
                }
            }
        }
        cv::Mat  rgbimg;
        cv::cvtColor(labimg, rgbimg, CV_Lab2RGB);//
        cv::Mat resultimg;
        rgbimg.convertTo(resultimg,CV_8UC3, 255.0f);
        char buf[256];
        sprintf(buf,"ch%02d.png",i);
        cv::imwrite(buf,resultimg);

        for(int h = 0; h < G2LTexConfig::get().IMAGE_HEIGHT; h++)
        {
            for(int w = 0; w < G2LTexConfig::get().IMAGE_WIDTH; w++)
            {
                texture_view.image->at(w, h, 0) = resultimg.at<cv::Vec3b>(h, w)[2];
                texture_view.image->at(w, h, 1) = resultimg.at<cv::Vec3b>(h, w)[1];
                texture_view.image->at(w, h, 2) = resultimg.at<cv::Vec3b>(h, w)[0];
            }
        }
        sprintf(buf,"chimg%02d.jpg", i);
        mve::image::save_file(texture_view.image, buf);
    }
}

double log_function(const double x){

    static const double inv_log_base = 1.0 / log(10.0);

    return log(x) * inv_log_base;
}


double exp_function(const double x){

    return pow(10.0,x);
}

void slipBaseandDetailImage(cv::Mat img, cv::Mat &base, cv::Mat &detail)
{
//    cv::Mat  img = cv::imread(filename);
//    outimg = cv::Mat(img.rows, img.cols, CV_8UC4, cv::Scalar(0,0,0,0));
    base = cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(0.0f));    
    detail = cv::Mat(img.rows, img.cols, CV_32FC1, cv::Scalar(0.0f));
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
            double kk = 1.0f+(float(img.at<cv::Vec3b>(i, j)[0])*0.3f+ float(img.at<cv::Vec3b>(i, j)[1])*0.59f + float(img.at<cv::Vec3b>(i, j)[2])*0.11);
//            double kk = (float(img.at<cv::Vec3b>(i, j)[0])*20.0f+ float(img.at<cv::Vec3b>(i, j)[1])*40.0 + float(img.at<cv::Vec3b>(i, j)[2])) / 61.0f;

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
        detail.at<float>(count%height, count/height) =  (*l) - (*f);
// detailimg.at<float>(count%width, count/width) =  (*l) - (*f);
        base.at<float>(count%height, count/height) = *f;
        count++;
    }

}


void color_harmonization_with_detail(UniGraph const  & graph, mve::TriangleMesh::ConstPtr mesh,
                         std::vector<TextureView>  &texture_views,
                         std::vector<std::vector<std::size_t> > &subgraphs,
                         std::vector<int>  &labelPatchCount,
                         std::vector<std::vector<std::size_t> > &patch_graph,
                         std::vector<myViewImageInfo>&  viewImageList)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();//所有顶点
    std::size_t  const num_faces = faces.size()/3;//面的个数
    
    //分离基本层和细节层并对基本层进行光照处理
    std::vector<BaseAndDetail>   baseimgs;
    for(int view_id = 0; view_id < texture_views.size(); view_id++)
    {
        TextureView  &texture_view =  texture_views.at(view_id);
        
        BaseAndDetail  imgs;
        
        slipBaseandDetailImage(texture_view.targeImage, imgs.baseLayer, imgs.detailLayer);
        
        baseimgs.push_back(imgs);
    }
    
    int iternum = 10;
    for(int iter = 0; iter < iternum; iter++)
    {
        for(int view_idx = 0; view_idx < texture_views.size(); view_idx++)
        {
            BaseAndDetail   &curbaseimg = baseimgs.at(view_idx);
            ColorHarmonValue   colorharmLL;
            int current_label = view_idx + 1;//当前视口的label
            TextureView  &texture_view =  texture_views.at(current_label - 1);//当前视口信息
            myViewImageInfo  &cur_view_infos = viewImageList.at(current_label - 1);
            cv::Mat  cur_depth = cur_view_infos.depth;

            math::Matrix4f  cmat = texture_view.getWorldToCamMatrix();
            Eigen::Matrix4f  currentMat = mathToEigen(cmat);//模型到当前视口的相机变换矩阵
            cv::Mat  currentLab = cur_view_infos.LabImg;

//            std::cout<<"id:"<<view_idx<<" a:"<<cur_view_infos.L_alpha<<" b:"<<cur_view_infos.L_beta<<std::endl;
            
            int chartnum = labelPatchCount[view_idx];//当前标签在chart列表中个数
            if(chartnum == 0)//当前视口没有被选为纹理，不需要处理
            {
                continue;
            }
            int chart_start = getChartIndex(labelPatchCount, current_label);//当前标签在chart列表中的起始位置
            for(int chart_idx = 0; chart_idx < chartnum; chart_idx++)//遍历当前视口所有的chart块
            {
                std::vector<std::size_t> cur_chart = subgraphs[chart_start+chart_idx];//取出当前chart块
                std::vector<std::size_t>   adj_patches = patch_graph[chart_start+chart_idx];//取出当前chart块相邻块索引
                if(adj_patches.size() == 0)//没有邻域不考虑
                {
                    continue;
                }
                
                //当前patch投影到所有的相邻patch上应该光照一致
                for(int adj_chart_idx = 0; adj_chart_idx < adj_patches.size(); adj_chart_idx++)//遍历所有的邻接chart
                {
                    std::size_t  adj_chartidx = adj_patches[adj_chart_idx];//adj patch index
                    std::vector<std::size_t>  adj_chart = subgraphs[adj_chartidx];//邻接的块中所有的面（不存在与当前块相同标签的面）
                    int adj_label = getChartLabel(labelPatchCount, adj_chartidx, texture_views.size());//邻接chart的标签
                    if(adj_label == 0)
                    {
                        continue;
                    }
                    myViewImageInfo  &adj_view_infos = viewImageList.at(adj_label - 1);
                    TextureView       adj_texture_view =  texture_views.at(adj_label - 1);//邻接块对应的视口
                    BaseAndDetail   &adjbaseimg = baseimgs.at(adj_label - 1);

                    cv::Mat  adj_depth = adj_view_infos.depth;

                    math::Matrix4f   amat =   adj_texture_view.getWorldToCamMatrix();
                    Eigen::Matrix4f   adj_chart_world2cam = mathToEigen(amat);
                     std::map<int, int>  computeFlag;
                     for(int face_idx = 0; face_idx < cur_chart.size(); face_idx++)//遍历当前chart上所有的顶点
                     {
                         int f_i = cur_chart[face_idx];//当前面面索引
                         for(int v_idx = 0; v_idx < 3; v_idx++)//取出面上的三个顶点
                         {
                             int   f_v_idx = faces[f_i*3 + v_idx];//面上顶点索引
                             if(computeFlag.count(f_v_idx) != 0)//已经计算过
                             {
                                 continue;
                             }
                             computeFlag[f_v_idx] = 100;
                             
                             math::Vec3f       v_pos = vertices[f_v_idx];
                             Eigen::Vector4f   q0;
                             q0<<v_pos(0), v_pos(1), v_pos(2), 1;//三维点
                             Eigen::Vector4f   q = currentMat * q0;//把顶点变换到相机坐标
                             
                             //color
                             float v_0 = q(0) * G2LTexConfig::get().IMAGE_FX / q(2) + G2LTexConfig::get().IMAGE_CX;
                             float u_0 = q(1) * G2LTexConfig::get().IMAGE_FY / q(2) + G2LTexConfig::get().IMAGE_CY;
                             
                             float dv0 = q(0) * G2LTexConfig::get().DEPTH_FX / q(2) + G2LTexConfig::get().DEPTH_CX;
                             float du0 = q(1) * G2LTexConfig::get().DEPTH_FY / q(2) + G2LTexConfig::get().DEPTH_CY;
                             
                             //投影到邻接平面上
                             Eigen::Vector4f    p = adj_chart_world2cam * q0;
                             float v_1 = p(0) * G2LTexConfig::get().IMAGE_FX / p(2) + G2LTexConfig::get().IMAGE_CX;
                             float u_1 = p(1) * G2LTexConfig::get().IMAGE_FY / p(2) + G2LTexConfig::get().IMAGE_CY;
                             
                             float dv1 = p(0) * G2LTexConfig::get().DEPTH_FX / p(2) + G2LTexConfig::get().DEPTH_CX;
                             float du1 = p(1) * G2LTexConfig::get().DEPTH_FY / p(2) + G2LTexConfig::get().DEPTH_CY;
                             
                             if( v_0 >= 0 && v_0 <= G2LTexConfig::get().IMAGE_WIDTH-1 &&
                                     u_0 >= 0 && u_0 <= G2LTexConfig::get().IMAGE_HEIGHT-1 &&
                                     v_1 >= 0 && v_1 <= G2LTexConfig::get().IMAGE_WIDTH-1 &&
                                     u_1 >= 0 && u_1 <= G2LTexConfig::get().IMAGE_HEIGHT-1)//ablity防止插值越界
                             {
                                 if(dv0 < 0 || dv0 > G2LTexConfig::get().DEPTH_WIDTH - 1 ||
                                         du0 <0 && du0 > G2LTexConfig::get().DEPTH_HEIGHT- 1 ||
                                         dv1 < 0 && dv1 > G2LTexConfig::get().DEPTH_WIDTH - 1 ||
                                         du1 < 0 && du1 > G2LTexConfig::get().DEPTH_HEIGHT - 1)
                                 {
                                    continue;
                                 }
//                                 std::cout<<"---------------->q:"<<q(2)<<" p:"<<p(2)<<std::endl;

                                 if(newcheckDepth(cur_depth, dv0, du0,q(2), 0.05) == false ||
                                         newcheckDepth(adj_depth, dv1, du1,p(2), 0.05) == false )
                                 {
                                     continue;
                                 }

                                 float cur_base = curbaseimg.baseLayer.at<float>(u_0, v_0);
                                 float adj_base = adjbaseimg.baseLayer.at<float>(u_1, v_1);

                                 Lab_L2  llv;
                                 llv.A_v = cur_base;
                                 llv.B_v = adj_base;
                                 llv.A_a = adj_view_infos.L_alpha;
                                 llv.A_b = adj_view_infos.L_beta;
                                 colorharmLL.mypush_back(llv);

                             }
                         }
                     }
                }
                
            }
            //optimization
            ceres::Problem  Lproblem;
            double lweight = 1.0 / colorharmLL.nums;
//            double lweight = 1.0;

            double Lalpha = cur_view_infos.L_alpha;
            double Lbeta = cur_view_infos.L_beta;

            for(int m_i = 0; m_i<colorharmLL.nums; m_i++)
            {
                Lab_L2  lab_l = colorharmLL.at(m_i);
                ceres::LossFunction* lossfunction = new ceres::CauchyLoss(1.0);//new ceres::CauchyLoss(1.0);
                ceres::CostFunction *cost_function1_l = colorCostFunctor1::create(lweight, lab_l.A_v, lab_l.B_v, lab_l.A_a, lab_l.A_b);
                Lproblem.AddResidualBlock(cost_function1_l, lossfunction, &Lalpha, &Lbeta);//三通道
            }

            double l2 = 0.33;
            ceres::CostFunction *cost_function2 = colorCostFunctor2::Create(l2);
            Lproblem.AddResidualBlock(cost_function2, NULL, &Lalpha, &Lbeta);

            ceres::Solver::Options options;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.linear_solver_type = ceres::ITERATIVE_SCHUR;//ceres::DENSE_SCHUR; //ceres::LEVENBERG_MARQUARDT;
            //options.linear_solver_type = ceres::ITERATIVE_SCHUR;

            options.minimizer_progress_to_stdout = false;
            options.max_num_iterations = 100;//100;
            options.num_threads = 2;
            options.num_linear_solver_threads = 2;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &Lproblem, &summary);
            //             std::cout << summary.FullReport() << "\n";
            cur_view_infos.L_alpha = Lalpha;
            cur_view_infos.L_beta = Lbeta;
//            std::cout<<"after La:"<<Lalpha<<" Lb:"<<Lbeta<<std::endl;
//            std::cout<<"---i:"<<view_idx<<" num:"<<colorharmLL.nums<<std::endl;
        }
    }//end iter


    //correct color
    for(int i = 0;i<texture_views.size();i++)
    {
        TextureView  &texture_view =  texture_views.at(i);//当前视口信息
        BaseAndDetail   &curbaseimg = baseimgs.at(i);

        myViewImageInfo  &cur_view_infos = viewImageList.at(i);

//        cv::Mat  labimg = cv::Mat(texture_view.height, texture_view.width, CV_32FC3);
        std::cout<<"i:"<<i<<" La:"<<cur_view_infos.L_alpha<<" Lb:"<<cur_view_infos.L_beta<<std::endl;

        cv::Mat  orgGrayImg;
        cv::cvtColor(texture_view.targeImage, orgGrayImg, CV_RGB2GRAY);
        cv::Mat  colorimg = cv::Mat(texture_view.height, texture_view.width, CV_8UC3, cv::Scalar(0, 0 ,0));

        for(int h = 0; h < texture_view.height; h++)
        {
            for(int w = 0; w< texture_view.width; w++)
            {

                float value = curbaseimg.baseLayer.at<float>(h,w)*cur_view_infos.L_alpha + cur_view_infos.L_beta;
                value =  exp_function(value + curbaseimg.detailLayer.at<float>(h,w));
                float grayvalue = std::max(0.0f, std::min(255.0f, value));
                float rate = 1.0;
                if(orgGrayImg.at<uchar>(h, w) > 0)
                {
                    rate =float(grayvalue)/ orgGrayImg.at<uchar>(h, w);
                }
//                std::cout<<"gray:"<<grayvalue<<"---orggray:"<<float(orgGrayImg.at<uchar>(w, h))<<" rage:"<<<std::endl;
//                colorimg.at<cv::Vec3b>(h, w)[0] = std::min(255.0f, texture_view.targeImage.at<cv::Vec3b>(h, w)[0]*rate);
//                colorimg.at<cv::Vec3b>(h, w)[1] = std::min(255.0f, texture_view.targeImage.at<cv::Vec3b>(h, w)[1]*rate);
//                colorimg.at<cv::Vec3b>(h, w)[2] = std::min(255.0f, texture_view.targeImage.at<cv::Vec3b>(h, w)[2]*rate);

                texture_view.image->at(w, h, 2) = std::min(255.0f, texture_view.targeImage.at<cv::Vec3b>(h, w)[0]*rate);
                texture_view.image->at(w, h, 1) = std::min(255.0f, texture_view.targeImage.at<cv::Vec3b>(h, w)[1]*rate);
                texture_view.image->at(w, h, 0) = std::min(255.0f, texture_view.targeImage.at<cv::Vec3b>(h, w)[2]*rate);

            }
        }

        char buf[256];

//        sprintf(buf,"grag%02d.png",i);
//        cv::imwrite(buf, grayimg);

//        cv::Mat img;
//        cv::cvtColor(grayimg, img, CV_GRAY2BGR);
//        sprintf(buf,"ch%02d.png",i);
//        cv::imwrite(buf, colorimg);

//        for(int h = 0; h < G2LTexConfig::get().IMAGE_HEIGHT; h++)
//        {
//            for(int w = 0; w < G2LTexConfig::get().IMAGE_WIDTH; w++)
//            {
//                texture_view.image->at(w, h, 0) = resultimg.at<cv::Vec3b>(h, w)[2];
//                texture_view.image->at(w, h, 1) = resultimg.at<cv::Vec3b>(h, w)[1];
//                texture_view.image->at(w, h, 2) = resultimg.at<cv::Vec3b>(h, w)[0];
//            }
//        }
//        sprintf(buf,"chimg%02d.jpg", i);
//        mve::image::save_file(texture_view.image, buf);
    }
}


bool newcheckDepth(cv::Mat depth, float u, float v, float d, float threshold)
{
    bool flag = false;

    int x = int(u + 0.5);
    int y = int(v + 0.5);
    ushort p_d = depth.at<ushort>(y, x);
//    std::cout<<"----------cd:"<<p_d<<std::endl;

    if(p_d > 0)
    {
        float c_d = p_d/1000.0f;

        if(std::abs(c_d - d) < threshold)
        {
            flag = true;
        }
    }

    return flag;
}
TEX_NAMESPACE_END
