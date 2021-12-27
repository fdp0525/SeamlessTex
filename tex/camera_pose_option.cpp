/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */

#include "texturing.h"
#include "util.h"

TEX_NAMESPACE_BEGIN

void camera_pose_option(std::vector<TextureView> texture_views, int current_label, int adj_chart_label,
                            std::vector<unsigned int> const & faces, std::vector<math::Vec3f> const & vertices,
                            std::vector<std::size_t> cur_chart, std::vector<std::size_t> adj_chart,
                            std::vector<myViewImageInfo>&  viewImageList,
                        Eigen::MatrixXd &JTc, Eigen::VectorXd &JRc,
                        Eigen::MatrixXd &JTd, Eigen::VectorXd &JRd)
{

    int nvariable = 6;
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


    TextureView  adj_texture_view =  texture_views.at(adj_chart_label - 1);//邻接块对应的视口
    //取出邻接视口对应的图片信息
    cv::Mat  adj_chart_img = viewImageList[adj_chart_label - 1].img;

    math::Matrix4f  amat =   adj_texture_view.getWorldToCamMatrix();
    Eigen::Matrix4f adj_chart_world_to_cam = mathToEigen(amat);
    std::map<int, int>  computeFlag;
    for(int face_idx = 0; face_idx < cur_chart.size(); face_idx++)//遍历当前chart上所有的顶点
    {
        int f_i = cur_chart[face_idx];//当前面面索引
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
    for(int face_idx = 0; face_idx < adj_chart.size(); face_idx++)//遍历所有的面
    {
        int f_i = adj_chart[face_idx];//邻接面中的面索引
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

    JTc = c_JTJ;
    JRc = c_JTr;
    JTd = d_JTJ;
    JRd = d_JTr;

}


bool camera_pose_option_with_detail(std::vector<TextureView> texture_views, int current_label, int adj_chart_label,
                                    std::vector<unsigned int> const & faces, std::vector<math::Vec3f> const & vertices,
                                    std::vector<std::size_t> cur_chart, std::vector<std::size_t> adj_chart,
                                    std::vector<myViewImageInfo>&  viewImageList,
                                    Eigen::MatrixXd &JTc, Eigen::VectorXd &JRc,
                                    Eigen::MatrixXd &JTd, Eigen::VectorXd &JRd,
                                    Eigen::MatrixXd &JTt, Eigen::VectorXd &JRt)
{

    bool stopflag = false;
    int nvariable = 6;
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

    //纹理项
    Eigen::MatrixXd    t_JTJ(nvariable, nvariable);
    Eigen::MatrixXd    t_JTr(nvariable, 1);
    Eigen::MatrixXd    t_J(nvariable, 1);
    t_JTJ.setZero();
    t_JTr.setZero();
    t_J.setZero();

    //当前标签对应的信息
    TextureView  &texture_view =  texture_views.at(current_label - 1);
    math::Matrix4f  cmat = texture_view.getWorldToCamMatrix();
    Eigen::Matrix4f  currentMat = mathToEigen(cmat);
//            Eigen::Matrix4f  currentMat = faceInfoList[c_f_x].world_to_cam;//当前块的相机变换

    cv::Mat  currentImg = viewImageList[current_label - 1].img;
    cv::Mat  currentDepth = viewImageList[current_label - 1].depth;
    cv::Mat currentGradXImg = viewImageList[current_label - 1].gradxImg;//得到当前面对应视口的梯度图应用颜色一致性追踪
    cv::Mat currentGradYImg = viewImageList[current_label - 1].gradyImg;//得到当前面对应视口的梯度图应用颜色一致性追踪

    cv::Mat cur_detailimg = texture_view.detailMap;
    cv::Mat curDetailGradXImg = viewImageList[current_label - 1].detailgradxImg;//得到当前面对应视口的梯度图应用颜色一致性追踪
    cv::Mat curDetailGradYImg = viewImageList[current_label - 1].detailgradyImg;//得到当前面对应视口的梯度图应用颜色一致性追踪

    TextureView  adj_texture_view =  texture_views.at(adj_chart_label - 1);//邻接块对应的视口
    //取出邻接视口对应的图片信息
    cv::Mat  adj_chart_img = viewImageList[adj_chart_label - 1].img;
    cv::Mat  adj_detail_img = adj_texture_view.detailMap;

    math::Matrix4f  amat =   adj_texture_view.getWorldToCamMatrix();
    Eigen::Matrix4f adj_chart_world_to_cam = mathToEigen(amat);
    std::map<int, int>  computeFlag;


    for(int face_idx = 0; face_idx < cur_chart.size(); face_idx++)//遍历当前chart上所有的顶点
    {
        int f_i = cur_chart[face_idx];//当前面面索引
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

//            if(v_0 < G2LTexConfig::get().BOARD_IGNORE || v_0 > (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
//               u_0 < G2LTexConfig::get().BOARD_IGNORE || u_0 > (G2LTexConfig::get().IMAGE_HEIGHT- G2LTexConfig::get().BOARD_IGNORE))
//            {
//                stopflag = true;
//                break;
//            }

            Eigen::Vector4f    p = adj_chart_world_to_cam * q0;//投影到邻接平面上
            float v_1 = p(0) * G2LTexConfig::get().IMAGE_FX / p(2) + G2LTexConfig::get().IMAGE_CX;
            float u_1 = p(1) * G2LTexConfig::get().IMAGE_FY / p(2) + G2LTexConfig::get().IMAGE_CY;
//#pragma omp critical
            if( v_0 >= G2LTexConfig::get().BOARD_IGNORE && v_0 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                    u_0 >= G2LTexConfig::get().BOARD_IGNORE && u_0 <= (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) &&
                    v_1 >= G2LTexConfig::get().BOARD_IGNORE && v_1 <= (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) &&
                    u_1 >= G2LTexConfig::get().BOARD_IGNORE && u_1 <= (G2LTexConfig::get().IMAGE_HEIGHT-G2LTexConfig::get().BOARD_IGNORE) )//ablity
            {

                //color
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
                    c_J(0) = -q(2) * k1 + q(1) * k2;
                    c_J(1) =  q(2) * k0 - q(0) * k2;
                    c_J(2) = -q(1) * k0 + q(0) * k1;
                    c_J(3) = k0;
                    c_J(4) = k1;
                    c_J(5) = k2;
                    c_JTJ += c_J * c_J.transpose();
                    c_JTr += c_J * c_r;
                }

                //texture detail
                {

                    //处理细节图
                    float  curdetail = getInterColorFromGrayImg(cur_detailimg, v_0, u_0);
                    float  adjdetail = getInterColorFromGrayImg(adj_detail_img, v_1, u_1);
                    float  t_r = (curdetail - adjdetail)*255.0f;
//                    std::cout<<"-c:"<<curdetail<<" a:"<<adjdetail<<" v1:"<<v_1<<" U1:"<<u_1<<std::endl;
//                    std::cout<<t_r<<std::endl;

                    float invz = 1.0f / q(2);
                    float  gx = getInterColorFromGrayImgV2(curDetailGradXImg, v_0, u_0);
                    float  gy = getInterColorFromGrayImgV2(curDetailGradYImg, v_0, u_0);

                    float  k0 = gx * G2LTexConfig::get().IMAGE_FX * invz;
                    float  k1 = gy * G2LTexConfig::get().IMAGE_FY * invz;
                    float  k2 = -(k0 * q(0) + k1 * q(1))*invz;

                    t_J.setZero();
                    t_J(0) = -q(2) * k1 + q(1) * k2;
                    t_J(1) =  q(2) * k0 - q(0) * k2;
                    t_J(2) = -q(1) * k0 + q(0) * k1;
                    t_J(3) = k0;
                    t_J(4) = k1;
                    t_J(5) = k2;
                    t_JTJ += t_J * t_J.transpose();
                    t_JTr += t_J * t_r;
//                    std::cout<<t_r<<std::endl;
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
//            #pragma omp critical
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

//    if(stopflag == true)
//    {
//        return stopflag;
//    }

    //邻接面,（根据论文这个应该不需要考虑：）
    computeFlag.clear();
    for(int face_idx = 0; face_idx < adj_chart.size(); face_idx++)//遍历所有的面
    {
        int f_i = adj_chart[face_idx];//邻接面中的面索引
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

                //texture detail
                {

                    //处理细节图
                    float  curdetail = getInterColorFromGrayImg(cur_detailimg, v_0, u_0);
                    float  adjdetail = getInterColorFromGrayImg(adj_detail_img, v_1, u_1);
                    float  t_r = (curdetail - adjdetail)*255.0f;
//                    std::cout<<"-c:"<<curdetail<<" a:"<<adjdetail<<" v1:"<<v_1<<" U1:"<<u_1<<std::endl;
//                    std::cout<<t_r<<std::endl;

                    float invz = 1.0f / q(2);
                    float  gx = getInterColorFromGrayImgV2(curDetailGradXImg, v_0, u_0);
                    float  gy = getInterColorFromGrayImgV2(curDetailGradYImg, v_0, u_0);

                    float  k0 = gx * G2LTexConfig::get().IMAGE_FX * invz;
                    float  k1 = gy * G2LTexConfig::get().IMAGE_FY * invz;
                    float  k2 = -(k0 * q(0) + k1 * q(1))*invz;

                    t_J.setZero();
                    t_J(0) = -q(2) * k1 + q(1) * k2;
                    t_J(1) =  q(2) * k0 - q(0) * k2;
                    t_J(2) = -q(1) * k0 + q(0) * k1;
                    t_J(3) = k0;
                    t_J(4) = k1;
                    t_J(5) = k2;
                    t_JTJ += t_J * t_J.transpose();
                    t_JTr += t_J * t_r;
//                    std::cout<<t_r<<std::endl;
                }
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
            }//depth
        }

    }

    JTc = c_JTJ;
    JRc = c_JTr;
    JTd = d_JTJ;
    JRd = d_JTr;

    JTt = t_JTJ;
    JRt = t_JTr;
//    std::cout<<"-------------1------------"<<std::endl;
//    std::cout<<JTt<<std::endl;
//    std::cout<<JRt<<std::endl;


    return stopflag;
}

TEX_NAMESPACE_END
