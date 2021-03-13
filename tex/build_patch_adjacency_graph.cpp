/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */

#include "texturing.h"
TEX_NAMESPACE_BEGIN
#include <vector>

int  getChartIndex(std::vector<int>  labelPatchCount, int label)
{
    int sum = 0;
    for(int i =0; i < label - 1; i++)
    {
        sum += labelPatchCount[i];
    }
    return sum;
}

//根据chart的索引得到他对应的标签
int getChartLabel(std::vector<int>  labelPatchCount, int index, int view_num)
{
    int sum = 0;
    int label = -1;
    for(int i = 0; i < view_num; i++)
    {
        int count = labelPatchCount[i];
        sum = sum + count;
        if(sum > index)
        {
            label = i;
            break;
        }
    }
    return label+1;
}

/**
 * @brief build_patch_adjacency_graph  建立所有chart之间的邻接关系
 * @param graph             网格图
 * @param subgraphs      每个标签对应的所有chart对应的面
 * @param labelPatchCount   每个标签对应chart的数量
 * @param patch_graph        建立chart之间的邻接图
 */
void build_patch_adjacency_graph(UniGraph const & graph, std::vector<std::vector<std::size_t> >  subgraphs, std::vector<int>  labelPatchCount,
                                 std::vector<std::vector<std::size_t> > &patch_graph, std::vector<myFaceInfo>   &faceInfoList)
{
    patch_graph.resize(subgraphs.size());
    for(int i = 0; i < subgraphs.size(); i++)
    {
        std::vector<std::size_t> sg = subgraphs[i];//label聚类中所有的面
        for(int j = 0; j < sg.size(); j++)//遍历所有的面
        {
            int face_idx = sg[j];//面索引
            int label = graph.get_label(face_idx);//标签
            if(label == 0)
            {
                continue;
            }

            std::vector<std::size_t> nodelist = graph.get_adj_nodes(face_idx);//每个面的邻接面
            for(int k= 0; k < nodelist.size(); k++ )//所有的邻接面
            {
                faceInfoList[face_idx].chart_id = i;//每个面属于那个chart
                int node_idx = nodelist[k];//邻接面的面索引
                int nodelabel = graph.get_label(node_idx);//邻接面的标签
                if(nodelabel == label || nodelabel == 0)//相同不考虑
                {
                    continue;
                }
                faceInfoList[face_idx].isChartBoundaryFace =  true;//标记边界上的面
                //存在邻接，并找出这个面在那个chart中间
                int idx = getChartIndex(labelPatchCount, nodelabel);//邻接块在队列中的起始位置
                for(int l_x = 0; l_x < labelPatchCount[nodelabel - 1]; l_x++)//该label包含的块数
                {
                    std::vector<std::size_t> ng = subgraphs[idx+l_x];//取出一个块
                   std::vector<std::size_t>::iterator iter = find(ng.begin(), ng.end(), node_idx);//是否可以在这个块中找到对应的面
                    if(iter != ng.end())//找到
                    {
                        std::vector<std::size_t> &patch = patch_graph[i];//这个邻接关系是否已经存在
                        std::vector<std::size_t>::iterator iter2 = find(patch.begin(), patch.end(), idx + l_x);
                        if(iter2 == patch.end())//没找到，那么不存在
                        {
                            patch.push_back(idx + l_x);//加入一条邻接边
                        }
                        break;
                    }
                }
            }
        }
    }
}

void build_twopass_adjacency_graph(UniGraph const & graph, std::vector<std::vector<std::size_t> > subgraphs, std::vector<int>  labelPatchCount,
                                 std::vector<std::vector<std::size_t> >   &patch_graph)
{
    patch_graph.resize(subgraphs.size());
    for(int i = 0; i < subgraphs.size(); i++)
    {
        std::vector<std::size_t> sg = subgraphs[i];//label聚类中所有的面
        for(int j = 0; j < sg.size(); j++)//遍历所有的面
        {
            int face_idx = sg[j];//面索引
            int label = graph.get_twoPassLabel(face_idx);//得到双通的标签
            if(label == 0)
            {
                continue;
            }
            std::vector<std::size_t> nodelist = graph.get_adj_nodes(face_idx);//每个面的邻接面
            for(int k= 0; k < nodelist.size(); k++ )//所有的邻接面
            {
                int node_face_idx = nodelist[k];//邻接面的面索引
                int nodelabel = graph.get_twoPassLabel(node_face_idx);//邻接面的标签
                if(nodelabel == label || nodelabel == 0)//相同不考虑
                {
                    continue;
                }

                int idx = getChartIndex(labelPatchCount, nodelabel);//邻接块在队列中的起始位置
                for(int l_x = 0; l_x < labelPatchCount[nodelabel - 1]; l_x++)//该label包含的块数
                {
                    std::vector<std::size_t> ng = subgraphs[idx+l_x];//取出一个块
                    std::vector<std::size_t>::iterator iter = find(ng.begin(), ng.end(), node_face_idx);//是否可以在这个块中找到对应的面
                    if(iter != ng.end())//找到
                    {
                        std::vector<std::size_t> &patch = patch_graph[i];//这个邻接关系是否已经存在
                        std::vector<std::size_t>::iterator iter2 = find(patch.begin(), patch.end(), idx + l_x);
                        if(iter2 == patch.end())//没找到，那么不存在
                        {
                            patch.push_back(idx + l_x);//加入一条邻接边
                        }
                        break;//整个块只建立一次
                    }

                }

            }

        }

    }

}

//合并较小的path块，避免优化失败。
//合并的时候考虑时候越界。
void combineSmallPath(UniGraph & graph, mve::TriangleMesh::ConstPtr mesh,
                      std::vector<TextureView> texture_views, bool zeroflag)
{
    std::vector<unsigned int> const & faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const & vertices = mesh->get_vertices();//所有顶点

    std::vector<std::vector<std::size_t> > subgraphs;
    std::vector<int>  labelPatchCount;
    int startsize;
    int endsize;
    for(int i = 0; i < texture_views.size(); i++)
    {
        int const label = i + 1;
        startsize = subgraphs.size();
        graph.get_subgraphs(label, &subgraphs);
        endsize = subgraphs.size();
        labelPatchCount.push_back(endsize - startsize);
    }

    for(int i = 0; i < subgraphs.size();i++)
    {
        std::vector<std::size_t> sg = subgraphs[i];//label聚类中所有的面
        if(sg.size() > G2LTexConfig::get().MIN_CHART_NUM)
        {
            continue;
        }

        //for test
        std::vector<int> adjlist;

        int curlabel = graph.get_label(sg[0]);
        std::vector<int>  bestlabel;
        std::vector<int>  bestindex;
        for(int j = 0; j < sg.size(); j++)
        {
            int face_idx = sg[j];
            int label = graph.get_label(face_idx);
            std::vector<std::size_t> nodelist = graph.get_adj_nodes(face_idx);//每个面的邻接面

            for(int k= 0; k<nodelist.size(); k++ )//所有的邻接面
            {
                int node_idx = nodelist[k];//邻接面的面索引
                int nodelabel = graph.get_label(node_idx);//邻接面的标签
                if(nodelabel == label || nodelabel == 0)//相同不考虑,来自同一块或者无标签
                {
                    continue;
                }

                int idx = getChartIndex(labelPatchCount, nodelabel);//邻接块在队列中的起始位置
                for(int l_x = 0; l_x < labelPatchCount[nodelabel - 1]; l_x++)//该label包含的块数
                {
                    std::vector<std::size_t> ng = subgraphs[idx + l_x];//取出一个块
                    if(ng.size() < G2LTexConfig::get().MIN_CHART_NUM)
                    {
                        continue;
                    }

                   std::vector<std::size_t>::iterator iter = find(ng.begin(), ng.end(), node_idx);//是否可以在这个块中找到对应的面
                    if(iter != ng.end())//找到
                    {
                        bestlabel.push_back(nodelabel);
                        bestindex.push_back(idx+l_x);
                        adjlist.push_back(idx + l_x);
                    }
                }
            }
        }

        int changelabel = 0;
        int changeindex;
        if(bestlabel.size() != 0)
        {

            TextureView cur_view = texture_views[curlabel - 1];
            math::Vec3f cur_view_dir = cur_view.get_viewing_direction();

            float angle = 0;
            for(int k = 0;k<bestlabel.size();k++)
            {
                //修改面的小块标签
                bool  change = true;
                TextureView view = texture_views[bestlabel[k] - 1];
                math::Vec3f view_dir = view.get_viewing_direction();
                for(int j = 0; j < sg.size(); j++)
                {
                    int face_idx = sg[j];
                    math::Vec3f   v_1 = vertices[faces[face_idx*3]];
                    math::Vec3f   v_2 = vertices[faces[face_idx*3 + 1]];
                    math::Vec3f   v_3 = vertices[faces[face_idx*3 + 2]];
                    math::Vec2f p1 = view.get_pixel_coords(v_1);
                    math::Vec2f p2 = view.get_pixel_coords(v_2);
                    math::Vec2f p3 = view.get_pixel_coords(v_3);
                    if(p1(0) < G2LTexConfig::get().BOARD_IGNORE || p1(0) > (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                            p1(1) < G2LTexConfig::get().BOARD_IGNORE || p1(1) > (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) ||
                            p2(0) < G2LTexConfig::get().BOARD_IGNORE || p2(0) > (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                            p2(1) < G2LTexConfig::get().BOARD_IGNORE || p2(1) > (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) ||
                            p3(0) < G2LTexConfig::get().BOARD_IGNORE || p3(0) > (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                            p3(1) < G2LTexConfig::get().BOARD_IGNORE || p3(1) > (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE))
                    {
                        change = false;
                    }
                }

                if(change == true)
                {
                    float value = cur_view_dir(0)*view_dir(0) + cur_view_dir(1)*view_dir(1) + cur_view_dir(2)*view_dir(2);
                    if(value > angle && subgraphs[bestindex[k]].size() >G2LTexConfig::get().MIN_CHART_NUM)
                    {
                        angle = value;
                        changeindex = bestindex[k];
                        changelabel = bestlabel[k];
                    }
                }
            }
        }

        if(changelabel != 0)
        {

            for(int j = 0; j < sg.size(); j++)
            {
                int face_idx = sg[j];
                graph.set_label(face_idx, changelabel);
            }
        }
        else if(adjlist.size() == 0)
        {
            if(zeroflag == true)
            {
                for(int j = 0; j < sg.size(); j++)
                {
                    int face_idx = sg[j];
                    graph.set_label(face_idx, 0);
                }
            }
        }
        else if (sg.size() < G2LTexConfig::get().MIN_CHART_NUM && zeroflag == true)
        {
////            std::cout<<"1--------sg.size():"<<sg.size()<<std::endl;
//            std::sort(adjlist.begin(), adjlist.end());
//            adjlist.erase(std::unique(adjlist.begin(), adjlist.end()), adjlist.end());
//            int count_f = 0;
//            int pos = -1;
//            for(int iter = 0;iter<adjlist.size();iter++)
//            {
//                int f_c = subgraphs[adjlist[iter]].size();
//                if(count_f < f_c)
//                {
//                    count_f = f_c;
//                    pos = adjlist[iter];
//                }
//            }
//            std::cout<<"pos:"<<pos<<std::endl;

//            if(pos != -1)
//            {
//                std::vector<std::size_t>  ag = subgraphs[pos];
//                std::cout<<"conbinie:"<<ag.size()<<std::endl;
//                int f_label = graph.get_label(ag[0]);

//                for(int j = 0; j < sg.size(); j++)
//                {
//                    int face_idx = sg[j];
//                    graph.set_label(face_idx, f_label);
//                }
//            }
        }
    }
}

void combineTwoPassSmallPatch(UniGraph & graph, mve::TriangleMesh::ConstPtr mesh, std::vector<TextureView> texture_views, bool zeroflag)
{
    std::vector<unsigned int> const &faces = mesh->get_faces();//所有的面
    std::vector<math::Vec3f> const &vertices = mesh->get_vertices();//所有顶点

    std::vector<std::vector<std::size_t> > subgraphs;
    std::vector<int>  labelPatchCount;
    int startsize;
    int endsize;
    for(int i = 0; i < texture_views.size(); i++)
    {
        int const label = i + 1;
        startsize = subgraphs.size();
//        graph.get_subgraphs(label, &subgraphs);
        graph.get_twoPassSubgraphs(label, &subgraphs);
        endsize = subgraphs.size();
        labelPatchCount.push_back(endsize - startsize);
    }

    for(int i = 0; i < subgraphs.size();i++)
    {
        std::vector<std::size_t> sg = subgraphs[i];//label聚类中所有的面
        if(sg.size() > G2LTexConfig::get().MIN_CHART_NUM)//label聚类大于阈值中所有的面
        {
            continue;
        }

        //for test
        std::vector<int> adjlist;

        int curlabel = graph.get_twoPassLabel(sg[0]);
        std::vector<int>  bestlabel;
        std::vector<int>  bestindex;
        for(int j = 0; j < sg.size(); j++)
        {
            int face_idx = sg[j];
            int label = graph.get_twoPassLabel(face_idx);
            std::vector<std::size_t> nodelist = graph.get_adj_nodes(face_idx);//每个面的邻接面

            for(int k= 0; k<nodelist.size(); k++ )//所有的邻接面
            {
                int node_idx = nodelist[k];//邻接面的面索引
                int nodelabel = graph.get_twoPassLabel(node_idx);//邻接面的标签
                if(nodelabel == label || nodelabel == 0)//相同不考虑,来自同一块或者无标签
                {
                    continue;
                }

                int idx = getChartIndex(labelPatchCount, nodelabel);//邻接块在队列中的起始位置
                for(int l_x = 0; l_x < labelPatchCount[nodelabel - 1]; l_x++)//该label包含的块数
                {
                    std::vector<std::size_t> ng = subgraphs[idx + l_x];//取出一个块
                    if(ng.size() < G2LTexConfig::get().MIN_CHART_NUM)
                    {
                        continue;
                    }

                   std::vector<std::size_t>::iterator iter = find(ng.begin(), ng.end(), node_idx);//是否可以在这个块中找到对应的面
                    if(iter != ng.end())//找到
                    {
                        bestlabel.push_back(nodelabel);
                        bestindex.push_back(idx+l_x);
                        adjlist.push_back(idx + l_x);
                    }
                }
            }
        }

        int changelabel = 0;
        int changeindex;
        if(bestlabel.size() != 0)
        {

            TextureView cur_view = texture_views[curlabel - 1];
            math::Vec3f cur_view_dir = cur_view.get_viewing_direction();

            float angle = 0;
            for(int k = 0;k<bestlabel.size();k++)
            {
                //修改面的小块标签
                bool  change = true;
                TextureView view = texture_views[bestlabel[k] - 1];
                math::Vec3f view_dir = view.get_viewing_direction();
                for(int j = 0; j < sg.size(); j++)
                {
                    int face_idx = sg[j];
                    math::Vec3f   v_1 = vertices[faces[face_idx*3]];
                    math::Vec3f   v_2 = vertices[faces[face_idx*3 + 1]];
                    math::Vec3f   v_3 = vertices[faces[face_idx*3 + 2]];
                    math::Vec2f p1 = view.get_pixel_coords(v_1);
                    math::Vec2f p2 = view.get_pixel_coords(v_2);
                    math::Vec2f p3 = view.get_pixel_coords(v_3);
                    if(p1(0) < G2LTexConfig::get().BOARD_IGNORE || p1(0) > (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                            p1(1) < G2LTexConfig::get().BOARD_IGNORE || p1(1) > (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) ||
                            p2(0) < G2LTexConfig::get().BOARD_IGNORE || p2(0) > (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                            p2(1) < G2LTexConfig::get().BOARD_IGNORE || p2(1) > (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE) ||
                            p3(0) < G2LTexConfig::get().BOARD_IGNORE || p3(0) > (G2LTexConfig::get().IMAGE_WIDTH - G2LTexConfig::get().BOARD_IGNORE) ||
                            p3(1) < G2LTexConfig::get().BOARD_IGNORE || p3(1) > (G2LTexConfig::get().IMAGE_HEIGHT - G2LTexConfig::get().BOARD_IGNORE))
                    {
                        change = false;
                    }
                }

                if(change == true)
                {
                    float value = cur_view_dir(0)*view_dir(0) + cur_view_dir(1)*view_dir(1) + cur_view_dir(2)*view_dir(2);
                    if(value > angle && subgraphs[bestindex[k]].size() >G2LTexConfig::get().MIN_CHART_NUM)
                    {
                        angle = value;
                        changeindex = bestindex[k];
                        changelabel = bestlabel[k];
                    }
                }
            }
        }

        if(changelabel != 0)
        {

            for(int j = 0; j < sg.size(); j++)
            {
                int face_idx = sg[j];
                graph.set_TwoPasslabel(face_idx, changelabel);
            }
        }
        else if(adjlist.size() == 0)
        {
            if(zeroflag == true)
            {
                for(int j = 0; j < sg.size(); j++)
                {
                    int face_idx = sg[j];
                    graph.set_TwoPasslabel(face_idx, 0);
                }
            }
        }
        else if (sg.size() < 10)
        {
//            std::cout<<"1--------sg.size():"<<sg.size()<<std::endl;
//            std::sort(adjlist.begin(), adjlist.end());
//            adjlist.erase(std::unique(adjlist.begin(), adjlist.end()), adjlist.end());
//            int count_f = 0;
//            int pos = -1;
//            for(int iter = 0;iter<adjlist.size();iter++)
//            {
//                int f_c = subgraphs[adjlist[iter]].size();
//                if(count_f < f_c)
//                {
//                    count_f = f_c;
//                    pos = adjlist[iter];
//                }
//            }
//            std::cout<<"pos:"<<pos<<std::endl;

//            if(pos != -1)
//            {
//                std::vector<std::size_t>  ag = subgraphs[pos];
//                std::cout<<"conbinie:"<<ag.size()<<std::endl;
//                int f_label = graph.get_label(ag[0]);

//                for(int j = 0; j < sg.size(); j++)
//                {
//                    int face_idx = sg[j];
//                    graph.set_label(face_idx, f_label);
//                }
//            }
        }
    }

}


TEX_NAMESPACE_END
