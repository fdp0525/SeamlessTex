/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <util/timer.h>

#include "util.h"
#include "texturing.h"
#include "mapmap/full.h"
#include "mrf/graph.h"

TEX_NAMESPACE_BEGIN

struct FaceInfo
{
    std::size_t component;//属于那个标签集合
    std::size_t id;//面的索引
};

/** Potts model */
float potts(int v1, int v2, int l1, int l2, float v1_r, float v1_g, float v1_b, float v2_r, float v2_g, float v2_b)
{
    //    return (l1 == l2 && l1 != 0 && l2 != 0) ? 0.0f : 1.0f;
    if(v1_r > 100 || v1_g > 100 || v1_b > 100 ||
            v2_r > 100 || v2_g > 100 || v2_b > 100)
    {
        return 100.0f;
    }
    float r = v1_r - v2_r;
    float g = v1_g- v2_g;
    float b = v1_b - v2_b;

//    float value = std::sqrt(r*r+g*g+b*b);
    float value = r*r+g*g+b*b;

    return value;
}

/** Potts model */
float orgpotts(int, int, int l1, int l2, float v1, float v2, float v3, float v4,float v5, float v6)
{
    return (l1 == l2 && l1 != 0 && l2 != 0) ? 0.0f : 1.0f;
}

/** Setup the neighborhood of the MRF. */
void set_neighbors(UniGraph const & graph, std::vector<FaceInfo> const & face_infos, std::vector<mrf::Graph::Ptr> const & mrfs)
{
    for (std::size_t i = 0; i < graph.num_nodes(); ++i)//所有面
    {
        std::vector<std::size_t> adj_faces = graph.get_adj_nodes(i);//每个面的邻接面
        for (std::size_t j = 0; j < adj_faces.size(); ++j)
        {
            std::size_t adj_face = adj_faces[j];
            /* The solver expects only one call of setNeighbours for two neighbours a and b. */
            if (i < adj_face)//为了保证值调用一次
            {
                assert(face_infos[i].component == face_infos[adj_face].component);//相邻面在同一个标签聚类
                const std::size_t component = face_infos[i].component;
                const std::size_t cid1 = face_infos[i].id;
                const std::size_t cid2 = face_infos[adj_face].id;
                mrfs[component]->set_neighbors(cid1, cid2);//同一个标签聚类相邻面之间建立上相互连接关系
            }
        }
    }
}

/** Set the data costs of the MRF. */
void set_data_costs(std::vector<FaceInfo> const & face_infos, DataCosts const & data_costs,
    std::vector<mrf::Graph::Ptr> const & mrfs)
{

    /* Set data costs for all labels except label 0 (undefined) */
    for (std::size_t i = 0; i < data_costs.rows(); i++)//所有的视口（也就是标签）
    {
        DataCosts::Row const & data_costs_for_label = data_costs.row(i);//所有的面在这个视口上投影信息

        std::vector<std::vector<mrf::SparseDataCost> > costs(mrfs.size());
        for(std::size_t j = 0; j < data_costs_for_label.size(); j++)
        {
            const std::size_t id = data_costs_for_label[j].first;//面索引
            const float data_cost = data_costs_for_label[j].second;//投影质量
            const std::size_t component = face_infos[id].component;//所属的标签
            const std::size_t cid = face_infos[id].id;//面
            //TODO change index type of mrf::Graph
            //            costs[component].push_back({static_cast<int>(cid), data_cost});
            costs[component].push_back({static_cast<int>(cid), data_cost, 1.0});

        }

        int label = i + 1;

        for (std::size_t j = 0; j < mrfs.size(); ++j)
        {
            mrfs[j]->set_data_costs(label, costs[j]);
        }
    }

    for (std::size_t i = 0; i < mrfs.size(); ++i)
    {
        /* Set costs for undefined label */
        std::vector<mrf::SparseDataCost> costs(mrfs[i]->num_sites());
        for (std::size_t j = 0; j < costs.size(); j++)
        {
            costs[j].site = j;
            costs[j].cost = 1.0f;
//            costs[j].smoothR = 1000.0f;
//            costs[j].smoothB = 1000.0f;
//            costs[j].smoothG = 1000.0f;
            costs[j].detailvalue = 1.0f;

        }
        mrfs[i]->set_data_costs(0, costs);
    }
}

void set_data_feature_costs(std::vector<FaceInfo> const & face_infos, DataCosts const & data_costs,
                           DataCosts const & feature_costs, std::vector<mrf::Graph::Ptr> const &  mrfs)
{

    /* Set data costs for all labels except label 0 (undefined) */
    for (std::size_t i = 0; i < data_costs.rows(); i++)//所有的视口（也就是标签）
    {
        DataCosts::Row const & data_costs_for_label = data_costs.row(i);//所有的面在这个视口上投影信息
        DataCosts::Row const & feature_costs_for_label = feature_costs.row(i);//所有的面在这个视口上投影信息

        if(data_costs_for_label.size() != feature_costs_for_label.size())
        {
            std::cout<<"----------------------error"<<std::endl;
        }
        std::vector<std::vector<mrf::SparseDataCost> > costs(mrfs.size());
        for(std::size_t j = 0; j < data_costs_for_label.size(); j++)
        {
            const std::size_t id = data_costs_for_label[j].first;//面索引
            const float data_cost = data_costs_for_label[j].second;//投影质量
            const std::size_t component = face_infos[id].component;//所属的标签
            const std::size_t cid = face_infos[id].id;//面
            const float feature_cost = feature_costs_for_label[j].second;
//            std::cout<<"---------smooth:"<<smooth_cost<<std::endl;
            //TODO change index type of mrf::Graph
            costs[component].push_back({static_cast<int>(cid), data_cost, feature_cost});
        }

        int label = i + 1;

        for (std::size_t j = 0; j < mrfs.size(); ++j)
        {
            mrfs[j]->set_data_costs(label, costs[j]);
        }
    }

    for (std::size_t i = 0; i < mrfs.size(); ++i)
    {
        /* Set costs for undefined label */
        std::vector<mrf::SparseDataCost> costs(mrfs[i]->num_sites());
        for (std::size_t j = 0; j < costs.size(); j++)
        {
            costs[j].site = j;
            costs[j].cost = 1.0f;
            costs[j].detailvalue = 1.0f;

        }
        mrfs[i]->set_data_costs(0, costs);
    }
}

void set_data_smooth_detail_costs(std::vector<FaceInfo> const & face_infos, DataCosts const & data_costs,
                           SmoothCosts  const & smooth_costs, DataCosts const & detail_costs, std::vector<mrf::Graph::Ptr> const &  mrfs)
{

    /* Set data costs for all labels except label 0 (undefined) */
    for (std::size_t i = 0; i < data_costs.rows(); i++)//所有的视口（也就是标签）
    {
        DataCosts::Row const & data_costs_for_label = data_costs.row(i);//所有的面在这个视口上投影信息
        SmoothCosts::Row const & Smooth_costs_for_label = smooth_costs.row(i);//所有的面在这个视口上投影信息
        DataCosts::Row const & detail_costs_for_label = detail_costs.row(i);//所有的面在这个视口上投影信息

        if(data_costs_for_label.size() != detail_costs_for_label.size())
        {
            std::cout<<"----------------------error"<<std::endl;
        }
        std::vector<std::vector<mrf::SparseDataCost> > costs(mrfs.size());
        for(std::size_t j = 0; j < data_costs_for_label.size(); j++)
        {
            const std::size_t id = data_costs_for_label[j].first;//面索引
            const float data_cost = data_costs_for_label[j].second;//投影质量
            const std::size_t component = face_infos[id].component;//所属的标签
            const std::size_t cid = face_infos[id].id;//面
            math::Vec3f smooth_cost = Smooth_costs_for_label[j].second;
            const float detail_cost = detail_costs_for_label[j].second;

//            std::cout<<"---------smooth:"<<smooth_cost<<std::endl;
            //TODO change index type of mrf::Graph
            costs[component].push_back({static_cast<int>(cid), data_cost, smooth_cost[0], smooth_cost[1], smooth_cost[2], detail_cost});
        }

        int label = i + 1;

        for (std::size_t j = 0; j < mrfs.size(); ++j)
        {
            mrfs[j]->set_data_costs(label, costs[j]);
        }
    }

    for (std::size_t i = 0; i < mrfs.size(); ++i)
    {
        /* Set costs for undefined label */
        std::vector<mrf::SparseDataCost> costs(mrfs[i]->num_sites());
        for (std::size_t j = 0; j < costs.size(); j++)
        {
            costs[j].site = j;
            costs[j].cost = 1.0f;
            costs[j].smoothR = 1000.0f;
            costs[j].smoothG =1000.0f;
            costs[j].smoothB = 1000.0f;
            costs[j].detailvalue = 1.0f;

        }
        mrfs[i]->set_data_costs(0, costs);
    }
}

void set_data_smooth_costs(std::vector<FaceInfo> const & face_infos, DataCosts const & data_costs,
                           SmoothCosts  const & smooth_costs, std::vector<mrf::Graph::Ptr> const &  mrfs)
{

    /* Set data costs for all labels except label 0 (undefined) */
    for (std::size_t i = 0; i < data_costs.rows(); i++)//所有的视口（也就是标签）
    {
        DataCosts::Row const & data_costs_for_label = data_costs.row(i);//所有的面在这个视口上投影信息
        SmoothCosts::Row const & Smooth_costs_for_label = smooth_costs.row(i);//所有的面在这个视口上投影信息

        if(data_costs_for_label.size() != Smooth_costs_for_label.size())
        {
            std::cout<<"----------------------error"<<std::endl;
        }
        std::vector<std::vector<mrf::SparseDataCost> > costs(mrfs.size());
        for(std::size_t j = 0; j < data_costs_for_label.size(); j++)
        {
            const std::size_t id = data_costs_for_label[j].first;//面索引
            const float data_cost = data_costs_for_label[j].second;//投影质量
            const std::size_t component = face_infos[id].component;//所属的标签
            const std::size_t cid = face_infos[id].id;//面
            math::Vec3f smooth_cost = Smooth_costs_for_label[j].second;
//            std::cout<<"---------smooth:"<<smooth_cost<<std::endl;
            //TODO change index type of mrf::Graph
//            costs[component].push_back({static_cast<int>(cid), data_cost, smooth_cost[0], smooth_cost[1], smooth_cost[2]});
        }

        int label = i + 1;

        for (std::size_t j = 0; j < mrfs.size(); ++j)
        {
            mrfs[j]->set_data_costs(label, costs[j]);
        }
    }

    for (std::size_t i = 0; i < mrfs.size(); ++i)
    {
        /* Set costs for undefined label */
        std::vector<mrf::SparseDataCost> costs(mrfs[i]->num_sites());
        for (std::size_t j = 0; j < costs.size(); j++)
        {
            costs[j].site = j;
            costs[j].cost = 1.0f;
//            costs[j].smoothR = 1000.0f;
//            costs[j].smoothG =1000.0f;
//            costs[j].smoothB = 1000.0f;
        }
        mrfs[i]->set_data_costs(0, costs);
    }
}

//void set_data_smooth_feature_costs(std::vector<FaceInfo> const & face_infos, DataCosts const & data_costs,
//                           SmoothCosts  const & smooth_costs, DataCosts const & feature_costs,std::vector<mrf::Graph::Ptr> const &  mrfs)
//{
//    /* Set data costs for all labels except label 0 (undefined) */
//    for (std::size_t i = 0; i < data_costs.rows(); i++)//所有的视口（也就是标签）
//    {
//        DataCosts::Row const & data_costs_for_label = data_costs.row(i);//所有的面在这个视口上投影信息
//        SmoothCosts::Row const & Smooth_costs_for_label = smooth_costs.row(i);//所有的面在这个视口上投影信息
//        DataCosts::Row const & feature_costs_for_label = feature_costs.row(i);//所有的面在这个视口上投影信息

//        if(data_costs_for_label.size() != Smooth_costs_for_label.size())
//        {
//            std::cout<<"----------------------error"<<std::endl;
//        }
//        std::vector<std::vector<mrf::SparseDataCost> > costs(mrfs.size());
//        for(std::size_t j = 0; j < data_costs_for_label.size(); j++)
//        {
//            const std::size_t id = data_costs_for_label[j].first;//面索引
//            const float data_cost = data_costs_for_label[j].second;//投影质量
//            const std::size_t component = face_infos[id].component;//所属的标签
//            const std::size_t cid = face_infos[id].id;//面
//            math::Vec3f smooth_cost = Smooth_costs_for_label[j].second;
//            const float feature_cost = feature_costs_for_label[j].second;//投影质量

////            std::cout<<"---------smooth:"<<smooth_cost<<std::endl;
//            //TODO change index type of mrf::Graph
//            costs[component].push_back({static_cast<int>(cid), data_cost, smooth_cost[0], smooth_cost[1], smooth_cost[2], feature_cost});

//        }
//        int label = i + 1;

//        for (std::size_t j = 0; j < mrfs.size(); ++j)
//        {
//            mrfs[j]->set_data_costs(label, costs[j]);
//        }
//    }

//    for (std::size_t i = 0; i < mrfs.size(); ++i)
//    {
//        /* Set costs for undefined label */
//        std::vector<mrf::SparseDataCost> costs(mrfs[i]->num_sites());
//        for (std::size_t j = 0; j < costs.size(); j++)
//        {
//            costs[j].site = j;
//            costs[j].cost = 1.0f;
//            costs[j].smoothR = 1000.0f;
//            costs[j].smoothG =1000.0f;
//            costs[j].smoothB = 1000.0f;
//            costs[j].featurevalue = -1.0f;
//        }
//        mrfs[i]->set_data_costs(0, costs);
//    }

//}

void view_selection(DataCosts const & data_costs, UniGraph * graph, Settings const & settings)
{
    using uint_t = unsigned int;
    using cost_t = float;
    constexpr uint_t simd_w = mapmap::sys_max_simd_width<cost_t>();
    using unary_t = mapmap::UnaryTable<cost_t, simd_w>;//数据项
    using pairwise_t = mapmap::PairwisePotts<cost_t, simd_w>;

    /* Construct graph */
    mapmap::Graph<cost_t> mgraph(graph->num_nodes());//all the faces

    for (std::size_t i = 0; i < graph->num_nodes(); ++i) //iteration for all the faces
    {
        if (data_costs.col(i).empty())
            continue;

        std::vector<std::size_t> adj_faces = graph->get_adj_nodes(i);
        for (std::size_t j = 0; j < adj_faces.size(); ++j)//构建邻接信息。此处可以利用
        {
            std::size_t adj_face = adj_faces[j];
            if (data_costs.col(adj_face).empty())
            {
                continue;
            }

            /* Uni directional */
            if (i < adj_face)//无向
            {
                mgraph.add_edge(i, adj_face, 1.0f);//构建无向边
            }
        }
    }
    mgraph.update_components();

    mapmap::LabelSet<cost_t, simd_w> label_set(graph->num_nodes(), false);
    for (std::size_t i = 0; i < data_costs.cols(); ++i)
    {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_iv_st<cost_t, simd_w> > labels;
        if (data_costs_for_node.empty())
        {
            labels.push_back(0);
        }
        else
        {
            labels.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j)
            {
                labels[j] = data_costs_for_node[j].first + 1;//等于那个视口，也就是那个图像，是否在这里设置不要与第一个相似呢。
            }
        }

        label_set.set_label_set_for_node(i, labels);
    }

    //cost data
    std::vector<unary_t> unaries;
    unaries.reserve(data_costs.cols());
    pairwise_t pairwise(1.0f);
    for (std::size_t i = 0; i < data_costs.cols(); ++i)
    {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_s_t<cost_t, simd_w> > costs;//每个面在所有视口的投影代价值
        if (data_costs_for_node.empty())
        {
            costs.push_back(1.0f);
        }
        else
        {
            costs.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j)
            {
                float cost = data_costs_for_node[j].second;
                costs[j] = cost;
            }

        }

        unaries.emplace_back(i, &label_set);
        unaries.back().set_costs(costs);//每个面在所有视口的投影代价
    }

    mapmap::StopWhenReturnsDiminish<cost_t, simd_w> terminate(5, 0.01);
    std::vector<mapmap::_iv_st<cost_t, simd_w> > solution;

    auto display = [](const mapmap::luint_t time_ms,
            const mapmap::_iv_st<cost_t, simd_w> objective)
    {
        std::cout << "\t\t" << time_ms / 1000 << "\t" << objective << std::endl;
    };

    /* Create mapMAP solver object. */
    mapmap::mapMAP<cost_t, simd_w> solver;
    solver.set_graph(&mgraph);
    solver.set_label_set(&label_set);
    for(std::size_t i = 0; i < graph->num_nodes(); ++i)
    {
        solver.set_unary(i, &unaries[i]);
    }
    solver.set_pairwise(&pairwise);
    solver.set_logging_callback(display);
    solver.set_termination_criterion(&terminate);

    /* Pass configuration arguments (optional) for solve. */
    mapmap::mapMAP_control ctr;
    ctr.use_multilevel = true;
    ctr.use_spanning_tree = true;
    ctr.use_acyclic = true;
    ctr.spanning_tree_multilevel_after_n_iterations = 5;
    ctr.force_acyclic = true;
    ctr.min_acyclic_iterations = 5;
    ctr.relax_acyclic_maximal = true;
    ctr.tree_algorithm = mapmap::LOCK_FREE_TREE_SAMPLER;

    /* Set false for non-deterministic (but faster) mapMAP execution. */
    ctr.sample_deterministic = true;
    ctr.initial_seed = 548923723;

    std::cout << "\tOptimizing:\n\t\tTime[s]\tEnergy" << std::endl;
//    solver.optimize(solution, ctr);

    try
        {
            solver.optimize(solution, ctr);
        }
        catch(std::runtime_error& e)
        {
            std::cout <<  "Caught an exception: "
                      << e.what()
                      << ", exiting..."
                      << std::endl;
        }
        catch(std::domain_error& e)
        {
            std::cout <<  "Caught an exception: "
                      << e.what()
                      << ", exiting..."
                      << std::endl;
        }


    /* Label 0 is undefined. */
    std::size_t num_labels = data_costs.rows() + 1;
    std::size_t undefined = 0;
    /* Extract resulting labeling from solver. */
    for (std::size_t i = 0; i < graph->num_nodes(); ++i)
    {
        int label = label_set.label_from_offset(i, solution[i]);
        if (label < 0 || num_labels <= static_cast<std::size_t>(label))
        {
            throw std::runtime_error("Incorrect labeling");
        }
        if (label == 0)
            undefined += 1;
        graph->set_label(i, static_cast<std::size_t>(label));
    }
    std::cout << '\t' << undefined << " faces have not been seen" << std::endl;
}

void view_selection_by_mmap(DataCosts const & data_costs, DataCosts const & smooth_costs, UniGraph * graph, Settings const & settings)
{
    std::cout<<"-----------AAAAAAA---------------"<<std::endl;
    using uint_t = unsigned int;
    using cost_t = float;
    constexpr uint_t simd_w = mapmap::sys_max_simd_width<cost_t>();
    using unary_t = mapmap::UnaryTable<cost_t, simd_w>;//数据项
    using pairwise_table_t = mapmap::PairwiseTable<cost_t, simd_w>;//平滑项
    using pairwise_t = mapmap::PairwisePotts<cost_t, simd_w>;//for test

    /* Construct graph */
    mapmap::Graph<cost_t> mgraph(graph->num_nodes());//all the faces

    for (std::size_t i = 0; i < graph->num_nodes(); ++i) //iteration for all the faces
    {
        if (data_costs.col(i).empty())
        {
            continue;
        }

        std::vector<std::size_t> adj_faces = graph->get_adj_nodes(i);
        for (std::size_t j = 0; j < adj_faces.size(); ++j)
        {
            std::size_t adj_face = adj_faces[j];
            if (data_costs.col(adj_face).empty())
            {
                continue;
            }

            /* Uni directional */
            if (i < adj_face)//无向
            {
                mgraph.add_edge(i, adj_face, 1.0f);
            }
        }
    }
    mgraph.update_components();

    mapmap::LabelSet<cost_t, simd_w> label_set(graph->num_nodes(), false);
    for (std::size_t i = 0; i < data_costs.cols(); ++i)
    {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_iv_st<cost_t, simd_w> > labels;
        if (data_costs_for_node.empty())
        {
            labels.push_back(0);
        }
        else
        {
            labels.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j)
            {
                labels[j] = data_costs_for_node[j].first + 1;//等于那个视口，也就是那个图像，是否在这里设置不要与第一个相似呢。
            }
        }

        label_set.set_label_set_for_node(i, labels);
    }

    std::vector<unary_t> unaries;
    unaries.reserve(data_costs.cols());

    pairwise_t    pairwise(1.0f);

    //add smooth cost
    for (std::size_t i = 0; i < data_costs.cols(); ++i)
    {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_s_t<cost_t, simd_w> > costs;

        if (data_costs_for_node.empty())
        {
            costs.push_back(1.0f);

        }
        else
        {
            costs.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j)
            {
                float cost = data_costs_for_node[j].second;
                costs[j] = cost;
            }

        }

        unaries.emplace_back(i, &label_set);//call unary_t constructor for data allocation
        unaries.back().set_costs(costs);//push_back data
    }


    mapmap::StopWhenReturnsDiminish<cost_t, simd_w> terminate(5, 0.01);
    std::vector<mapmap::_iv_st<cost_t, simd_w> > solution;

    auto display = [](const mapmap::luint_t time_ms,
            const mapmap::_iv_st<cost_t, simd_w> objective)
    {
        std::cout << "\t\t" << time_ms / 1000 << "\t" << objective << std::endl;
    };


    /* Create mapMAP solver object. */
    mapmap::mapMAP<cost_t, simd_w> solver;
    solver.set_graph(&mgraph);
    solver.set_label_set(&label_set);

    for(std::size_t i = 0; i < graph->num_nodes(); ++i)
    {
        solver.set_unary(i, &unaries[i]);

    }

    //smooth cost
    //    for (std::size_t f_idx = 0; f_idx < graph->num_nodes(); ++f_idx) //迭代所有的面
    int count = 0;
    for(std::size_t f_idx = 0; f_idx < graph->num_nodes(); ++f_idx)
    {
        //        std::vector<std::size_t> adj_faces = graph->get_adj_nodes(f_idx);//当前面的所有邻接面
        DataCosts::Column const & smooth_costs_for_node = smooth_costs.col(f_idx);//每个面在所有视口上的投影信息
        if(smooth_costs_for_node.empty())
        {
//            std::cout<<"f_idx:"<<f_idx<<std::endl;
//            solver.set_pairwise(edge_idx, &pairwise);
            continue;
        }

        std::vector<uint64_t>  edges = mgraph.inc_edges(f_idx);//与当前点相连的所有的边

        for(int adj_idx = 0; adj_idx < edges.size(); adj_idx++)//遍历当前面所有的边
        {
            int edge_idx = edges[adj_idx];

            mapmap::GraphEdge<cost_t>  edge = mgraph.edges()[edge_idx];
//            mapmap::myEdge  edge = mgraph.getEdgeinfo(edge_idx);//得到当前边两端的两个端点（也就是时相邻的两个面）

            if(edge.node_a != f_idx)//注意是无向边
            {
                continue;
            }
//                adj_face = edge.node_a;
            int adj_face = edge.node_b;


//            int adj_face = edge.node_b;

            DataCosts::Column const & adj_smooth_costs_for_node = smooth_costs.col(adj_face);//邻接面在所有视口上的投影信息

            std::vector<mapmap::_s_t<cost_t, simd_w> > smooths;//当前点和周围一个邻接点取不同标签的代价
            smooths.reserve(smooth_costs_for_node.size()*adj_smooth_costs_for_node.size());
            //构建代价表
            for(int c_idx = 0; c_idx < smooth_costs_for_node.size(); c_idx++)
            {
                float value = smooth_costs_for_node[c_idx].second;//当前面在当前视口的投影颜色（或者梯度度量）


                for(int a_idx = 0; a_idx < adj_smooth_costs_for_node.size(); a_idx++)
                {
                    float adj_value = adj_smooth_costs_for_node[a_idx].second;//邻接面在前视口的投影颜色（或者梯度度量）
                    float  dif = std::abs(value - adj_value);//color different between adjacent faces;

                    smooths[adj_smooth_costs_for_node.size()*c_idx + a_idx] = dif;
                }
            }

            std::unique_ptr<pairwise_table_t> smooth;
//           if(edge.nodaA != f_idx)//注意是无向边
//           {
               smooth = std::unique_ptr<pairwise_table_t>(new pairwise_table_t(f_idx, adj_face, &label_set, smooths));
//            smooth = std::unique_ptr<pairwise_table_t>(new pairwise_table_t(f_idx, adj_face, &label_set));
//           }
//           else
//           {
//               smooth = std::unique_ptr<pairwise_table_t>(new pairwise_table_t(adj_face, f_idx, &label_set, smooths));
//           }

//               std::cout<<"edge_idx"<<edge_idx<<std::endl;
            solver.set_pairwise(edge_idx, smooth.get());

//            solver.set_pairwise(edge_idx, &pairwise);

            count++;

        }
    }
    std::cout<<"-------num:"<<count<<"  node edge:"<<mgraph.num_edges()<<std::endl;

//    solver.set_pairwise(&pairwise);

    solver.set_logging_callback(display);
    solver.set_termination_criterion(&terminate);

    /* Pass configuration arguments (optional) for solve. */
    mapmap::mapMAP_control ctr;
    ctr.use_multilevel = false;
    ctr.use_spanning_tree = true;
    ctr.use_acyclic = true;
    ctr.spanning_tree_multilevel_after_n_iterations = 5;
    ctr.force_acyclic = true;
    ctr.min_acyclic_iterations = 5;
    ctr.relax_acyclic_maximal = true;
    ctr.tree_algorithm = mapmap::LOCK_FREE_TREE_SAMPLER;

    /* Set false for non-deterministic (but faster) mapMAP execution. */
    ctr.sample_deterministic = false;
    ctr.initial_seed = 548923723;

    std::cout << "\tOptimizing:\n\t\tTime[s]\tEnergy" << std::endl;
//    solver.optimize(solution, ctr);

    try
        {
            solver.optimize(solution, ctr);
        }
        catch(std::runtime_error& e)
        {
            std::cout <<  "Caught an exception: "
                      << e.what()
                      << ", exiting..."
                      << std::endl;
        }
        catch(std::domain_error& e)
        {
            std::cout <<  "Caught an exception: "
                      << e.what()
                      << ", exiting..."
                      << std::endl;
        }

    /* Label 0 is undefined. */
    std::size_t num_labels = data_costs.rows() + 1;
    std::size_t undefined = 0;
    /* Extract resulting labeling from solver. */
    for (std::size_t i = 0; i < graph->num_nodes(); ++i)
    {
        int label = label_set.label_from_offset(i, solution[i]);
        if (label < 0 || num_labels <= static_cast<std::size_t>(label))
        {
            throw std::runtime_error("Incorrect labeling");
        }
        if (label == 0)
            undefined += 1;
        graph->set_label(i, static_cast<std::size_t>(label));
    }
    std::cout << '\t' << undefined << " faces have not been seen" << std::endl;
}

void view_selection_and_feature(DataCosts const & data_costs, SmoothCosts const & smooth_costs, DataCosts const &feature_costs,
                                UniGraph * graph, Settings const & settings)
{
    UniGraph mgraph(*graph);//邻接面图构造无向图
    isolate_unseen_faces(&mgraph, data_costs);//检查任意视口都不可见的面，并移除与改面相关的所有的邻接关系

    unsigned int num_components = 0;

    std::vector<FaceInfo> face_infos(mgraph.num_nodes());//也就是面的个数
    //    std::cout<<"-------->mgraph.num_nodes():"<<mgraph.num_nodes()<<std::endl;
    std::vector<std::vector<std::size_t> > components;//对三角网格进行剖分（相邻面相同标签聚类在一起），每个剖分存储所有面的索引
    mgraph.get_subgraphs(0, &components);//相同标签相邻面聚合在一起
    for (std::size_t i = 0; i < components.size(); ++i)
    {
        if (components.size() > 1000)
        {
            num_components += 1;
        }
        for (std::size_t j = 0; j < components[i].size(); ++j)//遍历每个patch中的所有面
        {
            face_infos[components[i][j]] = {i, j};//剖分集合的索引，面的索引
        }
    }
//    std::cout<<"------------>"<<components.size()<<std::endl;

#ifdef RESEARCH
    mrf::SOLVER_TYPE solver_type = mrf::GCO;
#else
    mrf::SOLVER_TYPE solver_type = mrf::LBP;
//        mrf::SOLVER_TYPE solver_type = mrf::ICM;
#endif

    /* Label 0 is undefined.*/
    const std::size_t num_labels = data_costs.rows() + 1;//视口的个数+1，0表示没有使用
    std::vector<mrf::Graph::Ptr>   mrfs(components.size());
    for (std::size_t i = 0; i < components.size(); ++i)
    {
        mrfs[i] = mrf::Graph::create(components[i].size(), num_labels, solver_type);//每一个块创建一个
    }

    /* Set neighbors must be called prior to set_data_costs (LBP). */
    set_neighbors(mgraph, face_infos, mrfs);//设置MRF邻接点信息（统一标签聚类之间相邻面之间建立双向连接关系）

//    std::cout<<"-------------->aa"<<std::endl;
    //    set_data_costs(face_infos, data_costs, mrfs);//设置MRF代价信息
    //set_data_smooth_costs(face_infos, data_costs, smooth_costs, mrfs);//设置MRF代价信息
    set_data_feature_costs(face_infos, data_costs, feature_costs, mrfs);//设置MRF代价信息

    bool multiple_components_simultaneously = false;//是否多个同时优化
#ifdef RESEARCH
    multiple_components_simultaneously = true;
#endif
#ifndef _OPENMP
    multiple_components_simultaneously = false;
#endif

    if (multiple_components_simultaneously)
    {
        if (num_components > 0)
        {
            //            std::cout << "\tOptimizing " << num_components
            //                << " components simultaneously." << std::endl;
        }
        //        std::cout << "\tComp\tIter\tEnergy\t\tRuntime" << std::endl;
    }
#ifdef RESEARCH
#pragma omp parallel for schedule(dynamic)
#endif
    for (std::size_t i = 0; i < components.size(); ++i)
    {
        switch (settings.smoothness_term)
        {
        case SMOOTHNESS_TERM_POTTS:
        {
            //                mrfs[i]->set_smooth_cost(*potts);
//            std::cout<<"---MRF Potts"<<std::endl;
            mrfs[i]->set_smooth_cost(*orgpotts);
        }
            break;
        case SMOOTHNESS_TERM_FUNC:
        {
//            std::cout<<"---MRF Func"<<std::endl;
            mrfs[i]->set_smooth_cost(*potts);

        }
            break;

        }

        bool verbose = mrfs[i]->num_sites() > 10000;

        util::WallTimer timer;

        mrf::ENERGY_TYPE const zero = mrf::ENERGY_TYPE(0);
        mrf::ENERGY_TYPE last_energy = zero;
        mrf::ENERGY_TYPE energy = mrfs[i]->compute_energy();
        mrf::ENERGY_TYPE diff = last_energy - energy;
        unsigned int iter = 0;

        std::string const comp = util::string::get_filled(i, 4);

        if (verbose && !multiple_components_simultaneously)
        {
            //            std::cout << "\tComp\tIter\tEnergy\t\tRuntime" << std::endl;
        }
        while (diff != zero)
//        while(iter<50)
        {
#pragma omp critical
            if (verbose)
            {
                                std::cout << "\t" << comp << "\t" << iter << "\t" << energy
                                    << "\t" << timer.get_elapsed_sec() << std::endl;
            }
            last_energy = energy;
            ++iter;
            energy = mrfs[i]->optimize(1);
            diff = last_energy - energy;
            if (diff <= zero)
                break;
        }

#pragma omp critical
        if (verbose)
        {
            //            std::cout << "\t" << comp << "\t" << iter << "\t" << energy << std::endl;
            if (diff == zero)
            {
                //                std::cout << "\t" << comp << "\t" << "Converged" << std::endl;
            }
            if (diff < zero) {
                //                std::cout << "\t" << comp << "\t"
                //                    << "Increase of energy - stopping optimization" << std::endl;
            }
        }

        /* Extract resulting labeling from MRF. */
        for (std::size_t j = 0; j < components[i].size(); ++j)
        {
            int label = mrfs[i]->what_label(static_cast<int>(j));
            assert(0 <= label && static_cast<std::size_t>(label) < num_labels);
            graph->set_label(components[i][j], static_cast<std::size_t>(label));
        }
    }
}

void view_selection_with_detail(DataCosts const & data_costs, SmoothCosts const & smooth_costs, DataCosts & detail_costs,
                               UniGraph * graph, Settings const & settings)

{
    UniGraph mgraph(*graph);//邻接面图构造无向图
    isolate_unseen_faces(&mgraph, data_costs);//检查任意视口都不可见的面，并移除与改面相关的所有的邻接关系

    unsigned int num_components = 0;

    std::vector<FaceInfo> face_infos(mgraph.num_nodes());//也就是面的个数
    //    std::cout<<"-------->mgraph.num_nodes():"<<mgraph.num_nodes()<<std::endl;
    std::vector<std::vector<std::size_t> > components;//对三角网格进行剖分（相邻面相同标签聚类在一起），每个剖分存储所有面的索引
    mgraph.get_subgraphs(0, &components);//相同标签相邻面聚合在一起
    for (std::size_t i = 0; i < components.size(); ++i)
    {
        if (components.size() > 1000)
        {
            num_components += 1;
        }
        for (std::size_t j = 0; j < components[i].size(); ++j)//遍历每个patch中的所有面
        {
            face_infos[components[i][j]] = {i, j};//剖分集合的索引，面的索引
        }
    }
//    std::cout<<"------------>"<<components.size()<<std::endl;

#ifdef RESEARCH
    mrf::SOLVER_TYPE solver_type = mrf::GCO;
#else
    mrf::SOLVER_TYPE solver_type = mrf::LBP;
//        mrf::SOLVER_TYPE solver_type = mrf::ICM;
#endif

    /* Label 0 is undefined.*/
    const std::size_t num_labels = data_costs.rows() + 1;//视口的个数+1，0表示没有使用
    std::vector<mrf::Graph::Ptr>   mrfs(components.size());
    for (std::size_t i = 0; i < components.size(); ++i)
    {
        mrfs[i] = mrf::Graph::create(components[i].size(), num_labels, solver_type);//每一个块创建一个
    }

    /* Set neighbors must be called prior to set_data_costs (LBP). */
    set_neighbors(mgraph, face_infos, mrfs);//设置MRF邻接点信息（统一标签聚类之间相邻面之间建立双向连接关系）

//    std::cout<<"-------------->aa"<<std::endl;
    //    set_data_costs(face_infos, data_costs, mrfs);//设置MRF代价信息
//    set_data_smooth_costs(face_infos, data_costs, smooth_costs, mrfs);//设置MRF代价信息
    set_data_smooth_detail_costs(face_infos, data_costs, smooth_costs, detail_costs, mrfs);//设置MRF代价信息


    bool multiple_components_simultaneously = false;//是否多个同时优化
#ifdef RESEARCH
    multiple_components_simultaneously = true;
#endif
#ifndef _OPENMP
    multiple_components_simultaneously = false;
#endif

    if (multiple_components_simultaneously)
    {
        if (num_components > 0)
        {
            //            std::cout << "\tOptimizing " << num_components
            //                << " components simultaneously." << std::endl;
        }
        //        std::cout << "\tComp\tIter\tEnergy\t\tRuntime" << std::endl;
    }

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < components.size(); ++i)
    {
        switch (settings.smoothness_term)
        {
        case SMOOTHNESS_TERM_POTTS:
        {
            //                mrfs[i]->set_smooth_cost(*potts);
//            std::cout<<"---MRF Potts"<<std::endl;
            mrfs[i]->set_smooth_cost(*orgpotts);
        }
            break;
        case SMOOTHNESS_TERM_FUNC:
        {
            mrfs[i]->set_smooth_cost(*potts);
        }
            break;

        }

        bool verbose = mrfs[i]->num_sites() > 10000;

        util::WallTimer timer;

        mrf::ENERGY_TYPE const zero = mrf::ENERGY_TYPE(0);
        mrf::ENERGY_TYPE last_energy = zero;
        mrf::ENERGY_TYPE energy = mrfs[i]->compute_energy();
        mrf::ENERGY_TYPE diff = last_energy - energy;
        unsigned int iter = 0;

        std::string const comp = util::string::get_filled(i, 4);

        if (verbose && !multiple_components_simultaneously)
        {
            //            std::cout << "\tComp\tIter\tEnergy\t\tRuntime" << std::endl;
        }
//        while (diff != zero)
        while(iter < 30)
        {
//#pragma omp critical
//            if (verbose)
//            {
//                                std::cout << "\t" << comp << "\t" << iter << "\t" << energy
//                                    << "\t" << timer.get_elapsed_sec() << std::endl;
//            }
            last_energy = energy;
            ++iter;
            energy = mrfs[i]->optimize(1);
            diff = last_energy - energy;
//            if (diff <= zero)
//                break;
        }
//                    std::cout << "\t" << comp << "\t" << iter << "\t" << energy << std::endl;

//#pragma omp critical
//        if (verbose)
//        {
//            //            std::cout << "\t" << comp << "\t" << iter << "\t" << energy << std::endl;
//            if (diff == zero)
//            {
//                //                std::cout << "\t" << comp << "\t" << "Converged" << std::endl;
//            }
//            if (diff < zero) {
//                //                std::cout << "\t" << comp << "\t"
//                //                    << "Increase of energy - stopping optimization" << std::endl;
//            }
//        }

        /* Extract resulting labeling from MRF. */
        for (std::size_t j = 0; j < components[i].size(); ++j)
        {
            int label = mrfs[i]->what_label(static_cast<int>(j));
            assert(0 <= label && static_cast<std::size_t>(label) < num_labels);
            graph->set_label(components[i][j], static_cast<std::size_t>(label));
        }
    }
}

void view_selectionMRF(DataCosts const & data_costs, UniGraph * graph, Settings const & settings)
{
    UniGraph mgraph(*graph);//邻接面图构造无向图
    isolate_unseen_faces(&mgraph, data_costs);//检查任意视口都不可见的面，并移除与改面相关的所有的邻接关系

    unsigned int num_components = 0;

    std::vector<FaceInfo> face_infos(mgraph.num_nodes());//也就是面的个数
    //    std::cout<<"-------->mgraph.num_nodes():"<<mgraph.num_nodes()<<std::endl;
    std::vector<std::vector<std::size_t> > components;//对三角网格进行剖分（相邻面相同标签聚类在一起），每个剖分存储所有面的索引
    mgraph.get_subgraphs(0, &components);//相同标签相邻面聚合在一起
    for (std::size_t i = 0; i < components.size(); ++i)
    {
        if (components.size() > 1000)
        {
            num_components += 1;
        }
        for (std::size_t j = 0; j < components[i].size(); ++j)
        {
            face_infos[components[i][j]] = {i, j};//剖分集合的索引，面的索引
        }
    }
//    std::cout<<"------------>"<<components.size()<<std::endl;

#ifdef RESEARCH
    mrf::SOLVER_TYPE solver_type = mrf::GCO;
#else
    mrf::SOLVER_TYPE solver_type = mrf::LBP;
    //    mrf::SOLVER_TYPE solver_type = mrf::ICM;
#endif

    /* Label 0 is undefined.*/
    const std::size_t num_labels = data_costs.rows() + 1;//视口的个数+1，0表示没有使用
    std::vector<mrf::Graph::Ptr>   mrfs(components.size());
    for (std::size_t i = 0; i < components.size(); ++i)
    {
        mrfs[i] = mrf::Graph::create(components[i].size(), num_labels, solver_type);//每一个块创建一个
    }

    /* Set neighbors must be called prior to set_data_costs (LBP). */
    set_neighbors(mgraph, face_infos, mrfs);//设置MRF邻接点信息（统一标签聚类之间相邻面之间建立双向连接关系）

    set_data_costs(face_infos, data_costs, mrfs);//设置MRF代价信息

    bool multiple_components_simultaneously = false;//是否多个同时优化
    #ifdef RESEARCH
    multiple_components_simultaneously = true;
    #endif
    #ifndef _OPENMP
    multiple_components_simultaneously = false;
    #endif

    if (multiple_components_simultaneously)
    {
        if (num_components > 0)
        {
//            std::cout << "\tOptimizing " << num_components
//                << " components simultaneously." << std::endl;
        }
//        std::cout << "\tComp\tIter\tEnergy\t\tRuntime" << std::endl;
    }
    #ifdef RESEARCH
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (std::size_t i = 0; i < components.size(); ++i)
    {
        switch (settings.smoothness_term)
        {
        case SMOOTHNESS_TERM_POTTS:
            mrfs[i]->set_smooth_cost(*orgpotts);
            break;
        }

        bool verbose = mrfs[i]->num_sites() > 10000;

        util::WallTimer timer;

        mrf::ENERGY_TYPE const zero = mrf::ENERGY_TYPE(0);
        mrf::ENERGY_TYPE last_energy = zero;
        mrf::ENERGY_TYPE energy = mrfs[i]->compute_energy();
        mrf::ENERGY_TYPE diff = last_energy - energy;
        unsigned int iter = 0;

        std::string const comp = util::string::get_filled(i, 4);

        if (verbose && !multiple_components_simultaneously)
        {
//            std::cout << "\tComp\tIter\tEnergy\t\tRuntime" << std::endl;
        }
        while (diff != zero)
        {
            #pragma omp critical
            if (verbose)
            {
//                std::cout << "\t" << comp << "\t" << iter << "\t" << energy
//                    << "\t" << timer.get_elapsed_sec() << std::endl;
            }
            last_energy = energy;
            ++iter;
            energy = mrfs[i]->optimize(1);
            diff = last_energy - energy;
            if (diff <= zero)
                break;
        }

        #pragma omp critical
        if (verbose)
        {
//            std::cout << "\t" << comp << "\t" << iter << "\t" << energy << std::endl;
            if (diff == zero)
            {
//                std::cout << "\t" << comp << "\t" << "Converged" << std::endl;
            }
            if (diff < zero) {
//                std::cout << "\t" << comp << "\t"
//                    << "Increase of energy - stopping optimization" << std::endl;
            }
        }

        /* Extract resulting labeling from MRF. */
        for (std::size_t j = 0; j < components[i].size(); ++j)
        {
            int label = mrfs[i]->what_label(static_cast<int>(j));
            assert(0 <= label && static_cast<std::size_t>(label) < num_labels);
            graph->set_label(components[i][j], static_cast<std::size_t>(label));
        }
    }
}


void view_selectionTwoPass(DataCosts const & data_costs, UniGraph * graph, Settings const & settings)
{
    using uint_t = unsigned int;
    using cost_t = float;
    constexpr uint_t simd_w = mapmap::sys_max_simd_width<cost_t>();
    using unary_t = mapmap::UnaryTable<cost_t, simd_w>;
    using pairwise_t = mapmap::PairwisePotts<cost_t, simd_w>;

    /* Construct graph */
    mapmap::Graph<cost_t> mgraph(graph->num_nodes());//all the faces

    for (std::size_t i = 0; i < graph->num_nodes(); ++i) //iteration for all the faces
    {
        if (data_costs.col(i).empty())
            continue;

        std::vector<std::size_t> adj_faces = graph->get_adj_nodes(i);
        for (std::size_t j = 0; j < adj_faces.size(); ++j)
        {
            std::size_t adj_face = adj_faces[j];
            if (data_costs.col(adj_face).empty())
            {
                continue;
            }

            /* Uni directional */
            if (i < adj_face)//无向
            {
                mgraph.add_edge(i, adj_face, 1.0f);
            }
        }
    }
    mgraph.update_components();

    mapmap::LabelSet<cost_t, simd_w> label_set(graph->num_nodes(), false);
    for (std::size_t i = 0; i < data_costs.cols(); ++i)//所有的面
    {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_iv_st<cost_t, simd_w> > labels;
        if (data_costs_for_node.empty())
        {
            labels.push_back(0);
        }
        else
        {
            labels.resize(data_costs_for_node.size());
//            std::size_t  first_label = graph->get_label(i);//第一次分割的结果
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j)
            {
//                if(first_label == data_costs_for_node[j].first + 1 )
//                {
////                    labels[j] = 0;
//                    labels.push_back(0);

//                }
//                else
                {
                    labels[j] = data_costs_for_node[j].first + 1;//等于那个视口，也就是那个图像，是否在这里设置不要与第一个相似呢。
                }
            }
        }

        label_set.set_label_set_for_node(i, labels);
    }

    std::vector<unary_t> unaries;
    unaries.reserve(data_costs.cols());
    pairwise_t pairwise(1.0f);
    for (std::size_t i = 0; i < data_costs.cols(); ++i)
    {
        DataCosts::Column const & data_costs_for_node = data_costs.col(i);

        std::vector<mapmap::_s_t<cost_t, simd_w> > costs;
        if (data_costs_for_node.empty())
        {
            costs.push_back(1.0f);
        }
        else
        {
            costs.resize(data_costs_for_node.size());
            std::size_t  first_label = graph->get_label(i);//第一次分割的结果
//            std::cout<<"------->aa"<<first_label<<std::endl;
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j)
            {
                float weight = 1;
//                if(first_label == data_costs_for_node[j].first + 1)
//                {
//                    weight = 0.0;
//                    costs.push_back(1.0f);
//                }
//                else
                {
                    float cost = data_costs_for_node[j].second;
                    costs[j] = cost*weight;
                }

            }

        }

        unaries.emplace_back(i, &label_set);
        unaries.back().set_costs(costs);
    }

    mapmap::StopWhenReturnsDiminish<cost_t, simd_w> terminate(5, 0.01);
    std::vector<mapmap::_iv_st<cost_t, simd_w> > solution;

    auto display = [](const mapmap::luint_t time_ms,
            const mapmap::_iv_st<cost_t, simd_w> objective)
    {
        std::cout << "\t\t" << time_ms / 1000 << "\t" << objective << std::endl;
    };

    /* Create mapMAP solver object. */
    mapmap::mapMAP<cost_t, simd_w> solver;
    solver.set_graph(&mgraph);
    solver.set_label_set(&label_set);
    for(std::size_t i = 0; i < graph->num_nodes(); ++i)
    {
        solver.set_unary(i, &unaries[i]);
    }
    solver.set_pairwise(&pairwise);
    solver.set_logging_callback(display);
    solver.set_termination_criterion(&terminate);

    /* Pass configuration arguments (optional) for solve. */
    mapmap::mapMAP_control ctr;
    ctr.use_multilevel = true;
    ctr.use_spanning_tree = true;
    ctr.use_acyclic = true;
    ctr.spanning_tree_multilevel_after_n_iterations = 5;
    ctr.force_acyclic = true;
    ctr.min_acyclic_iterations = 5;
    ctr.relax_acyclic_maximal = true;
    ctr.tree_algorithm = mapmap::LOCK_FREE_TREE_SAMPLER;

    /* Set false for non-deterministic (but faster) mapMAP execution. */
    ctr.sample_deterministic = true;
    ctr.initial_seed = 548923723;

    std::cout << "\tOptimizing:\n\t\tTime[s]\tEnergy" << std::endl;
    solver.optimize(solution, ctr);

    /* Label 0 is undefined. */
    std::size_t num_labels = data_costs.rows() + 1;
    std::size_t undefined = 0;
    /* Extract resulting labeling from solver. */
    for (std::size_t i = 0; i < graph->num_nodes(); ++i)
    {
        int label = label_set.label_from_offset(i, solution[i]);
        if (label < 0 || num_labels <= static_cast<std::size_t>(label))
        {
            throw std::runtime_error("Incorrect labeling");
        }
        if (label == 0)
            undefined += 1;
        graph->set_TwoPasslabel(i, static_cast<std::size_t>(label));
    }
    std::cout << '\t' << undefined << " faces have not been seen" << std::endl;
}

//面在任意视口都不可见，//检查不可见的面，并移除所有的邻接关系
/** Remove all edges of nodes which corresponding face has not been seen in any texture view. */
void  isolate_unseen_faces(UniGraph * graph, DataCosts const & data_costs)
{
    int num_unseen_faces = 0;
    for (std::uint32_t i = 0; i < data_costs.cols(); i++) //每个面所有的有效视口的信息（面的个数）
    {
        DataCosts::Column const & data_costs_for_face = data_costs.col(i);//每个面所有有效视口的信息

        if (data_costs_for_face.size() == 0) //没有有效信息说明没有任意一个视口可以看见这个面
        {
            num_unseen_faces++;

            std::vector<std::size_t>  const  & adj_nodes = graph->get_adj_nodes(i);//这个面的所有邻接面
            for (std::size_t j = 0; j < adj_nodes.size(); j++)
            {
                graph->remove_edge(i, adj_nodes[j]);//存在不可见的面，移除该面所有的邻接关系
            }
        }

    }
    std::cout << "\t" << num_unseen_faces << " faces have not been seen by a view." << std::endl;
}

TEX_NAMESPACE_END
