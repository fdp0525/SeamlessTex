/*
 * Copyright (C) 2015, Nils Moehrle, Benjamin Richter
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <algorithm>
#include <iostream>
#include "lbp_graph.h"

MRF_NAMESPACE_BEGIN

float featuremapcost(int, int, int l1, int l2, float v1, float v2)
{
//    std::cout<<"v1:"<<v1<<" v2:"<<v2<<std::endl;
    if(l1 ==0 || l2 == 0)
    {
        return 1.0f;
    }

    if(l1 == l2 )
    {
        if(v1 < 0.5 && v2 < 0.5)
        {
            return 0;
        }
        else
        {
            return 0.1;
        }
    }
    else
    {
        if(v1 < 0.3 && v2 < 0.3)
        {
            return 0.9;
        }
        else
        {
            return 1;
        }
    }
}
/** Potts model */
float pottscost(int, int, int l1, int l2)
{
    return (l1 == l2 && l1 != 0 && l2 != 0) ? 0.0f : 1.0f;
    //    float value = 0.0;
    //    if(l1 == l2 && l1 != 0 && l2 != 0)
    //    {
    //        value = 0.0f;
    //    }
    //    else
    //    {
    //        value = std::abs(v1 - v2);
    //    }
    //    return value;
}

float detailsmooth(int, int, int l1, int l2, float v1 , float v2)
{
//    if(l1 == l2)
//    {
//        return 0;
//    }
//    std::cout<<"---------v1:"<<v1<<"  v2:"<<v2<<std::endl;
    //应该考虑两边相似的时候两端的值是否相同？？？？？？？？？？？？？？？？/
   float value = 1.0;
    if(l1 == l2)
    {
        value= std::abs(v1 - v2);
//        std::cout<<"---------------------->"<<value<<std::endl;
    }
    else
    {
        value = 1.0f;
    }

    return value;

}
/**
 * @brief LBPGraph::LBPGraph
 * @param num_sites  相同标签相邻面的聚类个数
 * @param num_labels 第二个参数就是标签的个数
 */
LBPGraph::LBPGraph(int num_sites, int num_labels)
    : vertices(num_sites)
{

}

ENERGY_TYPE LBPGraph::compute_energy() {
    ENERGY_TYPE energy = 0;

    #pragma omp parallel for reduction(+:energy)
    for (std::size_t vertex_idx = 0; vertex_idx < vertices.size(); ++vertex_idx)
    {
        Vertex const & vertex = vertices[vertex_idx];
        energy += vertex.data_cost;
    }

    #pragma omp parallel for reduction(+:energy)
    for (std::size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx)
    {
        DirectedEdge const & edge = edges[edge_idx];
//        float v1_R = vertices[edge.v1].smooth_costR;
//        float v1_G = vertices[edge.v1].smooth_costG;
//        float v1_B = vertices[edge.v1].smooth_costB;
//        float v2 = vertices[edge.v2].smooth_cost;
//        float v2_R = vertices[edge.v2].smooth_costR;
//        float v2_G = vertices[edge.v2].smooth_costG;
//        float v2_B = vertices[edge.v2].smooth_costB;

//        energy += 100* smooth_cost_func(edge.v1, edge.v2,
//            vertices[edge.v1].label, vertices[edge.v2].label, v1, v2);
//        std::cout<<"4-----------v1_R:"<<v1_R<<"  G:"<<v1_G<<" B:"<<v1_B<<" v2_R:"<<v2_R<<" v2_G:"<<v2_G<<" v2_B:"<<v2_B<<std::endl;

        float detail_v1 = vertices[edge.v1].feature_cost;
        float detail_v2 = vertices[edge.v2].feature_cost;
        //for feature
        energy += detailsmooth(edge.v1, edge.v2, vertices[edge.v1].label, vertices[edge.v2].label, detail_v1, detail_v2)
                +pottscost(edge.v1, edge.v2, vertices[edge.v1].label, vertices[edge.v2].label);

       //for orgi;
//        energy += pottscost(edge.v1, edge.v2, vertices[edge.v1].label, vertices[edge.v2].label);
    }

    return energy;
}

ENERGY_TYPE LBPGraph::optimize(int num_iterations)
{
    for (int i = 0; i < num_iterations; ++i)
    {
        #pragma omp parallel for
        for (std::size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx)
        {
            DirectedEdge & edge = edges[edge_idx];
            std::vector<int> const & labels1 = vertices[edge.v1].labels;
            std::vector<int> const & labels2 = vertices[edge.v2].labels;
            for (std::size_t j = 0; j < labels2.size(); ++j)
            {
                int label2 = labels2[j];
//                std::cout<<"------------------label:"<<label2<<std::endl;

                ENERGY_TYPE min_energy = std::numeric_limits<ENERGY_TYPE>::max();
                for (std::size_t k = 0; k < labels1.size(); ++k)
                {
                    int label1 = labels1[k];
//                    float v1 = vertices[edge.v1].smooth_cost;
//                    float v2 = vertices[edge.v2].smooth_cost;
//                    ENERGY_TYPE energy = 100*smooth_cost_func(edge.v1, edge.v2, label1, label2, v1, v2)
//                        + vertices[edge.v1].data_costs[k];

//                    float v1_R = vertices[edge.v1].data_costRs[k];
//                    float v1_G = vertices[edge.v1].data_costGs[k];
//                    float v1_B = vertices[edge.v1].data_costBs[k];
//            //        float v2 = vertices[edge.v2].smooth_cost;
//                    float v2_R = vertices[edge.v2].data_costRs[j];
//                    float v2_G = vertices[edge.v2].data_costGs[j];
//                    float v2_B = vertices[edge.v2].data_costBs[j];
//                    float value1 = smooth_cost_func(edge.v1, edge.v2, label1, label2, v1_R,v1_G,v1_B, v2_R,v2_G,v2_B);
//                    float value2 =  vertices[edge.v1].data_costs[k];

                    //featurevalue
                    float f1 = vertices[edge.v1].feature_costs[k];
                    float f2 = vertices[edge.v2].feature_costs[j];
//                    std::cout<<f1<<"-------------------"<<f2<<" abs:"<<std::abs(f1 - f2)<<std::endl;

//                    std::cout<<"----------------v:"<<vertices[edge.v1].data_costs[k]<<std::endl;
//                    std::cout<<"v1_R:"<<v1_R<<" v1_G:"<<v1_G<<" v1_B:"<<v1_B<<"---v2_R:"<<v2_R<<" v2_G:"<<v2_G<<" v2_B:"<<v2_B<<std::endl;
                    ///B1 A1
                    //for feature
                    ENERGY_TYPE energy =  detailsmooth(edge.v1, edge.v2, label1, label2, f1, f2)
                           +pottscost(edge.v1, edge.v2, label1, label2)
                        + vertices[edge.v1].data_costs[k];

                    //for orig
//                    ENERGY_TYPE energy =   pottscost(edge.v1, edge.v2, label1, label2)
//                        + vertices[edge.v1].data_costs[k];

                    std::vector<int> const& incoming_edges1 = vertices[edge.v1].incoming_edges;
                    for (std::size_t n = 0; n < incoming_edges1.size(); ++n)//遍历所有的相邻边
                    {
                        DirectedEdge const& pre_edge = edges[incoming_edges1[n]];//取出一个相邻边
                        if (pre_edge.v1 == edge.v2)//不是当前计算的边
                            continue;
                        energy += pre_edge.old_msg[k];//其大所有边的最小能量函数。
                    }

                    if (energy < min_energy)//当前边是否是最小的能量
                        min_energy = energy;
                }
                edge.new_msg[j] = min_energy;//每个label的最优找最小
            }
        }

        #pragma omp parallel for
        for (std::size_t edge_idx = 0; edge_idx < edges.size(); ++edge_idx)
        {
            DirectedEdge & edge = edges[edge_idx];
            edge.new_msg.swap(edge.old_msg);//当前边
            ENERGY_TYPE min_msg = std::numeric_limits<ENERGY_TYPE>::max();
            for (ENERGY_TYPE msg : edge.old_msg)//找到当前边最小的
               min_msg = std::min(min_msg, msg);
            for (ENERGY_TYPE &msg : edge.old_msg)//都减掉这个最新小的
               msg -= min_msg;
        }
    }

    #pragma omp parallel for
    for (std::size_t vertex_idx = 0; vertex_idx < vertices.size(); ++vertex_idx)
    {
        Vertex & vertex = vertices[vertex_idx];
        ENERGY_TYPE min_energy = std::numeric_limits<ENERGY_TYPE>::max();
        for (std::size_t j = 0; j < vertex.labels.size(); ++j)
        {
            ENERGY_TYPE energy = vertex.data_costs[j];
            for (int incoming_edge_idx : vertex.incoming_edges)
            {
                energy += edges[incoming_edge_idx].old_msg[j];//当前标签当前节点所有入射边的和
            }

            if (energy < min_energy)//找标签最好的得到的能量函数最小的。
            {
                min_energy = energy;
                vertex.label = vertex.labels[j];
                vertex.data_cost = vertex.data_costs[j];
                vertex.feature_cost = vertex.feature_costs[j];
//                vertex.smooth_costR = vertex.data_costRs[j];
//                vertex.smooth_costG = vertex.data_costGs[j];
//                vertex.smooth_costB = vertex.data_costBs[j];
            }
        }
    }

    return compute_energy();
}

void LBPGraph::set_smooth_cost(SmoothCostFunction func) {
    smooth_cost_func = func;
}

/**
 * @brief LBPGraph::set_neighbors  两个邻接面之间建立双向的连接关系
 * @param site1
 * @param site2
 */
void LBPGraph::set_neighbors(int site1, int site2)
{
    edges.push_back(DirectedEdge(site1, site2));
    vertices[site2].incoming_edges.push_back(edges.size() - 1);
    edges.push_back(DirectedEdge(site2, site1));
    vertices[site1].incoming_edges.push_back(edges.size() - 1);
}


void LBPGraph::set_data_costs(int label, std::vector<SparseDataCost> const & costs)
{
    for (std::size_t i = 0; i < costs.size(); ++i)
    {
        Vertex & vertex = vertices[costs[i].site];
        vertex.labels.push_back(label);
        ENERGY_TYPE data_cost = costs[i].cost;
        vertex.data_costs.push_back(data_cost);

//        ENERGY_TYPE smooth_cost = costs[i].smooth;
//        vertex.data_costRs.push_back(costs[i].smoothR);
//        vertex.data_costGs.push_back(costs[i].smoothG);
//        vertex.data_costBs.push_back(costs[i].smoothB);
        vertex.feature_costs.push_back(costs[i].detailvalue);

        if (data_cost < vertex.data_cost)
        {
            vertex.label = label;
            vertex.data_cost = data_cost;
//            vertex.smooth_costR = costs[i].smoothR;
//            vertex.smooth_costG = costs[i].smoothG;
//            vertex.smooth_costB = costs[i].smoothB;
            vertex.feature_cost = costs[i].detailvalue;
        }

        for (int j : vertex.incoming_edges)
        {
            DirectedEdge & incoming_edge = edges[j];
            incoming_edge.old_msg.push_back(0);
            incoming_edge.new_msg.push_back(0);
        }
    }
}

int LBPGraph::what_label(int site) {
    return vertices[site].label;
}

int LBPGraph::num_sites() {
    return static_cast<int>(vertices.size());
}

MRF_NAMESPACE_END
