/*
 * Copyright (C) 2015, Nils Moehrle, Benjamin Richter
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MRF_LBPGRAPH_HEADER
#define MRF_LBPGRAPH_HEADER

#include "graph.h"

MRF_NAMESPACE_BEGIN

/** Implementation of the loopy belief propagation algorithm. */
class LBPGraph : public Graph {
    private:
        struct DirectedEdge {
            int v1;
            int v2;
            std::vector<ENERGY_TYPE> new_msg;//取不同的label与v1相连的边的代价函数和。
            std::vector<ENERGY_TYPE> old_msg;
            DirectedEdge(int v1, int v2) : v1(v1), v2(v2) {}
        };

        struct Vertex
        {
            int label;
            ENERGY_TYPE data_cost;
//            ENERGY_TYPE smooth_costR;
//            ENERGY_TYPE smooth_costG;
//            ENERGY_TYPE smooth_costB;
            ENERGY_TYPE feature_cost;

            std::vector<int> labels;
            std::vector<ENERGY_TYPE>   data_costs;
//            std::vector<ENERGY_TYPE>   data_costRs;
//            std::vector<ENERGY_TYPE>   data_costGs;
//            std::vector<ENERGY_TYPE>   data_costBs;
            std::vector<ENERGY_TYPE>   feature_costs;

            std::vector<ENERGY_TYPE>   gradient_costs;
            std::vector<int> incoming_edges;
            Vertex() : label(0), data_cost(std::numeric_limits<ENERGY_TYPE>::max()),
                feature_cost(std::numeric_limits<ENERGY_TYPE>::max())
                {}
        };


        std::vector<DirectedEdge> edges;
        std::vector<Vertex> vertices;
        SmoothCostFunction smooth_cost_func;
    public:
        LBPGraph(int num_sites, int num_labels);

        void set_smooth_cost(SmoothCostFunction func);
        void set_data_costs(int label, std::vector<SparseDataCost> const & costs);
        void set_neighbors(int site1, int site2);
        ENERGY_TYPE compute_energy();
        ENERGY_TYPE optimize(int num_iterations);
        int what_label(int site);

        int num_sites();//相同标签相邻面的聚类个数
};

MRF_NAMESPACE_END

#endif /* MRF_LBPGRAPH_HEADER */
