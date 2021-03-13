/*
 * Copyright (C) 2015, Nils Moehrle, Benjamin Richter
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MRF_ICMGRAPH_HEADER
#define MRF_ICMGRAPH_HEADER

#include "graph.h"

MRF_NAMESPACE_BEGIN

/** Implementation of the iterated conditional mode algorithm. */
class ICMGraph : public Graph {
    private:
    /**
         * @brief The Site struct  记录一个面的所有的投影信息
         */
        struct Site
        {
            int label;
            ENERGY_TYPE data_cost;
            ENERGY_TYPE smooth_costR;
            ENERGY_TYPE smooth_costG;
            ENERGY_TYPE smooth_costB;

            std::vector<int> labels;
            std::vector<ENERGY_TYPE> data_costs;
            std::vector<ENERGY_TYPE>   data_costRs;
            std::vector<ENERGY_TYPE>   data_costGs;
            std::vector<ENERGY_TYPE>   data_costBs;

            std::vector<int> neighbors;
            Site() : label(0), data_cost(std::numeric_limits<ENERGY_TYPE>::max()),
                smooth_costR(std::numeric_limits<ENERGY_TYPE>::max()),
                smooth_costG(std::numeric_limits<ENERGY_TYPE>::max()),
                smooth_costB(std::numeric_limits<ENERGY_TYPE>::max())
            {

            }
        };

        std::vector<Site> sites;//所有面的投影信息，site包含每个面的所有视口的投影信息
        SmoothCostFunction smooth_cost_func;
    public:
        ICMGraph(int num_sites, int num_labels);
        ENERGY_TYPE smooth_cost(int site, int label);

        void set_smooth_cost(SmoothCostFunction func);
        void set_data_costs(int label, std::vector<SparseDataCost> const & costs);
        void set_neighbors(int site1, int site2);
        ENERGY_TYPE compute_energy();
        ENERGY_TYPE optimize(int num_iterations);
        int what_label(int site);

        int num_sites();
};

MRF_NAMESPACE_END

#endif /* MRF_ICMGRAPH_HEADER */
