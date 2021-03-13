/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <limits>
#include <list>

#include "uni_graph.h"

UniGraph::UniGraph(std::size_t nodes)
{
    adj_lists.resize(nodes);//每个面的邻接面
    labels.resize(nodes);//
    twoPassLabels.resize(nodes);
    edges = 0;
}

/**
 * @brief UniGraph::get_subgraphs   所有相邻的面具有相同的标签的面放入同一个队列
 * @param label
 * @param subgraphs
 */
void UniGraph::get_subgraphs(std::size_t label, std::vector<std::vector<std::size_t> > * subgraphs) const
{

    std::vector<bool>   used(adj_lists.size(), false);//所有效有面

    for(std::size_t i = 0; i < adj_lists.size(); ++i) //遍历所有的面
    {
        if (labels[i] == label && !used[i]) //与给定的标签相同并且没有聚类
        {
            subgraphs->push_back(std::vector<std::size_t>());//每个剖分聚类建立一个列表

            std::list<std::size_t> queue;

            queue.push_back(i);
            used[i] = true;//标记为已经使用

            while (!queue.empty())
            {
                std::size_t node = queue.front();
                queue.pop_front();

                subgraphs->back().push_back(node);//把节点放入队列

                /* Add all unused neighbours with the same label to the queue. */
                std::vector<std::size_t> const & adj_list = adj_lists[node];
                for(std::size_t j = 0; j < adj_list.size(); ++j)
                {
                    std::size_t adj_node = adj_list[j];
                    assert(adj_node < labels.size() && adj_node < used.size());//label每个面的标签

                    if (labels[adj_node] == label && !used[adj_node])//相邻的相同的标签面都放入同一个队列
                    {
                        queue.push_back(adj_node);
                        used[adj_node] = true;
                    }
                }
            }
        }
    }
}

void UniGraph::get_twoPassSubgraphs(std::size_t label, std::vector<std::vector<std::size_t> > * subgraphs) const
{
    std::vector<bool>   used(adj_lists.size(), false);//所有效有面

    for(std::size_t i = 0; i < adj_lists.size(); ++i) //遍历所有的面
    {
        if (twoPassLabels[i] == label && !used[i]) //与给定的标签相同并且没有聚类
        {
            subgraphs->push_back(std::vector<std::size_t>());//每个剖分聚类建立一个列表

            std::list<std::size_t> queue;

            queue.push_back(i);
            used[i] = true;//标记为已经使用

            while (!queue.empty())
            {
                std::size_t node = queue.front();
                queue.pop_front();

                subgraphs->back().push_back(node);//把节点放入队列

                /* Add all unused neighbours with the same label to the queue. */
                std::vector<std::size_t> const & adj_list = adj_lists[node];
                for(std::size_t j = 0; j < adj_list.size(); ++j)
                {
                    std::size_t adj_node = adj_list[j];
                    assert(adj_node < twoPassLabels.size() && adj_node < used.size());//label每个面的标签

                    if (twoPassLabels[adj_node] == label && !used[adj_node])//相邻的相同的标签面都放入同一个队列
                    {
                        queue.push_back(adj_node);
                        used[adj_node] = true;
                    }
                }
            }
        }
    }
}
