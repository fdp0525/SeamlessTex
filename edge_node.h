#ifndef EDGE_NODE_H
#define EDGE_NODE_H
#include <iostream>

struct edgenodelist
{
    int    vertex_id;//当前边界点的索引
    int    chart_id;//当前边界点所属的chart
    int    chart_label;//当前边界点所属chart的label
    int  pre_vertex_id;
    int  next_vertex_id;
    edgenodelist *next;//当前边界点的下一个邻接边界点
    edgenodelist *pre;//当前边界点的下一个邻接边界点
};

#endif // EDGE_NODE_H
