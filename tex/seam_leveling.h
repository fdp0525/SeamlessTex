/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_SEAMLEVELING_HEADER
#define TEX_SEAMLEVELING_HEADER

#include <vector>

#include <mve/mesh.h>

#include "defines.h"
#include "uni_graph.h"

TEX_NAMESPACE_BEGIN

//每个顶点的投影信息
struct VertexProjectionInfo
{
    std::size_t        texture_patch_id;//所属纹理块的索引
    std::size_t        vertex_id;//顶点的索引
    math::Vec2f      projection;//点投影纹理坐标
    std::vector<std::size_t>  faces;//所有共享这个顶点的面索引(邻接面)
    bool   isChartBoundaryV;//是否是chart边界上的点
    std::vector<std::size_t>  adj_charts;//当前顶点的所有邻接chart。
    std::vector<std::size_t>  edge_verts;//当前顶点在当前chart的两条边连接的两个顶点。
    std::size_t      v_label;//当前顶点所在chart的labal，只考虑chart的时候可用
//    std::size_t      v_twopass_label;//当前顶点在twopasschart上的label

//    std::size_t      adj_chart_ID;//当前顶点相邻的chart


    bool   isModelBoundaryVer;//是否是chart边界上的点


    bool operator<(VertexProjectionInfo const & other) const
    {
        return texture_patch_id < other.texture_patch_id;
    }

    VertexProjectionInfo()
    {

    }
    VertexProjectionInfo(std::size_t t_id,
                         math::Vec2f p,
                         std::vector<std::size_t>   f)
    {
        texture_patch_id = t_id;
        projection(0) = p(0);  projection(1) = p(1);
        faces = f;
        v_label = 0;
    }


    VertexProjectionInfo(std::size_t t_id,//texture id
                         std::size_t v_id,//vertex id
                         math::Vec2f p,//projection pixel
                         std::vector<std::size_t>   f,//adj faces
                         bool   isCBoundary,
                         std::vector<std::size_t>  charts,
                         std::vector<std::size_t>  verts,
                         std::size_t  label,
                         bool iscb)
    {
        texture_patch_id = t_id;
        vertex_id = v_id;
        projection(0) = p(0);  projection(1) = p(1);
        faces = f;

        isChartBoundaryV = isCBoundary;
        adj_charts = charts;
        edge_verts = verts;
        v_label = label;
//        v_twopass_label = tp_label;
        isModelBoundaryVer = iscb;
    }
};

struct EdgeProjectionInfo {
    std::size_t texture_patch_id;
    math::Vec2f p1;
    math::Vec2f p2;

    bool operator<(EdgeProjectionInfo const & other) const {
        return texture_patch_id < other.texture_patch_id;
    }
};

struct MeshEdge {
    std::size_t v1;
    std::size_t v2;
};

void
find_seam_edges(UniGraph const & graph, mve::TriangleMesh::ConstPtr mesh,
    std::vector<MeshEdge> * seam_edges);

void
find_mesh_edge_projections(
    std::vector<std::vector<VertexProjectionInfo> > const & vertex_projection_infos,
    MeshEdge mesh_edge, std::vector<EdgeProjectionInfo> * projected_edge_infos);

TEX_NAMESPACE_END

#endif /* TEX_SEAMLEVELING_HEADER */
