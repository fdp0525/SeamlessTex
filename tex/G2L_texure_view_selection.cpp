/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */

#include "texturing.h"
#include "debug.h"
#include "mve/image_io.h"

TEX_NAMESPACE_BEGIN

void G2LTexSelection(mve::TriangleMesh::ConstPtr mesh, mve::MeshInfo const & mesh_info, std::vector<TextureView> * texture_views, UniGraph * graph , Settings const & settings)
{
    tex::build_adjacency_graph(mesh, mesh_info, graph);
    std::size_t const num_faces = mesh->get_faces().size() / 3;

    tex::DataCosts data_costs(num_faces, texture_views->size());//facesNum*ImageNum的稀疏矩阵
    tex::SmoothCosts smooth_costs(num_faces, texture_views->size());//facesNum*ImageNum的稀疏矩阵
    tex::DataCosts detail_costs(num_faces, texture_views->size());//facesNum*ImageNum的稀疏矩阵

    tex::calculate_data_costs_and_detail(mesh, texture_views, settings, &data_costs, &smooth_costs, &detail_costs);
    tex::view_selection_with_detail(data_costs, smooth_costs, detail_costs, graph, settings);


    tex::combineSmallPath(*graph, mesh, *texture_views, false);

    tex::combineSmallPath(*graph, mesh, *texture_views, false);
    tex::combineSmallPath(*graph, mesh, *texture_views, false);

    tex::combineSmallPath(*graph, mesh, *texture_views, true);
}


void G2LTexSelectionTwoPass(mve::TriangleMesh::ConstPtr mesh, mve::MeshInfo const & mesh_info, std::vector<TextureView> * texture_views, UniGraph * graph , Settings const & settings)
{

    tex::build_adjacency_graph(mesh, mesh_info, graph);//建立邻接面之间的连接图

    std::size_t const num_faces = mesh->get_faces().size() / 3;//每个面包含三个顶点

    tex::DataCosts data_costs(num_faces, texture_views->size());//facesNum*ImageNum的稀疏矩阵
    tex::calculate_data_costs(mesh, texture_views, settings, &data_costs);//计算目标函数的数据项
    tex::view_selection(data_costs, graph, settings);//选择最优的视口
}

void G2LsaveLabelModel(std::string name, Settings const & settings, UniGraph const & graph, mve::TriangleMesh::ConstPtr mesh, mve::MeshInfo const & mesh_info,
                    std::vector<tex::TextureView> texture_views, int passiter, bool debug)
{
    tex::TextureAtlases texture_atlases;

//    if (G2LTexConfig::get().write_view_selection_model)//默认为false
    {
        texture_atlases.clear();
        {
            tex::TexturePatches texture_patches;
            if(debug == true)
            {
                generate_debug_embeddings(&texture_views);//显示纹理分割的结果
            }
            tex::VertexProjectionInfos vertex_projection_infos; // Will only be written
            tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
                settings, &vertex_projection_infos, &texture_patches, passiter);
            tex::generate_texture_atlases(&texture_patches, settings, &texture_atlases);
        }

//        std::cout << "Building debug objmodel:" << std::endl;
        {
            tex::Model model;
            tex::build_model(mesh, texture_atlases, &model);
            std::cout << "\tSaving model... " << std::flush;
            tex::Model::save(model, name);
            std::cout << "done." << std::endl;
        }
    }
}

void load_seams_images(std::vector<TextureView> * texture_views)
{
    for(int i = 0;i<texture_views->size();i++)
    {
        TextureView * texture_view = &(texture_views->at(i));

        char buf[256];
        sprintf(buf,"contour%02d.png", i);
        mve::ByteImage::Ptr image = mve::image::load_file(buf);
        texture_view->bind_image(image);
    }

}

void G2LsaveLabelModelWithSeams(std::string name, Settings const & settings, UniGraph const & graph, mve::TriangleMesh::ConstPtr mesh, mve::MeshInfo const & mesh_info,
                    std::vector<tex::TextureView> texture_views)
{
    tex::TextureAtlases texture_atlases;

//    if (G2LTexConfig::get().write_view_selection_model)//默认为false
    {
        texture_atlases.clear();
        {
            tex::TexturePatches texture_patches;
//            generate_debug_embeddings(&texture_views);//显示纹理分割的结果
            load_seams_images(&texture_views);
            tex::VertexProjectionInfos vertex_projection_infos; // Will only be written
            tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
                settings, &vertex_projection_infos, &texture_patches);
            tex::generate_texture_atlases(&texture_patches, settings, &texture_atlases);
        }

//        std::cout << "Building debug objmodel:" << std::endl;
        {
            tex::Model model;
            tex::build_model(mesh, texture_atlases, &model);
            std::cout << "\tSaving model... " << std::flush;
            tex::Model::save(model, name);
            std::cout << "done." << std::endl;
        }
    }
}
TEX_NAMESPACE_END
