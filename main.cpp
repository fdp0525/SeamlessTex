/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */
#include <iostream>
#include <fstream>
#include <vector>

#include <util/timer.h>
#include <util/system.h>
#include <util/file_system.h>
#include <mve/mesh_io_ply.h>

#include "tex/util.h"
#include "tex/timer.h"
#include "tex/debug.h"
#include "tex/texturing.h"
#include "tex/progress_counter.h"
#include "mve/image_io.h"

//#include "arguments.h"
#include "paramArgs.h"
#include "G2LTexConfig.h"

void saveLabelModel(std::string name, paramArgs  conf, UniGraph const & graph,
                    mve::TriangleMesh::ConstPtr  mesh, mve::MeshInfo const & mesh_info,
                    std::vector<tex::TextureView> texture_views)
{
    tex::TextureAtlases texture_atlases;


    if (conf.write_view_selection_model)//默认为false
    {
        texture_atlases.clear();
        std::cout << "Generating debug texture patches:" << std::endl;
        {
            tex::TexturePatches texture_patches;
            generate_debug_embeddings(&texture_views);
            tex::VertexProjectionInfos vertex_projection_infos; // Will only be written
            tex::generate_texture_patches(graph, mesh, mesh_info, &texture_views,
                conf.settings, &vertex_projection_infos, &texture_patches);
            tex::generate_texture_atlases(&texture_patches, conf.settings, &texture_atlases);
        }

        std::cout << "Building debug objmodel:" << std::endl;
        {
            tex::Model model;
            tex::build_model(mesh, texture_atlases, &model);
            std::cout << "\tSaving model... " << std::flush;
            tex::Model::save(model, name);
            std::cout << "done." << std::endl;
        }
    }
}

int main(int argc, char **argv)
{
    paramArgs conf;

    util::WallTimer totaltimer;

    try
    {
        conf = parse_args(argc, argv);//read parameter from commod line, such as the name of the model and the directory of the color images.
    }
    catch (std::invalid_argument & ia)
    {
        std::cerr << ia.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    mve::TriangleMesh::Ptr mesh;
    try
    {
        mesh = mve::geom::load_ply_mesh(conf.in_mesh);//读取网格文件。
    }
    catch (std::exception& e)
    {
        std::cerr << "\tCould not load mesh: "<< e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    mve::MeshInfo mesh_info(mesh);
    tex::prepare_mesh(&mesh_info, mesh);

    std::cout << "Generating texture views: " << std::endl;
    tex::TextureViews     texture_views;
    tex::generate_texture_views(conf.in_scene, &texture_views, "tmp");//读取每个时口彩色图深度图以及相关的相机内外参。
    std::size_t const num_faces = mesh->get_faces().size() / 3;

    tex::Graph graph(num_faces);

    for(int i = 0; i<texture_views.size(); i++)
    {
        tex::TextureView &view = texture_views[i];
        view.generateFeaturemap(view.image_file);
        view.myGenerateDetailImage(view.image_file, i);
    }


    util::WallTimer chargentimer;

    bool  readlablefile = false;
    if(readlablefile == false)
    {
        util::WallTimer timer;
        tex::G2LTexSelection(mesh, mesh_info, &texture_views, &graph, conf.settings);
        std::cout << "chart generation:" << timer.get_elapsed() << " ms" << std::endl;

        std::vector<std::size_t> labeling(graph.num_nodes());
//        std::vector<std::size_t> twopasslabeling(graph.num_nodes());
        for (std::size_t i = 0; i < graph.num_nodes(); ++i)
        {
            labeling[i] = graph.get_label(i);
//            twopasslabeling[i] = graph.get_twoPassLabel(i);
        }
        vector_to_file( "labeling.vec", labeling);
//        vector_to_file( "twopasslabeling.vec", twopasslabeling);
    }
    else
    {
        tex::build_adjacency_graph(mesh, mesh_info, &graph);

        std::cout<<" read label from file"<<std::endl;
        std::vector<std::size_t> labeling = vector_from_file<std::size_t>("labeling.vec");
        if (labeling.size() != graph.num_nodes())
        {
            std::cerr << "Wrong labeling file for this mesh/scene combination... aborting!" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        for (std::size_t i = 0; i < labeling.size(); ++i)
        {
            const std::size_t label = labeling[i];
            if (label > texture_views.size())
            {
                std::cerr << "Wrong labeling file for this mesh/scene combination... aborting!" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            graph.set_label(i, label);
        }

    }

    std::cout << "chart gen:" << chargentimer.get_elapsed() << " ms" << std::endl;

    std::vector<tex::myFaceInfo>               faceInfoList;
    std::vector<tex::myViewImageInfo>       viewImageList;
    std::vector<std::vector<std::size_t> >   chart_graph;
    std::vector<std::vector<std::size_t> >   subchartgraphs;
    tex::G2LsaveLabelModel ("nooption", conf.settings, graph, mesh, mesh_info, texture_views, 1, false);

    util::WallTimer posestimer;
    std::vector<int>  labelPatchCount;
    tex::combineOptionCameraPosesWithDetail(graph, mesh, mesh_info, texture_views, conf.in_scene, faceInfoList, viewImageList, chart_graph, subchartgraphs, labelPatchCount, conf.settings);
    std::cout << "camera poses optimization:" << posestimer.get_elapsed() << " ms" << std::endl;

    tex::G2LsaveLabelModel ("noseamless", conf.settings, graph, mesh, mesh_info, texture_views, 1, false);

    tex::VertexProjectionInfos    in_vertex_projection_infos;
    tex::VertexProjectionInfos    edge_vertex_infos;
    tex::generate_chart_vertexInfo(graph, mesh, mesh_info, in_vertex_projection_infos,
                                   edge_vertex_infos, faceInfoList, subchartgraphs);
    util::WallTimer seamlesstimer;
    tex::texturepatchseamless(graph, mesh, &texture_views, mesh_info, faceInfoList, subchartgraphs,
                              in_vertex_projection_infos, edge_vertex_infos, conf.in_scene);
    std::cout << "seams optimization:" << seamlesstimer.get_elapsed() << " ms" << std::endl;
    tex::G2LsaveLabelModel ("noharmcosis", conf.settings, graph, mesh, mesh_info, texture_views, 1, false);


    util::WallTimer harmontimer;
    tex::color_harmonization_with_detail(graph, mesh, texture_views, subchartgraphs, labelPatchCount, chart_graph, viewImageList);
    std::cout << "color harmonization:" << harmontimer.get_elapsed() << " ms" << std::endl;

    tex::G2LsaveLabelModel ("result", conf.settings, graph, mesh, mesh_info, texture_views, 1, false);

    std::cout << "total time:" << totaltimer.get_elapsed() << " ms" << std::endl;

    return EXIT_SUCCESS;
}
