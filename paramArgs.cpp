/**
 * @brief
 * @author ypfu@whu.edu.cn
 * @date
 */

#include "paramArgs.h"
#include "util/file_system.h"
#include <iostream>
#include <opencv2/opencv.hpp>


#define SKIP_GLOBAL_SEAM_LEVELING "skip_global_seam_leveling"
#define SKIP_GEOMETRIC_VISIBILITY_TEST "skip_geometric_visibility_test"
#define SKIP_LOCAL_SEAM_LEVELING "skip_local_seam_leveling"
#define NO_INTERMEDIATE_RESULTS "no_intermediate_results"
#define WRITE_TIMINGS "write_timings"
#define SKIP_HOLE_FILLING "skip_hole_filling"
#define KEEP_UNSEEN_FACES "keep_unseen_faces"

std::string bool_to_string(bool b)
{
    return b ? "True" : "False";
}

std::string paramArgs::to_string()
{
    std::stringstream out;
    out << "Input Images: \t" << in_scene << std::endl
        << "Input mesh: \t" << in_mesh << std::endl;

    return out.str();
}

//void paramArgs::init(int argc, char **argv)
//{
//    util::Arguments args;
//    args.set_exit_on_error(true);
//    args.set_nonopt_maxnum(2);
//    args.set_nonopt_minnum(2);
//    args.set_helptext_indent(34);
//    args.set_description("Global to Local Texture Optimization for 3D Reconstruction with RGB-D Sensor.");
//    args.set_usage("Usage: " + std::string(argv[0]) + " [options] IN_IMAGES IN_MESH");


//    //解析输入参数
//    args.parse(argc, argv);

//    //场景
//    in_scene = args.get_nth_nonopt(0);//第一个非可选参数
//    //网格模型（ply）
//    in_mesh = args.get_nth_nonopt(1);//第二个非可选参数
//    //输出文件夹
//    out_prefix = "./result";//第二个非可选参数

//    /* Set defaults for optional arguments. */
//    data_cost_file = "";
//    labeling_file = "";

//    write_timings = false;
//    write_view_selection_model = true;

//    cv::FileStorage  fs2("data.yml", cv::FileStorage::READ);

////depth
//    DEPTH_FX = (float)fs2["depth_fx"];
//    DEPTH_FY = (float)fs2["depth_fy"];
//    DEPTH_CX = (float)fs2["depth_cx"];
//    DEPTH_CY = (float)fs2["depth_cy"];
//    DEPTH_WIDTH = (int)fs2["depth_width"];
//    DEPTH_HEIGHT = (int)fs2["depth_height"];

//    //rgb
//    DEPTH_FX = (float)fs2["rgb_fx"];
//    DEPTH_FY = (float)fs2["rgb_fy"];
//    DEPTH_CX = (float)fs2["rgb_cx"];
//    DEPTH_CY = (float)fs2["rgb_cy"];
//    DEPTH_WIDTH = (int)fs2["rgb_width"];
//    DEPTH_HEIGHT = (int)fs2["rgb_height"];

//    GLOBAL_ITER_NUM = (int)fs2["Global_Iterations"];
//    LOCAL_ITER_NUM = (int)fs2["Lobal_Iterations"];

//    int wf = (int)fs2["write_view_selection_model"];
//    int gc = (int)fs2["global_color_correcting"];
//    if(wf == 1)
//    {
//        write_view_selection_model = true;
//    }
//    else
//    {
//        write_view_selection_model = false;
//    }

//    if(gc == 1)
//    {
//        global_color_correcting = true;
//    }
//    else
//    {
//        global_color_correcting = false;
//    }

//    MIN_CHART_NUM = 50;//最小的有效块的大小

//    SEARCH_OFFSET = 10;
//    BOARD_IGNORE = 20;
//}

paramArgs parse_args(int argc, char **argv)
{

    util::Arguments args;
    args.set_exit_on_error(true);
    args.set_nonopt_maxnum(2);
    args.set_nonopt_minnum(2);
    args.set_helptext_indent(34);
    args.set_description("Textures a mesh given images in form of a 3D scene.");
    args.set_description("Global to Local Texture Optimization for 3D Reconstruction with RGB-D Sensor.");
    args.set_usage("Usage: " + std::string(argv[0]) + " [options] IN_IMAGES IN_MESH");

    args.add_option('D',"data_cost_file", true,
                    "Skip calculation of data costs and use the ones provided in the given file");
    args.add_option('L',"labeling_file", true,
                    "Skip view selection and use the labeling provided in the given file");
    args.add_option('d',"data_term", true,
                    "Data term: {" +
                    choices<tex::DataTerm>() + "} [" +
                    choice_string<tex::DataTerm>(tex::DATA_TERM_GMI) + "]");
    args.add_option('s',"smoothness_term", true,
                    "Smoothness term: {" +
                    choices<tex::SmoothnessTerm>() + "} [" +
                    choice_string<tex::SmoothnessTerm>(tex::SMOOTHNESS_TERM_POTTS) + "]");
    args.add_option('o',"outlier_removal", true,
                    "Photometric outlier (pedestrians etc.) removal method: {" +
                    choices<tex::OutlierRemoval>() +  "} [" +
                    choice_string<tex::OutlierRemoval>(tex::OUTLIER_REMOVAL_NONE) + "]");
    args.add_option('t',"tone_mapping", true,
                    "Tone mapping method: {" +
                    choices<tex::ToneMapping>() +  "} [" +
                    choice_string<tex::ToneMapping>(tex::TONE_MAPPING_NONE) + "]");
    args.add_option('v',"view_selection_model", false,
                    "Write out view selection model [false]");
    args.add_option('\0', SKIP_GEOMETRIC_VISIBILITY_TEST, false,
                    "Skip geometric visibility test based on ray intersection [false]");
    args.add_option('\0', SKIP_GLOBAL_SEAM_LEVELING, false,
                    "Skip global seam leveling [false]");
    args.add_option('\0', SKIP_LOCAL_SEAM_LEVELING, false,
                    "Skip local seam leveling (Poisson editing) [false]");
    args.add_option('\0', SKIP_HOLE_FILLING, false,
                    "Skip hole filling [false]");
    args.add_option('\0', KEEP_UNSEEN_FACES, false,
                    "Keep unseen faces [false]");
    args.add_option('\0', WRITE_TIMINGS, false,
                    "Write out timings for each algorithm step (OUT_PREFIX + _timings.csv)");
    args.add_option('\0', NO_INTERMEDIATE_RESULTS, false,
                    "Do not write out intermediate results");

    //解析输入参数
    args.parse(argc, argv);

    paramArgs conf;
    //场景
    conf.in_scene = args.get_nth_nonopt(0);//第一个非可选参数
    //网格模型（ply）
    conf.in_mesh = args.get_nth_nonopt(1);//第二个非可选参数
    //输出文件夹
    conf.out_prefix = "./";//第二个非可选参数

    /* Set defaults for optional arguments. */
    conf.data_cost_file = "";
    conf.labeling_file = "";

    conf.write_timings = false;
    conf.write_intermediate_results = true;
    conf.write_view_selection_model = true;

    /* Handle optional arguments. */
    for (util::ArgResult const* i = args.next_option(); i != 0; i = args.next_option())
    {
        switch (i->opt->sopt) {
//        case 'v':
//            conf.write_view_selection_model = true;
//            break;
        case 'D':
            conf.data_cost_file = i->arg;
            break;
        case 'L':
            conf.labeling_file = i->arg;
            break;
        case 'd':
            conf.settings.data_term = parse_choice<tex::DataTerm>(i->arg);
            break;
        case 's':
            conf.settings.smoothness_term = parse_choice<tex::SmoothnessTerm>(i->arg);
            break;
        case 'o':
            conf.settings.outlier_removal = parse_choice<tex::OutlierRemoval>(i->arg);
            break;
        case 't':
            conf.settings.tone_mapping = parse_choice<tex::ToneMapping>(i->arg);
            break;
        case '\0':
            if (i->opt->lopt == SKIP_GEOMETRIC_VISIBILITY_TEST)
            {
                conf.settings.geometric_visibility_test = false;
            }
            else if (i->opt->lopt == SKIP_GLOBAL_SEAM_LEVELING)
            {
                conf.settings.global_seam_leveling = false;
            }
            else if (i->opt->lopt == SKIP_LOCAL_SEAM_LEVELING)
            {
                conf.settings.local_seam_leveling = false;
            }
            else if (i->opt->lopt == SKIP_HOLE_FILLING)
            {
                conf.settings.hole_filling = false;
            }
            else if (i->opt->lopt == KEEP_UNSEEN_FACES)
            {
                conf.settings.keep_unseen_faces = true;
            }
            else if (i->opt->lopt == WRITE_TIMINGS)
            {
                conf.write_timings = true;
            }
//            else if (i->opt->lopt == NO_INTERMEDIATE_RESULTS)
//            {
//                conf.write_intermediate_results = false;
//            }
            else
            {
                throw std::invalid_argument("Invalid long option");
            }
            break;
        default:
            throw std::invalid_argument("Invalid short option");
        }
    }
    return conf;
}




