DESTDIR += ./bin
OBJECTS_DIR += ./tmp
QMAKE_CXXFLAGS += -std=gnu++11
INCLUDEPATH += ./rayint ./mapmap ./patchmatch ./detail

QMAKE_CXXFLAGS += -fopenmp -O2
#QMAKE_LFLAGS +=  -fopenmp


HEADERS += \
    detail/array.h \
    detail/channel_image.h \
    detail/convolution_3D.h \
    detail/fft_3D.h \
    detail/fill_3D.h \
    detail/linear_bf.h \
    detail/load_EXR.h \
    detail/math_tools.h \
    detail/msg_stream.h \
    detail/support_3D.h \
    mapmap/header/cost_instances/pairwise_antipotts.h \
    mapmap/header/cost_instances/pairwise_linear_peak.h \
    mapmap/header/cost_instances/pairwise_potts.h \
    mapmap/header/cost_instances/pairwise_table.h \
    mapmap/header/cost_instances/pairwise_truncated_linear.h \
    mapmap/header/cost_instances/pairwise_truncated_quadratic.h \
    mapmap/header/cost_instances/unary_table.h \
    mapmap/header/multilevel_instances/group_same_label.h \
    mapmap/header/optimizer_instances/envelope_instances/pairwise_antipotts_envelope.h \
    mapmap/header/optimizer_instances/envelope_instances/pairwise_linear_peak_envelope.h \
    mapmap/header/optimizer_instances/envelope_instances/pairwise_potts_envelope.h \
    mapmap/header/optimizer_instances/envelope_instances/pairwise_truncated_linear_envelope.h \
    mapmap/header/optimizer_instances/envelope_instances/pairwise_truncated_quadratic_envelope.h \
    mapmap/header/optimizer_instances/dp_node.h \
    mapmap/header/optimizer_instances/dp_node_solver.h \
    mapmap/header/optimizer_instances/dp_node_solver_factory.h \
    mapmap/header/optimizer_instances/dynamic_programming.h \
    mapmap/header/optimizer_instances/envelope.h \
    mapmap/header/termination_instances/stop_after_iterations.h \
    mapmap/header/termination_instances/stop_after_time.h \
    mapmap/header/termination_instances/stop_when_flat.h \
    mapmap/header/termination_instances/stop_when_returns_diminish.h \
    mapmap/header/tree_sampler_instances/lock_free_tree_sampler.h \
    mapmap/header/tree_sampler_instances/optimistic_tree_sampler.h \
    mapmap/header/color.h \
    mapmap/header/cost_bundle.h \
    mapmap/header/costs.h \
    mapmap/header/defines.h \
    mapmap/header/dynamic_programming.h \
    mapmap/header/graph.h \
    mapmap/header/instance_factory.h \
    mapmap/header/mapmap.h \
    mapmap/header/multilevel.h \
    mapmap/header/parallel_templates.h \
    mapmap/header/termination_criterion.h \
    mapmap/header/timer.h \
    mapmap/header/tree.h \
    mapmap/header/tree_optimizer.h \
    mapmap/header/tree_sampler.h \
    mapmap/header/vector_math.h \
    mapmap/header/vector_types.h \
    mapmap/source/cost_instances/pairwise_antipotts.impl.h \
    mapmap/source/cost_instances/pairwise_linear_peak.impl.h \
    mapmap/source/cost_instances/pairwise_potts.impl.h \
    mapmap/source/cost_instances/pairwise_table.impl.h \
    mapmap/source/cost_instances/pairwise_truncated_linear.impl.h \
    mapmap/source/cost_instances/pairwise_truncated_quadratic.impl.h \
    mapmap/source/cost_instances/unary_table.impl.h \
    mapmap/source/multilevel_instances/group_same_label.impl.h \
    mapmap/source/optimizer_instances/envelope_instances/pairwise_antipotts_envelope.impl.h \
    mapmap/source/optimizer_instances/envelope_instances/pairwise_linear_peak_envelope.impl.h \
    mapmap/source/optimizer_instances/envelope_instances/pairwise_potts_envelope.impl.h \
    mapmap/source/optimizer_instances/envelope_instances/pairwise_truncated_linear_envelope.impl.h \
    mapmap/source/optimizer_instances/envelope_instances/pairwise_truncated_quadratic_envelope.impl.h \
    mapmap/source/optimizer_instances/dp_node_solver.impl.h \
    mapmap/source/optimizer_instances/dp_node_solver_factory.impl.h \
    mapmap/source/optimizer_instances/dynamic_programming.impl.h \
    mapmap/source/termination_instances/stop_after_iterations.impl.h \
    mapmap/source/termination_instances/stop_after_time.impl.h \
    mapmap/source/termination_instances/stop_when_flat.impl.h \
    mapmap/source/termination_instances/stop_when_returns_diminish.impl.h \
    mapmap/source/tree_sampler_instances/lock_free_tree_sampler.impl.h \
    mapmap/source/tree_sampler_instances/optimistic_tree_sampler.impl.h \
    mapmap/source/color.impl.h \
    mapmap/source/cost_bundle.impl.h \
    mapmap/source/costs.impl.h \
    mapmap/source/dynamic_programming.impl.h \
    mapmap/source/graph.impl.h \
    mapmap/source/instance_factory.impl.h \
    mapmap/source/mapmap.impl.h \
    mapmap/source/multilevel.impl.h \
    mapmap/source/parallel_templates.impl.h \
    mapmap/source/termination_criterion.impl.h \
    mapmap/source/timer.impl.h \
    mapmap/source/tree.impl.h \
    mapmap/source/tree_optimizer.impl.h \
    mapmap/source/tree_sampler.impl.h \
    mapmap/source/vector_math.impl.h \
    mapmap/dset.h \
    mapmap/full.h \
    math/accum.h \
    math/algo.h \
    math/bezier_curve.h \
    math/bspline.h \
    math/defines.h \
    math/functions.h \
    math/geometry.h \
    math/line.h \
    math/matrix.h \
    math/matrix_qr.h \
    math/matrix_svd.h \
    math/matrix_tools.h \
    math/octree_tools.h \
    math/permute.h \
    math/plane.h \
    math/quaternion.h \
    math/transform.h \
    math/vector.h \
    mrf/gco_graph.h \
    mrf/graph.h \
    mrf/icm_graph.h \
    mrf/lbp_graph.h \
    mve/bundle.h \
    mve/bundle_io.h \
    mve/camera.h \
    mve/defines.h \
    mve/depthmap.h \
    mve/image.h \
    mve/image_base.h \
    mve/image_color.h \
    mve/image_drawing.h \
    mve/image_exif.h \
    mve/image_io.h \
    mve/image_tools.h \
    mve/marching_cubes.h \
    mve/marching_tets.h \
    mve/mesh.h \
    mve/mesh_info.h \
    mve/mesh_io.h \
    mve/mesh_io_npts.h \
    mve/mesh_io_obj.h \
    mve/mesh_io_off.h \
    mve/mesh_io_pbrt.h \
    mve/mesh_io_ply.h \
    mve/mesh_io_smf.h \
    mve/mesh_tools.h \
    mve/scene.h \
    mve/view.h \
    mve/volume.h \
    patchmatch/allegro_emu.h \
    patchmatch/knn.h \
    patchmatch/nn.h \
    patchmatch/patch.h \
    patchmatch/patchmatch.h \
    patchmatch/simnn.h \
    patchmatch/simpatch.h \
    patchmatch/vecnn.h \
    patchmatch/vecpatch.h \
    rayint/acc/acceleration.h \
    rayint/acc/bvh_tree.h \
    rayint/acc/defines.h \
    rayint/acc/kd_tree.h \
    rayint/acc/primitives.h \
    rayint/math/algo.h \
    rayint/math/defines.h \
    rayint/math/vector.h \
    tex/debug.h \
    tex/defines.h \
    tex/histogram.h \
    tex/material_lib.h \
    tex/obj_model.h \
    tex/poisson_blending.h \
    tex/progress_counter.h \
    tex/rect.h \
    tex/rectangular_bin.h \
    tex/seam_leveling.h \
    tex/settings.h \
    tex/sparse_table.h \
    tex/texture_atlas.h \
    tex/texture_patch.h \
    tex/texture_view.h \
    tex/texturing.h \
    tex/timer.h \
    tex/tri.h \
    tex/uni_graph.h \
    tex/util.h \
    util/aligned_allocator.h \
    util/aligned_memory.h \
    util/arguments.h \
    util/defines.h \
    util/exception.h \
    util/file_system.h \
    util/frame_timer.h \
    util/ini_parser.h \
    util/logging.h \
    util/strings.h \
    util/system.h \
    util/timer.h \
    util/tokenizer.h \
    edge_node.h \
    G2LTexConfig.h \
    ll.h \
    paramArgs.h \
    texture_synthetise.h \
    patchmatch/patchmacthgpu.h

SOURCES += \
    detail/support_3D.cpp \
    mrf/gco_graph.cpp \
    mrf/graph.cpp \
    mrf/icm_graph.cpp \
    mrf/lbp_graph.cpp \
    mve/bundle.cc \
    mve/bundle_io.cc \
    mve/camera.cc \
    mve/depthmap.cc \
    mve/image_exif.cc \
    mve/image_io.cc \
    mve/image_tools.cc \
    mve/marching.cc \
    mve/mesh.cc \
    mve/mesh_info.cc \
    mve/mesh_io.cc \
    mve/mesh_io_npts.cc \
    mve/mesh_io_obj.cc \
    mve/mesh_io_off.cc \
    mve/mesh_io_pbrt.cc \
    mve/mesh_io_ply.cc \
    mve/mesh_io_smf.cc \
    mve/mesh_tools.cc \
    mve/scene.cc \
    mve/view.cc \
    mve/volume.cc \
    patchmatch/allegro_emu.cpp \
    patchmatch/knn.cpp \
    patchmatch/nn.cpp \
    patchmatch/patch.cpp \
    patchmatch/patchmatch.cpp \
    patchmatch/simnn.cpp \
    patchmatch/vecnn.cpp \
    tex/build_adjacency_graph.cpp \
    tex/build_obj_model.cpp \
    tex/build_patch_adjacency_graph.cpp \
    tex/calculate_data_costs.cpp \
    tex/camera_pose_option.cpp \
    tex/color_harmonization.cpp \
    tex/combineoption_cameraposes.cpp \
    tex/G2L_texure_view_selection.cpp \
    tex/generate_debug_embeddings.cpp \
    tex/generate_texture_atlases.cpp \
    tex/generate_texture_patches.cpp \
    tex/generate_texture_views.cpp \
    tex/global_seam_leveling.cpp \
    tex/histogram.cpp \
    tex/local_seam_leveling.cpp \
    tex/material_lib.cpp \
    tex/obj_model.cpp \
    tex/poisson_blending.cpp \
    tex/prepare_mesh.cpp \
    tex/rectangular_bin.cpp \
    tex/seam_leveling.cpp \
    tex/texture_atlas.cpp \
    tex/texture_patch.cpp \
    tex/texture_patch_seam.cpp \
    tex/texture_view.cpp \
    tex/texturetools.cpp \
    tex/timer.cpp \
    tex/tri.cpp \
    tex/uni_graph.cpp \
    tex/view_selection.cpp \
    util/arguments.cc \
    util/file_system.cc \
    util/ini_parser.cc \
    util/system.cc \
    main.cpp \
    paramArgs.cpp \
    texture_synthetise.cpp



INCLUDEPATH += /usr/include/eigen3 /usr/local/include
INCLUDEPATH += /usr/local/include/opencv
LIBS +=  -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videostab

LIBS += -ltiff -ltbb -lgomp -lceres
LIBS += -lglog

INCLUDEPATH += /usr/include/OpenEXR

LIBS += -lfftw3 -lIlmImf -lHalf


DISTFILES += \
    patchmatch/patchmacthgpu.cu

CUDA_SOURCES += \
    patchmatch/patchmacthgpu.cu

INCLUDEPATH += /usr/local/cuda-10.1/include

#LIBS += -L/usr/local/cuda-9.0/lib64 -lcudart

CUDA_SDK = "/usr/local/cuda-10.1"   # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda-10.1"            # Path to cuda toolkit install

# DO NOT EDIT BEYOND THIS UNLESS YOU KNOW WHAT YOU ARE DOING....

SYSTEM_NAME = unix         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_70           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math
#NVCCXCOMPLIER += -fPIC


# include paths
INCLUDEPATH += $$CUDA_DIR/include

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64

CUDA_OBJECTS_DIR = ./tmp


# Add the necessary libraries
CUDA_LIBS =  -lcudart

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#LIBS += $$join(CUDA_LIBS,'.so ', '', '.so')
LIBS += $$CUDA_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --compiler-options '-fPIC' --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --compiler-options '-fPIC' --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    QMAKE_EXTRA_COMPILERS += cuda
}
