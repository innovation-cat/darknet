cmake_minimum_required(VERSION 3.13)

project(darknet LANGUAGES C CXX)

include(cmake/all.cmake)

set(SOURCE
${CMAKE_CURRENT_SOURCE_DIR}/src/gemm.c
${CMAKE_CURRENT_SOURCE_DIR}/src/utils.c
${CMAKE_CURRENT_SOURCE_DIR}/src/cuda.c
${CMAKE_CURRENT_SOURCE_DIR}/src/deconvolutional_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/convolutional_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/list.c
${CMAKE_CURRENT_SOURCE_DIR}/src/image.c
${CMAKE_CURRENT_SOURCE_DIR}/src/image_opencv.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/activations.c
${CMAKE_CURRENT_SOURCE_DIR}/src/im2col.c
${CMAKE_CURRENT_SOURCE_DIR}/src/col2im.c
${CMAKE_CURRENT_SOURCE_DIR}/src/blas.c
${CMAKE_CURRENT_SOURCE_DIR}/src/crop_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/dropout_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/maxpool_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/softmax_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/data.c
${CMAKE_CURRENT_SOURCE_DIR}/src/matrix.c
${CMAKE_CURRENT_SOURCE_DIR}/src/network.c
${CMAKE_CURRENT_SOURCE_DIR}/src/connected_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/cost_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/parser.c
${CMAKE_CURRENT_SOURCE_DIR}/src/option_list.c
${CMAKE_CURRENT_SOURCE_DIR}/src/detection_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/route_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/upsample_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/box.c
${CMAKE_CURRENT_SOURCE_DIR}/src/normalization_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/avgpool_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/local_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/shortcut_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/logistic_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/activation_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/rnn_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/gru_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/crnn_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/demo.c
${CMAKE_CURRENT_SOURCE_DIR}/src/batchnorm_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/region_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/reorg_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/tree.c
${CMAKE_CURRENT_SOURCE_DIR}/src/lstm_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/l2norm_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/yolo_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/src/iseg_layer.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/captcha.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/lsd.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/super.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/art.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/tag.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/cifar.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/go.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/rnn.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/segmenter.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/regressor.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/classifier.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/coco.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/yolo.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/detector.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/nightmare.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/instance-segmenter.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/darknet.c
${CMAKE_CURRENT_SOURCE_DIR}/examples/webank.c
  )

  if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")
  elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")
  endif()

add_executable(darknet ${SOURCE})
find_package(OpenCV REQUIRED)
target_compile_definitions(darknet PRIVATE OPENCV)
target_include_directories(darknet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(darknet PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(darknet PRIVATE ${OpenCV_LIBS})
target_link_libraries(darknet PRIVATE m)
target_link_libraries(darknet PRIVATE pthread)
