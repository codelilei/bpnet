cmake_minimum_required(VERSION 3.4)

project(Bp C CXX)

set(CMAKE_BUILD_TYPE "Release")

option(USE_OPENCV  "Use opencv to draw image" OFF)

set(BP_INC_DIR ${PROJECT_SOURCE_DIR}/inc)
set(BP_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(BP_LINKER_LIBS "")

include_directories(${BP_INC_DIR})
aux_source_directory(${BP_SRC_DIR} BP_SRCS)
add_executable(bptest ${BP_SRCS})

if(USE_OPENCV)
  find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
  if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
  endif()
  list(APPEND BP_INC_DIR PUBLIC ${OpenCV_INCLUDE_DIRS})
  list(APPEND BP_LINKER_LIBS PUBLIC ${OpenCV_LIBS})
  target_compile_definitions(bptest PUBLIC -DUSE_OPENCV)
  message(STATUS "OpenCV found (${OpenCV_CONFIG_PATH})")
endif()

target_link_libraries(bptest ${BP_LINKER_LIBS})
