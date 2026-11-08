cmake_minimum_required(VERSION 3.16)
project(PoolBalls CXX)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR "$ENV{HOME}/opt/opencv-4.12.0/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

add_executable(pool myApplication.cpp mac/main.cpp)
target_include_directories(pool PRIVATE OpenCVExample)
target_link_libraries(pool PRIVATE ${OpenCV_LIBS})
target_compile_definitions(pool PRIVATE TIPS_DIR="${CMAKE_SOURCE_DIR}/OpenCVExample")
