cmake_minimum_required(VERSION 3.22)
project(creative-coding)

find_package(OpenMP REQUIRED)

set(MPG123_INCLUDE_DIR /usr/include/x86_64-linux-gnu)

# Add Cinder as a submodule
add_subdirectory(extern/Cinder)

# Create the executable
add_executable(example_0 src/example_0/main.cpp)
add_executable(example_1 src/example_1/main.cpp)
add_executable(example_2 src/example_2/main.cpp src/example_2/FluidGrid.cpp)
add_executable(example_3 src/example_3/main.cpp)

target_compile_options(example_3 PRIVATE -mavx2)

# Link against Cinder library
target_link_libraries(example_0 PUBLIC cinder)
target_link_libraries(example_1 PUBLIC cinder)
target_link_libraries(example_2 PUBLIC cinder OpenMP::OpenMP_CXX)
target_link_libraries(example_3 PUBLIC cinder OpenMP::OpenMP_CXX)
