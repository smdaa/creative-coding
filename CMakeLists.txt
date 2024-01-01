cmake_minimum_required(VERSION 3.26)
project(creative-coding)

# Add Cinder as a submodule
add_subdirectory(extern/Cinder)

# Create the executable
add_executable(example_0 src/example_0.cpp)

# Link against Cinder library
target_link_libraries(example_0 PUBLIC cinder)