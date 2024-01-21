cmake_minimum_required(VERSION 3.22)
project(creative-coding)

# Add Cinder as a submodule
add_subdirectory(extern/Cinder)

# Create the executable
add_executable(example_0 src/example_0/main.cpp)
add_executable(example_1 src/example_1/main.cpp)
add_executable(example_2 src/example_2/main.cpp)

# Link against Cinder library
target_link_libraries(example_0 PUBLIC cinder)
target_link_libraries(example_1 PUBLIC cinder)
target_link_libraries(example_2 PUBLIC cinder)
