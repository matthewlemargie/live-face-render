cmake_minimum_required(VERSION 3.10)

# Define the project name and version
project(MyProject VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)

# Specify the path to the lib directory
# link_directories(${CMAKE_SOURCE_DIR}/lib)
file(GLOB SOURCES "*.cpp")

# Add executable target
add_executable(main ${SOURCES})

# Link the library (e.g., static or shared)
target_link_libraries(main PRIVATE glfw GLEW GL)
