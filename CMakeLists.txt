# CMakeList.txt : CMake project for yopo-experiment, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

enable_language(CUDA)

list(APPEND CMAKE_PREFIX_PATH "D:/ProgramData/libtorch-win-shared-with-deps-debug-1.6.0/libtorch")
set(CUDNN_LIBRARY_PATH ${CUDA_LIBRARIES})
set(CUDNN_INCLUDE_PATH ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

project ("yopo-experiment" CXX CUDA)
find_package(Torch REQUIRED "D:/ProgramData/libtorch-win-shared-with-deps-debug-1.6.0/libtorch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB YOPO-EXP-HEADER *.h */*.h)
file(GLOB YOPO-EXP-SRC *.cpp */*.cpp)
message(WARNING ${TORCH_LIBRARIES})
# Add source to this project's executable.
add_executable (yopo-experiment ${YOPO-EXP-HEADER} ${YOPO-EXP-SRC})
target_link_libraries(yopo-experiment "${TORCH_LIBRARIES}")
set_property(TARGET yopo-experiment PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
add_custom_command(TARGET yopo-experiment
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:yopo-experiment>)
