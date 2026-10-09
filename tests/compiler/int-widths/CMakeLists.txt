cmake_minimum_required(VERSION 3.12)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(acpp-int-width-tests LANGUAGES C CXX)
  enable_testing()
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(ACPP_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_executable(acpp-int-width-test-driver driver.cpp
  "${ACPP_SOURCE_ROOT}/src/compiler/llvm-to-backend/clspv/LegalizeIntWidthsPass.cpp")
target_compile_features(acpp-int-width-test-driver PRIVATE cxx_std_17)
target_compile_definitions(acpp-int-width-test-driver PRIVATE HIPSYCL_COMPILER_COMPONENT)
target_include_directories(acpp-int-width-test-driver PRIVATE
  "${ACPP_SOURCE_ROOT}/include" ${LLVM_INCLUDE_DIRS})
if(TARGET LLVM)
  target_link_libraries(acpp-int-width-test-driver PRIVATE LLVM)
else()
  llvm_map_components_to_libnames(INT_WIDTH_LLVM_LIBS Core IRReader TransformUtils Analysis)
  target_link_libraries(acpp-int-width-test-driver PRIVATE ${INT_WIDTH_LLVM_LIBS})
endif()
add_test(NAME legalize-int-widths
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test.py"
    "$<TARGET_FILE:acpp-int-width-test-driver>" "${LLVM_TOOLS_BINARY_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}")
