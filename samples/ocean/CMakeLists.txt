option(ACPP_SAMPLES_BUILD_OCEAN "Build the SYCL/Vulkan ocean surface simulation sample" ON)

if(NOT ACPP_SAMPLES_BUILD_OCEAN)
  return()
endif()

# The rendering part of this sample needs Vulkan, GLFW and GLM. If any of them
# is missing, the sample is skipped instead of failing the configure step.
find_package(Vulkan)
find_package(glfw3 3.3 QUIET)
find_package(glm QUIET)

if(NOT glm_FOUND)
  find_path(GLM_INCLUDE_DIR glm/glm.hpp)
endif()

# A GLSL to SPIR-V compiler is required to build the shaders.
find_program(ACPP_GLSLC_EXECUTABLE NAMES glslc
  HINTS ${Vulkan_GLSLC_EXECUTABLE})
find_program(ACPP_GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator
  HINTS ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE})

if(NOT Vulkan_FOUND OR NOT TARGET glfw)
  message(STATUS "ocean sample: Vulkan or GLFW not found, skipping sample")
  return()
endif()

if(NOT glm_FOUND AND NOT GLM_INCLUDE_DIR)
  message(STATUS "ocean sample: GLM not found, skipping sample")
  return()
endif()

if(NOT ACPP_GLSLC_EXECUTABLE AND NOT ACPP_GLSLANG_VALIDATOR_EXECUTABLE)
  message(STATUS "ocean sample: no GLSL compiler (glslc or glslangValidator) "
                 "found, skipping sample")
  return()
endif()

add_executable(ocean
  main.cpp
  ocean_application.cpp
  ocean_simulation.cpp)

# Only the simulation is compiled by the SYCL compiler, the Vulkan based
# renderer is ordinary C++.
add_sycl_to_target(TARGET ocean SOURCES ocean_simulation.cpp)

target_link_libraries(ocean PRIVATE Vulkan::Vulkan glfw)

if(TARGET glm::glm)
  target_link_libraries(ocean PRIVATE glm::glm)
else()
  target_include_directories(ocean PRIVATE ${GLM_INCLUDE_DIR})
endif()

if(WIN32)
  target_compile_definitions(ocean PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Compile the shaders to SPIR-V next to the executable so that they can be
# found relative to it at runtime.
set(ACPP_OCEAN_SHADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY "${ACPP_OCEAN_SHADER_DIR}")

set(ACPP_OCEAN_SPIRV_FILES)
foreach(SHADER IN ITEMS ocean.vert ocean.frag)
  set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER}")
  set(SHADER_BINARY "${ACPP_OCEAN_SHADER_DIR}/${SHADER}.spv")

  if(ACPP_GLSLC_EXECUTABLE)
    set(SHADER_COMMAND ${ACPP_GLSLC_EXECUTABLE} ${SHADER_SOURCE}
      -o ${SHADER_BINARY})
  else()
    set(SHADER_COMMAND ${ACPP_GLSLANG_VALIDATOR_EXECUTABLE} -V ${SHADER_SOURCE}
      -o ${SHADER_BINARY})
  endif()

  add_custom_command(
    OUTPUT ${SHADER_BINARY}
    COMMAND ${SHADER_COMMAND}
    DEPENDS ${SHADER_SOURCE}
    COMMENT "Compiling shader ${SHADER}"
    VERBATIM)

  list(APPEND ACPP_OCEAN_SPIRV_FILES ${SHADER_BINARY})
endforeach()

add_custom_target(ocean_shaders DEPENDS ${ACPP_OCEAN_SPIRV_FILES})
add_dependencies(ocean ocean_shaders)

install(TARGETS ocean
        RUNTIME DESTINATION share/AdaptiveCpp/samples/)
install(FILES ${ACPP_OCEAN_SPIRV_FILES}
        DESTINATION share/AdaptiveCpp/samples/shaders/)
