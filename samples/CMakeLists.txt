cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(adaptivecpp-samples)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ACPP_SOURCE_ROOT}/cmake)

find_package(AdaptiveCpp CONFIG REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT ACPP_DEBUG_LEVEL)
  if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(ACPP_DEBUG_LEVEL 3 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  else()
    set(ACPP_DEBUG_LEVEL 2 CACHE STRING
      "Choose the debug level, options are: 0 (no debug), 1 (print errors), 2 (also print warnings), 3 (also print general information)"
FORCE)
  endif()
endif()

cmake_policy(SET CMP0005 NEW)
add_definitions(-DHIPSYCL_DEBUG_LEVEL=${ACPP_DEBUG_LEVEL})

if(WIN32)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

add_subdirectory(sycl_on_vulkan)
