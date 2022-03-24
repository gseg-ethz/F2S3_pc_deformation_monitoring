cmake_minimum_required(VERSION 2.8)

project(pc_tiling)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -Wall -pedantic -Wextra -Wno-unused-parameter -fPIC")

# Add liba
add_library(pc_tiling SHARED pc_tiling.cpp)

# Set up public headers for liba
#set_target_properties(supervoxel PROPERTIES PUBLIC_HEADER "supervoxel.h")


find_package(PCL 1.8 REQUIRED )
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

if(${PYTHON_VERSION_STRING} GREATER 3.0)
  message(STATUS "Using Python3")
  set(Boost_LIBRARIES -lboost_python3)
else()
  message(STATUS "Using Python2")
  set(Boost_LIBRARIES -lboost_python -lboost_numpy)
endif()

if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  
  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )
endif()

include_directories(${Boost_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# This is a CMake example for Python

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_SWIG_FLAGS "-Wall")

SET_SOURCE_FILES_PROPERTIES(pc_tiling.i PROPERTIES CPLUSPLUS ON)
#SET_SOURCE_FILES_PROPERTIES(example.i PROPERTIES SWIG_FLAGS "-includeall")
SWIG_ADD_MODULE(pc_tiling python pc_tiling.i pc_tiling.cpp)
SWIG_LINK_LIBRARIES(pc_tiling ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${PCL_LIBRARIES})








