cmake_minimum_required( VERSION 2.8 )

#------------Python Packages included-------------------#
find_package( PythonLibs 2.7 REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )
find_package( Boost COMPONENTS python REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
#------------Python Packages included-------------------#


#------------OpenCL and clBLAS libraries----------------# 
include_directories( /opt/intel/opencl/include )
#-------------Using the prebuilt libraries clBLAS obtained from GitHub-----------------------#
include_directories( /home/admin/Documents/SoftwareLibraries/clBLAS-2.12.0-Linux-x64/include )
link_directories(/home/admin/Documents/SoftwareLibraries/clBLAS-2.12.0-Linux-x64/lib64)
#-------------Using the prebuilt libraries clBLAS obtained from GitHub-----------------------#
#------------OpenCL and clBLAS libraries----------------# 


message("-------")
message( ${Boost_INCLUDE_DIR} )
message( ${PYTHON_INCLUDE_DIRS} )
message("-------")

#-------------Creation of the shared library---------# 
add_library( MMUL_clBLAS_Lib SHARED MMUL_clBLAS.cpp)
#-------------Creation of the shared library---------# 


target_link_libraries( MMUL_clBLAS_Lib libOpenCL.so clBLAS.so ${Boost_LIBRARIES} ${PYTHON_LIBRARY} libboost_numpy.so )
set_target_properties( MMUL_clBLAS_Lib PROPERTIES PREFIX "" )



