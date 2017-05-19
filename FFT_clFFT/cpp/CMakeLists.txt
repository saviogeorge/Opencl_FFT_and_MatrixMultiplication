cmake_minimum_required( VERSION 2.8 )

find_package( PythonLibs 2.7 REQUIRED )
include_directories( ${PYTHON_INCLUDE_DIRS} )
find_package( Boost COMPONENTS python REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )
message("-------")
message( ${Boost_INCLUDE_DIR} )
message( ${PYTHON_INCLUDE_DIRS} )
message("-------")

add_library( FFT_clFFT SHARED FFT_clFFT.cpp)
target_link_libraries( FFT_clFFT libOpenCL.so clFFT.so ${Boost_LIBRARIES} ${PYTHON_LIBRARY} libboost_numpy.so ) 
set_target_properties( FFT_clFFT PROPERTIES PREFIX "" )


