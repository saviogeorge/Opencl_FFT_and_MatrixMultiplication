//This program computes the FFT making us of the
//OpenCL implementation of the FFT library clFFT
//The FFT is computed on Intel Integrated GPU 
//The following source is compiled into a library 
//and is can be included and used in a python program  
#include <clFFT.h> //for the clFFT API's it also includes cl.h internally 
#include <cstdio>
//Necessary inclusion for the boost library packages being used
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/tuple.hpp>
#include <time.h>

namespace numn = boost::python::numpy;
namespace bn = boost::python::numeric;
using namespace boost::python;

#define FASTCONVERT(type) \
		{type * reinArray = reinterpret_cast<type*>(arrayFromPytest);\
		for(int i=0;i<m_N;i++)\
		{\
			X[i*2] = reinArray[i];\
			X[i*2 + 1] = 0;\
		}}
//We used a class here because the data members could 
//be used as globally declared variables and we had issues 
//while trying to create library
class OpenClFFT
{

public:
	OpenClFFT();
	~OpenClFFT();
	void SetParam(int N);
	void ResetParam();
	numn::ndarray doFFT(const numn::ndarray& s);
	double getExeTime();
	double getWriteTime();
	double getReadTime();
	
private:
	cl_int err;
	cl_platform_id platform ;
	cl_device_id device;
	cl_context ctx ;
	cl_command_queue queue;
	cl_mem bufX;

/* FFT library realted declarations */
	clfftSetupData fftSetup;
	clfftPlanHandle planHandle;
	clfftDim dim;
	float *X;
	int m_N;
	double exec_time;
	double write_time;
	double read_time;
};

//Function to print the error associated with an error code
void printError(cl_int error) {
  // Print error message
  switch(error)
  {
    case -1:
      printf("CL_DEVICE_NOT_FOUND ");
      break;
    case -2:
      printf("CL_DEVICE_NOT_AVAILABLE ");
      break;
    case -3:
      printf("CL_COMPILER_NOT_AVAILABLE ");
      break;
    case -4:
      printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
      break;
    case -5:
      printf("CL_OUT_OF_RESOURCES ");
      break;
    case -6:
      printf("CL_OUT_OF_HOST_MEMORY ");
      break;
    case -7:
      printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
      break;
    case -8:
      printf("CL_MEM_COPY_OVERLAP ");
      break;
    case -9:
      printf("CL_IMAGE_FORMAT_MISMATCH ");
      break;
    case -10:
      printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
      break;
    case -11:
      printf("CL_BUILD_PROGRAM_FAILURE ");
      break;
    case -12:
      printf("CL_MAP_FAILURE ");
      break;
    case -13:
      printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
      break;
    case -14:
      printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
      break;

    case -30:
      printf("CL_INVALID_VALUE ");
      break;
    case -31:
      printf("CL_INVALID_DEVICE_TYPE ");
      break;
    case -32:
      printf("CL_INVALID_PLATFORM ");
      break;
    case -33:
      printf("CL_INVALID_DEVICE ");
      break;
    case -34:
      printf("CL_INVALID_CONTEXT ");
      break;
    case -35:
      printf("CL_INVALID_QUEUE_PROPERTIES ");
      break;
    case -36:
      printf("CL_INVALID_COMMAND_QUEUE ");
      break;
    case -37:
      printf("CL_INVALID_HOST_PTR ");
      break;
    case -38:
      printf("CL_INVALID_MEM_OBJECT ");
      break;
    case -39:
      printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
      break;
    case -40:
      printf("CL_INVALID_IMAGE_SIZE ");
      break;
    case -41:
      printf("CL_INVALID_SAMPLER ");
      break;
    case -42:
      printf("CL_INVALID_BINARY ");
      break;
    case -43:
      printf("CL_INVALID_BUILD_OPTIONS ");
      break;
    case -44:
      printf("CL_INVALID_PROGRAM ");
      break;
    case -45:
      printf("CL_INVALID_PROGRAM_EXECUTABLE ");
      break;
    case -46:
      printf("CL_INVALID_KERNEL_NAME ");
      break;
    case -47:
      printf("CL_INVALID_KERNEL_DEFINITION ");
      break;
    case -48:
      printf("CL_INVALID_KERNEL ");
      break;
    case -49:
      printf("CL_INVALID_ARG_INDEX ");
      break;
    case -50:
      printf("CL_INVALID_ARG_VALUE ");
      break;
    case -51:
      printf("CL_INVALID_ARG_SIZE ");
      break;
    case -52:
      printf("CL_INVALID_KERNEL_ARGS ");
      break;
    case -53:
      printf("CL_INVALID_WORK_DIMENSION ");
      break;
    case -54:
      printf("CL_INVALID_WORK_GROUP_SIZE ");
      break;
    case -55:
      printf("CL_INVALID_WORK_ITEM_SIZE ");
      break;
    case -56:
      printf("CL_INVALID_GLOBAL_OFFSET ");
      break;
    case -57:
      printf("CL_INVALID_EVENT_WAIT_LIST ");
      break;
    case -58:
      printf("CL_INVALID_EVENT ");
      break;
    case -59:
      printf("CL_INVALID_OPERATION ");
      break;
    case -60:
      printf("CL_INVALID_GL_OBJECT ");
      break;
    case -61:
      printf("CL_INVALID_BUFFER_SIZE ");
      break;
    case -62:
      printf("CL_INVALID_MIP_LEVEL ");
      break;
    case -63:
      printf("CL_INVALID_GLOBAL_WORK_SIZE ");
      break;
    default:
      printf("UNRECOGNIZED ERROR CODE (%d)", error);
  }
}



void _checkError(int line,
		 const char *file,
		 cl_int error,
                 const char *msg,
                 ...); // does not return
#define checkError(status, ...) _checkError(__LINE__, __FILE__, status, __VA_ARGS__)



// Print line, file name, and error code if there is an error. Exits the
// application upon error.
void _checkError(int line,
                 const char *file,
                 cl_int error,
                 const char *msg,
                 ...) {
  // If not successful
  if(error != CL_SUCCESS) {
    // Print line and file
    printf("ERROR: ");
    printError(error);
    printf("\nLocation: %s:%d\n", file, line);

    // Print custom message.
    va_list vl;
    va_start(vl, msg);
    vprintf(msg, vl);
    printf("\n");
    va_end(vl);

    throw;
  }
}

//Definition of the constructor
OpenClFFT::OpenClFFT() : dim(CLFFT_1D)
{

	platform = 0;
	device = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	ctx = 0;
	X=NULL;
	bufX=NULL;

	int ret = 0;
	m_N = 0;
	err = clGetPlatformIDs( 1, &platform, NULL );
	checkError(err, "Failed to GetPlatformIDs");
	err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	checkError(err, "Failed to GetDeviceID");

	props[1] = (cl_context_properties)platform;
	ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
	checkError(err, "Failed to create Context");
	queue = clCreateCommandQueue( ctx, device, CL_QUEUE_PROFILING_ENABLE, &err );
	checkError(err, "Failed to create Command Queue");

	err = clfftInitSetupData(&fftSetup);
	checkError(err, "Failed to InitSetupData");
	err = clfftSetup(&fftSetup);
	checkError(err, "Failed to fftSetup");


}
//Definition of the Destructor
OpenClFFT::~OpenClFFT()
{	

	if(bufX)
	{
		/* Release OpenCL memory objects. */
		err = clReleaseMemObject( bufX );
		checkError(err, "Failed to release bufX");
	}
	if(X)
	{
		free(X);
	}
	/* Release the plan. */
	//err = clfftDestroyPlan( &planHandle );
	//checkError(err, "Failed to Destroy Plan");

	//todo I am a mem leak
	/* Release clFFT library. */
	//err = clfftTeardown( );
	//checkError(err, "Failed to release Command Queue");
	/* Release OpenCL working objects. */
	err = clReleaseCommandQueue( queue );
	checkError(err, "Failed to release Command Queue");
	err = clReleaseContext( ctx );
	checkError(err, "Failed to release Context");
}

//Function to compute the event execution time(read, write, kernel execution)
double get_event_exec_time(cl_event event)
{
	cl_ulong start_time, end_time;
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start_time,NULL);
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END, sizeof(cl_ulong),&end_time,NULL);
	double total_time = (end_time - start_time)*1e-6;
	return total_time;
}

//Function to set the necessary parameters for the clFFT and the 
//OpenCL to function
void OpenClFFT::SetParam(int N)
{

	/* Setup clFFT. */
	m_N = N;
	//printf("Set Param");
	if(X)
	{
		free(X);	
	}
	if(bufX)
	{
		err = clReleaseMemObject( bufX );
		checkError(err, "Failed to release bufX");
	}

	X = (float *)malloc(m_N * 2 * sizeof(*X));
	size_t clLengths[1] = {m_N};
	/* Prepare OpenCL memory objects  */
	bufX = clCreateBuffer( ctx, CL_MEM_READ_WRITE, m_N * 2 * sizeof(*X), NULL, &err );

	/* Create a default plan for a complex FFT. */
	err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);
	checkError(err, "Failed to CreateDefaultPlane");

	/* Set plan parameters. */
	err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
	checkError(err, "Failed to SetPlanPrecisio");
	err = clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
	checkError(err, "Failed to SetLayout");
	err = clfftSetResultLocation(planHandle, CLFFT_INPLACE);
	checkError(err, "Failed to SetResultLocation");

	/* Bake the plan. */
	err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
	checkError(err, "Failed to fftBakePlan");
}

//Function to reset the set parameters
void OpenClFFT::ResetParam()
{
	err = clfftDestroyPlan( &planHandle );
	checkError(err, "Failed to Destroy Plan");
}
//Function to get the kernel execution time
double OpenClFFT::getExeTime()
{
	return exec_time;
}
//Function to get the time taken for the data to get transferred to the device memory (GPU) 
//from the host (CPU)
double OpenClFFT::getWriteTime()
{
	return write_time;
}
//Function to get the time taken for the results to get transferred from the device memory (GPU)
//to the host (CPU) 
double OpenClFFT::getReadTime()
{
	return read_time;
}

//Function to compute the FFT and get back the results
numn::ndarray OpenClFFT::doFFT(const numn::ndarray& arrayFromPy)
{
	//create return type
	numn::dtype dt = numn::dtype::get_builtin<std::complex<float> >();
	tuple shape = make_tuple(m_N);
	numn::ndarray result= numn::empty(shape,dt);

	//get data from py and put it into buffer to transport to the OpenCl
	unsigned char * arrayFromPyRes = (unsigned char *)result.get_data();
	unsigned char * arrayFromPytest = (unsigned char *)arrayFromPy.get_data();
	numn::dtype pseudoSwitch = arrayFromPy.get_dtype();
	cl_event write_event=NULL;
	
	//this should be a swtich but is not allowed to be one because of C++98 restriction
	if(pseudoSwitch == numn::detail::get_complex_dtype<64>())
	{
		//this is the fastest yust use the buffer
		err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,m_N * 2 * sizeof( *X ), arrayFromPytest, 0, NULL, &write_event );
		checkError(err, "Failed to EnqueueWriteBuffer");
	} else 
	{
		if(pseudoSwitch == numn::detail::get_complex_dtype<128>())
		{
			double * reinArray = reinterpret_cast<double*>(arrayFromPytest);
			for(int i=0;i<m_N;i++)
			{
				X[i*2] = arrayFromPytest[i*2];
				X[i*2 + 1] = arrayFromPytest[i*2 + 1];
			}
		}else if(pseudoSwitch == numn::detail::get_float_dtype<32>())
		{
			FASTCONVERT(float)
		}else if(pseudoSwitch == numn::detail::get_float_dtype<64>())
		{
			//printf("ich bin ein marker");
			FASTCONVERT(double)
			//printf("ich bin auch ein marker");
		
		}else if(pseudoSwitch == numn::detail::get_int_dtype<8,true>())
		{
			FASTCONVERT(signed char)
		}else if(pseudoSwitch == numn::detail::get_int_dtype<16,true>())
		{
			FASTCONVERT(signed short)
		}else if(pseudoSwitch == numn::detail::get_int_dtype<32,true>())
		{
			FASTCONVERT(signed int)
		}else if(pseudoSwitch == numn::detail::get_int_dtype<64,true>())
		{
			FASTCONVERT(signed long long)
		}else if(pseudoSwitch == numn::detail::get_int_dtype<8,false>())
		{
			FASTCONVERT(unsigned char)
		}else if(pseudoSwitch == numn::detail::get_int_dtype<16,false>())
		{
			FASTCONVERT(unsigned short)
		}else if(pseudoSwitch == numn::detail::get_int_dtype<32,false>())
		{
			FASTCONVERT(unsigned int)
		}else if(pseudoSwitch == numn::detail::get_int_dtype<64,false>())
		{
			FASTCONVERT(unsigned long long)
		} else
		{
			printf("Warning unknown Type using slow Method\n");
			for(int i=0;i<m_N;i++)
			{
				object current = arrayFromPy.attr("__getitem__")(i);
				object realObj = current.attr("real");
				object imagObj = current.attr("imag");
				X[i*2] = boost::python::extract<float> ( realObj.attr("__float__")());
				X[i*2 + 1] = boost::python::extract<float> ( imagObj.attr("__float__")());
			}
		}
		err = clEnqueueWriteBuffer( queue, bufX, CL_TRUE, 0,m_N * 2 * sizeof( *X ), X, 0, NULL, &write_event );
		checkError(err, "Failed to EnqueueWriteBuffer");
	}
	
	/* Execute the plan. */
	cl_event Kernel_event=NULL;
	err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 1, &write_event, &Kernel_event, &bufX, NULL, NULL);
	checkError(err, "Failed to fftEnqueueTransform");

	/* Wait for calculations to be finished. */
	err = clFinish(queue);
	checkError(err, "Failed to Finish");


	/* Fetch results of calculations. */
	cl_event read_event=NULL;
	err = clEnqueueReadBuffer( queue, bufX, CL_TRUE, 0, m_N * 2 * sizeof( *X ), arrayFromPyRes, 1, &Kernel_event, &read_event );
	checkError(err, "Failed to EnqueueReadBuffer");

	write_time	=	get_event_exec_time(write_event);
	exec_time	=	get_event_exec_time(Kernel_event);
	read_time	=	get_event_exec_time(read_event);

	clReleaseEvent(write_event);
	clReleaseEvent(Kernel_event);
	clReleaseEvent(read_event);

	
	return result;
}

//Boost library related declarations 
//The functions declared here can be used as 
//API's in python
BOOST_PYTHON_MODULE(FFT_clFFT)
{
	bn::array::set_module_and_type("numpy","ndarray");
	using namespace boost::python;
	Py_Initialize();
	numn::initialize();
	class_<OpenClFFT>("OpenClFFT",init<>())
	.def("SetParam",&OpenClFFT::SetParam)
	.def("ResetParam",&OpenClFFT::ResetParam)
	.def("doFFT",&OpenClFFT::doFFT)
	.def("getWriteTime",&OpenClFFT::getWriteTime)
	.def("getExeTime",&OpenClFFT::getExeTime)
	.def("getReadTime",&OpenClFFT::getReadTime);

}
