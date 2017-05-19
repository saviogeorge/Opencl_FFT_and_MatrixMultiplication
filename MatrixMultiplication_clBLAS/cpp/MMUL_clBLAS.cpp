//This program computes the matrix multiplication making us of the
//OpenCL implementation of the maths library clBLAS
//The matrix multiplication is computed on Intel Integrated GPU 
//The following source is compiled into a library 
//and is can be included and used in a python program 
#include <clBLAS.h> //clBlas has cl.h included
#include <cstdio>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/tuple.hpp>
#include <time.h>

#define BUFFER_SIZE 10240

#define FASTCONVERT(type) \
		{type * reinArray = reinterpret_cast<type*>(arrayFromPytest);\
		for(int i=0;i<size;i++)\
		{\
			X[i] = reinArray[i];\
			printf("%f",X[i]);\
		}}

// Print the error associated with an error code
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

namespace numn = boost::python::numpy;
namespace bn = boost::python::numeric;
using namespace boost::python;

//We used a class here because the data members could 
//be used as globally declared variables and we had issues 
//while trying to create library
class OpenClMMUL_clBLAS
{
	public:
	OpenClMMUL_clBLAS();
	~OpenClMMUL_clBLAS();
	numn::ndarray doMMUL(const numn::ndarray& ma1,const numn::ndarray& ma2);
	double getExeTime();
	double getWriteTime();
	double getReadTime();
	void getPlatform_Device_Details();
	void SetParam(int Matrix1Row,int Matrix1ColMatrix2Row,int Matrix2Col );

	private:
	void bufferToFloatArray(const numn::ndarray& matr,cl_event* write_event,int size,float * X,cl_mem ClMem,int offset,int offsetRead);
	//The following members where initially declared globally
	//For some reason we where not able to generate
	//a shared library when the following members where global
	//(Note: However we could create an executable)
	//By moving these objects and variables into class solved the issue for us 
	cl_int err;
	cl_platform_id platform ;
	cl_device_id device;
	cl_context ctx ;
	cl_command_queue queue;
	cl_mem bufA;
	cl_mem bufB;
	cl_mem bufC;
	cl_program program;               
	cl_kernel kernel;                  

	float* MatrixIn1;
	float* MatrixIn2;
	float* MatrixOut3;
	int m_Matrix1Row;
	int m_Matrix1ColMatrix2Row;
	int m_Matrix2Col;
	double exec_time;
	double write_time;
	double read_time;
};

//Definition of the Destructor
OpenClMMUL_clBLAS::~OpenClMMUL_clBLAS()
{

	/* Release OpenCL memory objects. */
	err = clReleaseMemObject( bufA );
	checkError(err, "Failed to release bufA");
	err = clReleaseMemObject( bufB );
	checkError(err, "Failed to release bufB");
	err = clReleaseMemObject( bufC );
	checkError(err, "Failed to release bufC");
	//todo I am a mem leak
	/* Release clBlas library. */
	//clblasTeardown();
	//CHECKERROR
	/* Release OpenCL working objects. */
	cl_int t = 0;
	clGetCommandQueueInfo(queue, CL_QUEUE_REFERENCE_COUNT,sizeof(cl_uint),&t,NULL);
	printf("cmdque ref %d\n",t);
	err = clReleaseCommandQueue( queue );
	checkError(err, "Failed to release Command Queue");
	clGetContextInfo(ctx, CL_CONTEXT_REFERENCE_COUNT,sizeof(cl_uint),&t,NULL);
	printf("conque ref %d\n",t);
	err = clReleaseContext( ctx );
	checkError(err, "Failed to release Context");
	free(MatrixIn1);
	free(MatrixIn2);
	free(MatrixOut3);
}

double get_event_exec_time(cl_event event)
{
	cl_ulong start_time, end_time;
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(cl_ulong),&start_time,NULL);
	clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END, sizeof(cl_ulong),&end_time,NULL);
	double total_time = (end_time - start_time)*1e-6;
	return total_time;
}
//Definition of the constructor
OpenClMMUL_clBLAS::OpenClMMUL_clBLAS()
{

	platform = 0;
	device = 0;
	queue = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	ctx = 0;

	err = clGetPlatformIDs( 1, &platform, NULL );
	checkError(err, "Failed to get PlatformIDs");
	err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	checkError(err, "Failed to get DeviceIDs");

	props[1] = (cl_context_properties)platform;
	ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
	checkError(err, "Failed to create Context");
	queue = clCreateCommandQueue( ctx, device, CL_QUEUE_PROFILING_ENABLE, &err );
	checkError(err, "Failed to create Command Queue");

	MatrixIn1 	= NULL;
	MatrixIn2 	= NULL;
	MatrixOut3 	= NULL;

	bufA = NULL;
	bufB = NULL;
	bufC = NULL;

    
}
//Function to set the necessary parameters for the clFFT and the 
//OpenCL to function
void OpenClMMUL_clBLAS::SetParam(int Matrix1Row,int Matrix1ColMatrix2Row,int Matrix2Col )
{

	free(MatrixIn1);
	free(MatrixIn2);
	free(MatrixOut3);

	/* Release OpenCL memory objects. */
	if(bufA)
	{
		err = clReleaseMemObject( bufA );
		checkError(err, "Failed to release bufA");
	}
	if(bufB)
	{
		err = clReleaseMemObject( bufB );
		checkError(err, "Failed to release bufB");
	}
	if(bufC)
	{
		err = clReleaseMemObject( bufC );
		checkError(err, "Failed to release bufC");
	}


	m_Matrix1Row 		=	Matrix1Row;
	m_Matrix1ColMatrix2Row 	=	Matrix1ColMatrix2Row;
	m_Matrix2Col 		=	Matrix2Col;

	MatrixIn1 	= (float *)malloc(m_Matrix1Row * m_Matrix1ColMatrix2Row * sizeof(*MatrixIn1));
	MatrixIn2 	= (float *)malloc(m_Matrix1ColMatrix2Row * m_Matrix2Col * sizeof(*MatrixIn2));
	MatrixOut3 	= (float *)malloc(m_Matrix1Row * m_Matrix2Col 		* sizeof(*MatrixOut3));


	/* Prepare OpenCL memory objects and place data inside them. */
	bufA = clCreateBuffer( ctx, CL_MEM_READ_WRITE, m_Matrix1Row * m_Matrix1ColMatrix2Row * sizeof(*MatrixIn1), NULL, &err );
	checkError(err, "Failed to create Input Buffer A");
	/* Prepare OpenCL memory objects and place data inside them. */
	bufB = clCreateBuffer( ctx, CL_MEM_READ_WRITE, m_Matrix1ColMatrix2Row * m_Matrix2Col * sizeof(*MatrixIn2), NULL, &err );
	checkError(err, "Failed to create Input Buffer B");

	/* Prepare OpenCL memory objects and place data inside them. */
	bufC = clCreateBuffer( ctx, CL_MEM_READ_WRITE, m_Matrix1Row * m_Matrix2Col  * sizeof(*MatrixOut3), NULL, &err );
	checkError(err, "Failed to create Output Buffer C");	
}
//Function to get the kernel execution time
double OpenClMMUL_clBLAS::getExeTime()
{
	return exec_time;
}
//Function to get the time taken for the data to get transferred to the device memory (GPU) 
//from the host (CPU)
double OpenClMMUL_clBLAS::getWriteTime()
{
	return write_time;
}
//Function to get the time taken for the results to get transferred from the device memory (GPU)
//to the host (CPU)
double OpenClMMUL_clBLAS::getReadTime()
{
	return read_time;
}


void print(const float *arr, int m, int n)
{
	int i,j;
	for(i=0;i<m;i++)
	{
		for(j=0;j<n;j++)
		{
			printf("\t%f",*((arr+i*n)+j));
			
		}
		printf("\n");
	}
}

//Function to the platform and the device details
void OpenClMMUL_clBLAS::getPlatform_Device_Details()
{
	char buffer[BUFFER_SIZE];
	err = clGetPlatformInfo(platform,CL_PLATFORM_NAME,BUFFER_SIZE,buffer,NULL);
	checkError(err, "Failed to get Platform Name");
	printf("Platform Name:%s\n",buffer);

	err = clGetPlatformInfo(platform,CL_PLATFORM_VERSION,BUFFER_SIZE,buffer,NULL);
	checkError(err, "Failed to get Platform Version");
	printf("Platform Version:%s\n",buffer);

	err = clGetPlatformInfo(platform,CL_PLATFORM_VENDOR,BUFFER_SIZE,buffer,NULL);
	checkError(err, "Failed to get Platform Vendor");
	printf("Platform Vendor:%s\n",buffer);

	err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
	checkError(err, "Failed to get Device Name");
	printf("Device Name:%s\n",buffer);

	err = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
	checkError(err, "Failed to get Device Version");
	printf("Device Version:%s\n",buffer);

	err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
	checkError(err, "Failed to get Device Vendor");
	printf("Device Vendor:%s\n",buffer);
			
}

//Function to convert the received inputs from python to C and write to buffer
void OpenClMMUL_clBLAS::bufferToFloatArray(	const numn::ndarray& matr,
						cl_event* write_event,
						int size,
						float * X,
						cl_mem ClMem,
						int offset,
						int offsetRead	)
{
	//get data from py and put it into buffer to transport to the OpenCl
	unsigned char * arrayFromPytest = (unsigned char *)matr.get_data() +offsetRead;
	numn::dtype pseudoSwitch = matr.get_dtype();
	
	//this should be a swtich but is not allowed to be one because of C++98 restriction
	if(pseudoSwitch == numn::detail::get_float_dtype<32>())
	{
		//this is the fastest use the buffer
		err = clEnqueueWriteBuffer( queue, ClMem, CL_TRUE, offset,size * sizeof( float ), arrayFromPytest, 0, NULL, write_event );

		checkError(err, "Failed to Write to Buffer");
	} else 
	{
		if(pseudoSwitch == numn::detail::get_float_dtype<64>())
		{
			FASTCONVERT(double)
		
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
			for(int i=0;i<m_Matrix2Col;i++)
			{
				object current = matr.attr("__getitem__")(i);
				object realObj = current.attr("real");
				object imagObj = current.attr("imag");
				X[i*2] = boost::python::extract<float> ( realObj.attr("__float__")());
				X[i*2 + 1] = boost::python::extract<float> ( imagObj.attr("__float__")());
			}
		}
		err = clEnqueueWriteBuffer( queue, ClMem, CL_TRUE, offset,size * sizeof( float ), X, 0, NULL, write_event );
		checkError(err, "Failed to Write to buffer");
	}
}
//Function to compute the FFT and get back the results
numn::ndarray OpenClMMUL_clBLAS::doMMUL(const numn::ndarray& inMatr1,const numn::ndarray& inMatr2)
{
	//----------Parameters set for the clBLAS Library Function clblasSgemm-----------//  
	cl_float alpha=1.0;
	cl_float beta=0.0;
	size_t l_offset_A=0;
	size_t l_offset_B=0;
	size_t l_offset_C=0;
	cl_uint numberCommandQueues=1;
	cl_uint numberEventsinWaitList=2;
	cl_event Kernel_event=NULL;
	int lda = m_Matrix1ColMatrix2Row;
	int ldb = m_Matrix2Col;
	int ldc = m_Matrix2Col;
	cl_event write_event[2];
	write_event[0] = NULL;
	write_event[1] = NULL;
	//----------Parameters set for the clBLAS Library Function clblasSgemm-----------// 

	//create return type
	numn::dtype dt = numn::dtype::get_builtin<float>();
	tuple shape = make_tuple(m_Matrix1Row,m_Matrix2Col);
	numn::ndarray result= numn::empty(shape,dt);
	unsigned char * arrayFromPyRes = (unsigned char *)result.get_data();
	int l_offset=0;
	int l_offsetRead=0;

	//getPlatform_Device_Details();

	bufferToFloatArray(	inMatr1,
				&write_event[0],
				m_Matrix1Row*m_Matrix1ColMatrix2Row,
				MatrixIn1,
				bufA,
				l_offset,
				l_offsetRead	);

	bufferToFloatArray(	inMatr2,
				&write_event[1],
				m_Matrix1ColMatrix2Row*m_Matrix2Col,
				MatrixIn2,
				bufB,
				l_offset,
				l_offsetRead	);

	/* Execute the plan. */
	err = clblasSgemm( 	clblasRowMajor, 
				clblasNoTrans, 
				clblasNoTrans,
		                m_Matrix1Row, 
				m_Matrix2Col, 
				m_Matrix1ColMatrix2Row,
		                alpha, 
				bufA, 
				l_offset_A, 
				lda,
		                bufB, 
				l_offset_B, 
				ldb, 
				beta,
		                bufC, 
				l_offset_C, 
				ldc,
		                numberCommandQueues, 
				&queue, 
				numberEventsinWaitList, 
				write_event, 
				&Kernel_event 	);

	checkError(err, " clBlas Matrix Multiplication Function clblasSgemm failed");

	err=clWaitForEvents(1,&Kernel_event);
	checkError(err, "Wait for events failed");

	/* Fetch results of calculations. */
	cl_event read_event=NULL;
	err = clEnqueueReadBuffer( queue, bufC, CL_TRUE, 0,  m_Matrix1Row * m_Matrix2Col  * sizeof( float ), arrayFromPyRes, 0, NULL, &read_event );
	checkError(err, "Failed to read from Buffer");

	/* Wait for calculations to be finished. */
	err = clFinish(queue);
	checkError(err, "clFinish failed");

	write_time	=	get_event_exec_time(write_event[0])  + get_event_exec_time(write_event[1]);
	exec_time	=	get_event_exec_time(Kernel_event);
	read_time	=	get_event_exec_time(read_event);

	clReleaseEvent(write_event[0]);
	clReleaseEvent(write_event[1]);
	clReleaseEvent(Kernel_event);
	clReleaseEvent(read_event);

	return result;
}
//Boost library related declarations 
//The functions declared here can be used as 
//API's in python
BOOST_PYTHON_MODULE(MMUL_clBLAS_Lib)
{
	bn::array::set_module_and_type("numpy","ndarray");
	using namespace boost::python;
	Py_Initialize();
	numn::initialize();
	clblasSetup();
	class_<OpenClMMUL_clBLAS>("OpenClMMUL_clBLAS",init<>())
	.def("SetParam",&OpenClMMUL_clBLAS::SetParam)
	.def("doMMUL",&OpenClMMUL_clBLAS::doMMUL)
	.def("getWriteTime",&OpenClMMUL_clBLAS::getWriteTime)
	.def("getExeTime",&OpenClMMUL_clBLAS::getExeTime)
	.def("getReadTime",&OpenClMMUL_clBLAS::getReadTime);
}
