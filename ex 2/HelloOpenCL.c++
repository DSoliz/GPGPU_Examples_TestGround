// The OpenCL version of Hello, World

#include <iostream>
#include <string>

// OpenCL includes
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

const char* readSource(const char* fileName);

// A couple simple utility functions:
bool debug = false;
void checkStatus(std::string where, cl_int status, bool abortOnError)
{
	if (debug || (status != 0))
		std::cout << "Step " << where << ", status = " << status << '\n';
	if ((status != 0) && abortOnError)
		exit(1);
}

void reportVersion(cl_platform_id platform)
{
	// Get the version of OpenCL supported on this platform
	size_t strLength;
	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, nullptr, &strLength);
	char* version = new char[strLength+1];
	clGetPlatformInfo(platform, CL_PLATFORM_VERSION, strLength+1, version, &strLength);
	std::cout << version << '\n';
	delete [] version;
}

void showProgramBuildLog(cl_program pgm, cl_device_id dev)
{
	size_t size;
	clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
	char* log = new char[size+1];
	clGetProgramBuildInfo(pgm, dev, CL_PROGRAM_BUILD_LOG, size+1, log, nullptr);
	std::cout << "LOG:\n" << log << "\n\n";
	delete [] log;
}

// Typical OpenCL startup

// Some global state variables (These would be better packaged as
// instance variables of some class.)
// 1) Platforms
cl_uint numPlatforms = 0;
cl_platform_id* platforms = nullptr;
cl_platform_id curPlatform;
// 2) Devices
cl_uint numDevices = 0;
cl_device_id* devices = nullptr;

void typicalOpenCLProlog(cl_device_type desiredDeviceType)
{
	//-----------------------------------------------------
	// Discover and query the platforms
	//-----------------------------------------------------

	cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
	checkStatus("clGetPlatformIDs-0", status, true);
 
	platforms = new cl_platform_id[numPlatforms];
 
	status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
	checkStatus("clGetPlatformIDs-1", status, true);
	curPlatform = platforms[0];

	reportVersion(curPlatform);

	//-------------------------------------------------------------------------------
	// Discover and initialize the devices on the selected (current) platform
	//-------------------------------------------------------------------------------

	status = clGetDeviceIDs(curPlatform, desiredDeviceType, 0, nullptr, &numDevices);
	checkStatus("clGetDeviceIDs-0", status, true);

	devices = new cl_device_id[numDevices];

	status = clGetDeviceIDs(curPlatform, desiredDeviceType, numDevices, devices, nullptr);
	checkStatus("clGetDeviceIDs-1", status, true);
}

int main(int argc, char* argv[])
{
	int numDimsToUse = 1;
	if (argc > 1)
		numDimsToUse = atoi(argv[1]);

	typicalOpenCLProlog(CL_DEVICE_TYPE_DEFAULT);

	//-------------------------------------------------------------------
	// Create a context for all or some of the discovered devices
	//         (Here we are including all devices.)
	//-------------------------------------------------------------------

	cl_int status;
	cl_context context = clCreateContext(nullptr, numDevices, devices,
		nullptr, nullptr, &status);
	checkStatus("clCreateContext", status, true);

	//-------------------------------------------------------------
	// Create a command queue for ONE device in the context
	//         (There is one queue per device per context.)
	//-------------------------------------------------------------

	cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], 
		0, &status);
	checkStatus("clCreateCommandQueue", status, true);

	//-----------------------------------------------------
	// Create, compile, and link the program
	//----------------------------------------------------- 

	const char* programSource[] = { readSource("HelloOpenCL.cl") };
	cl_program program = clCreateProgramWithSource(context, 
		1, programSource, nullptr, &status);
	checkStatus("clCreateProgramWithSource", status, true);

	status = clBuildProgram(program, numDevices, devices, 
		nullptr, nullptr, nullptr);
	checkStatus("clBuildProgram", status, true);

	//-----------------------------------------------------------
	// Create a kernel from one of the __kernel functions
	//         in the source that was built.
	//-----------------------------------------------------------

	cl_kernel kernel = clCreateKernel(program, "helloOpenCL", &status);

	//-----------------------------------------------------
	// Configure the work-item structure
	//----------------------------------------------------- 

	size_t globalWorkSize[] = { 64, 32, 32 };    
	size_t localWorkSize[] = { 8, 8, 4 };    
	if (numDimsToUse == 1)
		localWorkSize[0] = 32;
	else if (numDimsToUse == 2)
		localWorkSize[0] = localWorkSize[1] = 16;

	//-----------------------------------------------------
	// Enqueue the kernel for execution
	//----------------------------------------------------- 

	status = clEnqueueNDRangeKernel(cmdQueue, 
		kernel, numDimsToUse, nullptr, globalWorkSize, 
		localWorkSize, 0, nullptr, nullptr);
	checkStatus("clEnqueueNDRangeKernel", status, true);

	// block until all commands have finished execution
	clFinish(cmdQueue);

	//-----------------------------------------------------
	// Release OpenCL resources
	//----------------------------------------------------- 

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseContext(context);

	// Free host resources
	delete [] platforms;
	delete [] devices;

	return 0;
}
