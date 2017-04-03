// This program uses OpenCL to multiply two double precision matrices:
// C = A * B

// System includes
#include <iostream>
#include <string>
#include <string.h>
#include <math.h>

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

// Return value is device index to use; -1 ==> no available devices
int typicalOpenCLProlog(cl_device_type desiredDeviceType)
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
	if (numPlatforms > 1)
	{
		size_t platformNameSize = 0;
		clGetPlatformInfo(curPlatform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize);
		char* name = new char[platformNameSize+1];
		clGetPlatformInfo(curPlatform, CL_PLATFORM_NAME, platformNameSize+1, name, nullptr);
		std::cout << "Found " << numPlatforms << " platforms. Arbitrarily using: " << name << '\n';
		delete [] name;
	}

	reportVersion(curPlatform);

	//----------------------------------------------------------
	// Discover and initialize the devices on a platform
	//----------------------------------------------------------

	status = clGetDeviceIDs(curPlatform, desiredDeviceType, 0, nullptr, &numDevices);
	checkStatus("clGetDeviceIDs-0", status, true);
	if (numDevices <= 0)
	{
		std::cout << "No devices on platform!\n";
		return -1;
	}

	devices = new cl_device_id[numDevices];

	status = clGetDeviceIDs(curPlatform, desiredDeviceType, numDevices, devices, nullptr);
	checkStatus("clGetDeviceIDs-1", status, true);
	// Find a device that supports double precision arithmetic
	int* possibleDevs = new int[numDevices];
	int nPossibleDevs = 0;
	std::cout << "\nLooking for a device that supports double precision...\n";
	for (int idx=0 ; idx<numDevices ; idx++)
	{
		size_t extLength;
		clGetDeviceInfo(devices[idx], CL_DEVICE_EXTENSIONS, 0, nullptr, &extLength);
		char* extString = new char[extLength+1];
		clGetDeviceInfo(devices[idx], CL_DEVICE_EXTENSIONS, extLength+1, extString, nullptr);
		const char* fp64 = strstr(extString, "cl_khr_fp64");
		if (fp64 != nullptr) // this device supports double precision
			possibleDevs[nPossibleDevs++] = idx;
	}
	if (nPossibleDevs == 0)
	{
		std::cerr << "\nNo device supports double precision.\n";
		return -1;
	}
	size_t nameLength;
	for (int i=0 ; i<nPossibleDevs ; i++)
	{
		clGetDeviceInfo(devices[possibleDevs[i]], CL_DEVICE_NAME, 0, nullptr, &nameLength);
		char* name = new char[nameLength+1];
		clGetDeviceInfo(devices[possibleDevs[i]], CL_DEVICE_NAME, nameLength+1, name, nullptr);
		std::cout << "Device " << i << ": [" << name << "] supports double precision.\n";
	}
	if (nPossibleDevs == 1)
	{
		std::cout << "\nNo other device in the requested device category supports double precision.\n"
		          << "You may want to try the -a command line option to see if there are others.\n"
		          << "For now, I will use the one I found.\n";
		return possibleDevs[0];
	}
	int devIndex = -1;
	while ((devIndex < 0) || (devIndex >= nPossibleDevs))
	{
		std::cout << "Which device do you want to use? ";
		std::cin >> devIndex;
	}
	return possibleDevs[devIndex];
}

void doTheKernelLaunch(cl_device_id dev, double* A, double* B, double* C, size_t N)
{
	//------------------------------------------------------------------------
	// Create a context for some or all of the devices on the platform
	// (Here we are including all devices.)
	//------------------------------------------------------------------------

	cl_int status;
	cl_context context = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &status);
	checkStatus("clCreateContext", status, true);

	//-------------------------------------------------------------
	// Create a command queue for one device in the context
	// (There is one queue per device per context.)
	//-------------------------------------------------------------

	cl_command_queue cmdQueue = clCreateCommandQueue(context, dev, 0, &status);
	checkStatus("clCreateCommandQueue", status, true);

	//----------------------------------------------------------
	// Create device buffers associated with the context
	//----------------------------------------------------------

	size_t datasize = N * N * sizeof(double);

	cl_mem d_A = clCreateBuffer( // Input array on the device
		context, CL_MEM_READ_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-A", status, true);

	cl_mem d_B = clCreateBuffer( // Input array on the device
		context, CL_MEM_READ_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-B", status, true);

	cl_mem d_C = clCreateBuffer( // Output array on the device
		context, CL_MEM_WRITE_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-C", status, true);

	//-----------------------------------------------------
	// Use the command queue to encode requests to
	//         write host data to the device buffers
	//----------------------------------------------------- 

	status = clEnqueueWriteBuffer(cmdQueue, 
		d_A, CL_FALSE, 0, datasize,                         
		A, 0, nullptr, nullptr);
	checkStatus("clEnqueueWriteBuffer-A", status, true);

	status = clEnqueueWriteBuffer(cmdQueue, 
		d_B, CL_FALSE, 0, datasize,                                  
		B, 0, nullptr, nullptr);
	checkStatus("clEnqueueWriteBuffer-B", status, true);

	//-----------------------------------------------------
	// Create, compile, and link the program
	//----------------------------------------------------- 

	const char* programSource[] = { readSource("matrixMultiplyV1.cl") };
	cl_program program = clCreateProgramWithSource(context, 
		1, programSource, nullptr, &status);
	checkStatus("clCreateProgramWithSource", status, true);

	status = clBuildProgram(program, 1, &dev, nullptr, nullptr, nullptr);
	checkStatus("clBuildProgram", status, false);
	if (status != 0)
		showProgramBuildLog(program, dev);

	//----------------------------------------------------------------------
	// Create a kernel using a "__kernel" function in the ".cl" file
	//----------------------------------------------------------------------

	cl_kernel kernel = clCreateKernel(program, "matrixMultiply", &status);

	//-----------------------------------------------------
	// Set the kernel arguments
	//----------------------------------------------------- 

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
	checkStatus("clSetKernelArg-A", status, true);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
	checkStatus("clSetKernelArg-B", status, true);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
	checkStatus("clSetKernelArg-C", status, true);
	status = clSetKernelArg(kernel, 3, sizeof(int), &N);
	checkStatus("clSetKernelArg-N", status, true);

	//-----------------------------------------------------
	// Configure the work-item structure
	//----------------------------------------------------- 

	size_t globalWorkSize[] = { N, N };    

	//-----------------------------------------------------
	// Enqueue the kernel for execution
	//----------------------------------------------------- 

	status = clEnqueueNDRangeKernel(cmdQueue, 
		kernel, 2, nullptr, globalWorkSize, 
		nullptr, 0, nullptr, nullptr);
	checkStatus("clEnqueueNDRangeKernel", status, true);

	//-----------------------------------------------------
	// Read the output buffer back to the host
	//----------------------------------------------------- 

	clEnqueueReadBuffer(cmdQueue, 
		d_C, CL_TRUE, 0, datasize, 
		C, 0, nullptr, nullptr);

	//-----------------------------------------------------
	// Release OpenCL resources
	//----------------------------------------------------- 

	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);
	clReleaseContext(context);

	// Free host resources
	delete [] platforms;
	delete [] devices;
}

double* do_MatrixMultiply(cl_device_id dev, size_t N)
{
	double* X = new double[N*N];
	double* Y = new double[N*N];
	double* Z = new double[N*N];
	for (int row=0 ; row<N ; row++)
		for (int col=0 ; col<N ; col++)
		{
			// make X be 2*I
			X[row*N + col] = (row == col) ? 2.0 : 0.0;
			Y[row*N + col] = 17.5;
		}
	doTheKernelLaunch(dev, X, Y, Z, N);

	delete [] X;
	delete [] Y;
	return Z;
}

void print(std::string label, double* M, size_t N)
{
	std::cout << label << ":\n";
	for (int row=0 ; row<N ; row++)
	{
		for (int col=0 ; col<N ; col++)
		{
			std::cout << M[row*N + col] << " ";
		}
		std::cout << '\n';
	}
}

int main(int argc, char* argv[])
{
	cl_device_type devType = CL_DEVICE_TYPE_DEFAULT;
	size_t N = 20;
	if (argc > 1)
	{
		for (int i=1 ; i<argc ; i++)
		{
			if (strcmp("-debug", argv[i]) == 0)
				debug = true;
			else if (argv[i][0] == '-')
			{
				switch (argv[i][1])
				{
					case 'a':
						devType = CL_DEVICE_TYPE_ALL;
						break;
					case 'c':
						devType = CL_DEVICE_TYPE_CPU;
						break;
					case 'g':
						devType = CL_DEVICE_TYPE_GPU;
						break;
				}
			}
			else
				N = atoi(argv[i]);
		}
	}
	int devIndex = typicalOpenCLProlog(devType);
	if (devIndex >= 0)
	{
		double* C = do_MatrixMultiply(devices[devIndex], N);
		print("The product is", C, N);
		delete [] C;
	}

	return 0;
}
