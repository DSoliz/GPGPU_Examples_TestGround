// SimpleOpenCL.c++: A very basic OpenCL program that shows the major
//                   steps required in an OpenCL program. The steps
//                   are ordered in such a way to show the extra steps
//                   required by OpenCL to query its computational
//                   environment and prepare discovered devices to
//                   execute code.
//
// All the code is in "main" - a terrible structure from a software
// engineering perspective, but done to emphasize the required flow
// of the program. An OpenCL program ought to be structured using,
// for example, good OOD practices and modularity to get reusable
// modules.

#include <iostream>
#include <string>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

const char* readSource(const char* fileName);

bool debug = false;
void checkStatus(std::string where, cl_int status)
{
	if (debug || (status != 0))
		std::cout << "Step " << where << ", status = " << status << '\n';
	if (status != 0)
		exit(1);
}

int main(int argc, char* argv[])
{
	// OPTIONAL: Look for command line arguments that specify the
	//		   types of devices for which I might want to look.
	cl_device_type devType = CL_DEVICE_TYPE_DEFAULT;
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
		}
	}

	// The GPU kernel I want to launch will create a lookup table
	// of trig functions for given angles. Specifically, given
	// arrays A and B, each of which hold angles, I want the GPU
	// code to fill the array C such that:
	//	C[2*i]   = cos(A[i]);
	//	C[2*i+1] = sin(B[i]);
	
	const int NUM_ELEMENTS = 16384;   
	// Allocate storage for the arrays A and B:
	float *A = new float[NUM_ELEMENTS];	 // Input array
	float *B = new float[NUM_ELEMENTS];	 // Input array
	// Allocate storage for array C (must be twice the size of A and B):
	float *C = new float[2 * NUM_ELEMENTS]; // Output array
	
	// Initialize the A and B arrays:
	float dTheta = M_PI / static_cast<float>(NUM_ELEMENTS-1);
	float theta = 0.0;
	for(int i = 0; i < NUM_ELEMENTS; i++)
	{
		A[i] = theta;
		B[i] = M_PI - theta;
		theta += dTheta;
	}

	// Now on to the OpenCL stuff:

	/*
	 * The steps shown below are listed in a numerical order. This order is
	 * one possible order that satisfies the required partial ordering. I
	 * picked this ordering because it makes it easier to compare CUDA and
	 * OpenCL implementations. Future OpenCL programs we study will perform
	 * some of these steps in a different order for a variety of reasons.
	 */

	// ----------------------------------------------------
	// STEP 1: Discover and initialize the platforms
	// ----------------------------------------------------
	
	cl_uint numPlatforms = 0;
	cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
	checkStatus("clGetPlatformIDs-0", status);
 
	cl_platform_id* platforms = new cl_platform_id[numPlatforms];
 
	status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
	checkStatus("clGetPlatformIDs-1", status);

	// ------------------------------------------------------------------
	// STEP 2: Discover and initialize the devices on a specific platform
	//         Here we arbitrarily use platforms[0], which is probably
	//         the only one.
	// ------------------------------------------------------------------
	
	cl_uint numDevices = 0;
	// See the code parsing argc-argv above for "devType"
	status = clGetDeviceIDs(platforms[0], devType, 0, nullptr, &numDevices);
	checkStatus("clGetDeviceIDs-0", status);

	cl_device_id* devices = new cl_device_id[numDevices];

	status = clGetDeviceIDs(platforms[0], devType, numDevices, devices, nullptr);
	checkStatus("clGetDeviceIDs-1", status);

	// ------------------------------------------------------------------------------
	// STEP 3: Create a context for one or more devices. Here we arbitrarily
	//         create a context containing all devices found in STEP 2.
	//         (Alternatively, could just include one or more select devices.)
	//
	//         Contexts manage portions of the OpenCL state. Most notably for
	//         now, buffers are created in a context (see STEP 7 below) and are
	//         accessible to all devices belonging to that context.
	//
	// ALSO NOTE: A device can belong to more than one context, however a device
	//            command queue (see STEP 4) is associated with exactly one device,
	//            and it belongs to exactly one context. Event synchronization (an
	//            advanced topic we will study later) is only possible between
	//            queues belonging to the same context.
	// ------------------------------------------------------------------------------
	
	cl_context context = clCreateContext(nullptr, numDevices, devices,
		nullptr, nullptr, &status);
	checkStatus("clCreateContext", status);

	// --------------------------------------------------------------
	// STEP 4: Create a command queue (must explicitly create a queue
	//         for each desired device)
	// --------------------------------------------------------------
	
	cl_command_queue cmdQueue = clCreateCommandQueue(context, devices[0], 
		0, &status);
	checkStatus("clCreateCommandQueue", status);

	// ----------------------------------------------------
	// STEP 5: Create and compile the program
	// ----------------------------------------------------
	 
	const char* programSource[] = { readSource("SimpleOpenCL.cl") };
	cl_program program = clCreateProgramWithSource(context, 
		1, programSource, nullptr, &status);
	checkStatus("clCreateProgramWithSource", status);

	status = clBuildProgram(program, numDevices, devices, 
		nullptr, nullptr, nullptr);
	checkStatus("clBuildProgram", status);
   
	// ----------------------------------------------------
	// STEP 6: Create the CPU-side kernel reference
	// ----------------------------------------------------

	cl_kernel kernel = clCreateKernel(program, "vecadd", &status);
	checkStatus("clCreateKernel", status);

	// ******************************************************************
	// ******************************************************************
	// ******** The corresponding cpu-side CUDA processing would ********
	// ******** start roughly here.                              ********
	// ******************************************************************
	// ******************************************************************

	// ---------------------------------------------------------------------
	// STEP 7: Create device buffers (shared by all devices in a context;
	//         not initially guaranteed to be on any device in the context.)
	// ---------------------------------------------------------------------

	size_t datasize = NUM_ELEMENTS * sizeof(float);

	cl_mem bufferA = clCreateBuffer( // Input array on the device
		context, CL_MEM_READ_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-A", status);

	cl_mem bufferB = clCreateBuffer( // Input array on the device
		context, CL_MEM_READ_ONLY, datasize, nullptr, &status);
	checkStatus("clCreateBuffer-B", status);

	cl_mem bufferC = clCreateBuffer( // Output array on the device
		context, CL_MEM_WRITE_ONLY, 2*datasize, nullptr, &status);
	checkStatus("clCreateBuffer-C", status);
	
	// ----------------------------------------------------
	// STEP 8: Write host data to device buffers
	// ----------------------------------------------------
	
	status = clEnqueueWriteBuffer(cmdQueue, 
		bufferA, CL_FALSE, 0, datasize,						 
		A, 0, nullptr, nullptr);
	checkStatus("clEnqueueWriteBuffer-A", status);
	
	status = clEnqueueWriteBuffer(cmdQueue, 
		bufferB, CL_FALSE, 0, datasize,								  
		B, 0, nullptr, nullptr);
	checkStatus("clEnqueueWriteBuffer-B", status);

	// ----------------------------------------------------
	// STEP 9: Set the kernel arguments
	// ----------------------------------------------------
	
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
	checkStatus("clSetKernelArg-0", status);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
	checkStatus("clSetKernelArg-1", status);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
	checkStatus("clSetKernelArg-2", status);

	// ----------------------------------------------------
	// STEP 10: Configure the work-item structure
	// ----------------------------------------------------
	
	size_t globalWorkSize[1] = { NUM_ELEMENTS };	

	// ----------------------------------------------------
	// STEP 11: Enqueue the kernel for execution
	// ----------------------------------------------------
	
	status = clEnqueueNDRangeKernel(cmdQueue, 
		kernel, 1, nullptr, globalWorkSize, 
		nullptr, 0, nullptr, nullptr);
	checkStatus("clEnqueueNDRangeKernel", status);

	// ----------------------------------------------------
	// STEP 12: Read the output buffer back to the host
	// ----------------------------------------------------
	
	status = clEnqueueReadBuffer(cmdQueue, 
		bufferC, CL_TRUE, 0, 2*datasize, 
		C, 0, nullptr, nullptr);
	checkStatus("clEnqueueReadBuffer", status);

	// Sanity Check: Did I get expected results?`
	int nDiffs = 0;
	float maxDiff = 0.0;
	for(int i = 0; i < NUM_ELEMENTS; i++)
	{
		float diff = fabs(C[2*i] - cos(A[i]));
		if(diff != 0.0)
		{
			nDiffs++;
			if (diff > maxDiff)
				maxDiff = diff;
		}
		diff = fabs(C[2*i+1] - sin(B[i]));
		if (diff != 0.0)
		{
			nDiffs++;
			if (diff > maxDiff)
				maxDiff = diff;
		}
	}
	// I will expect some differences - but very small ones - since the CPU sin and
	// cos functions return double, but the GPU code we invoked is single precision.
	std::cout << "There were " << NUM_ELEMENTS << " elements for a total of "
	          << (2*NUM_ELEMENTS) << " possible differences.\n";
	std::cout << "There were " << nDiffs << " differences, maxDiff = " << maxDiff << '\n';

	// ----------------------------------------------------
	// STEP 13: Release OpenCL resources
	// ----------------------------------------------------
	
	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseContext(context);

	// Free host resources
	delete [] A;
	delete [] B;
	delete [] C;
	delete [] platforms;
	delete [] devices;

	return 0;
}
