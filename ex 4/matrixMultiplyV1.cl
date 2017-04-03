#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
void matrixMultiply(__global double* A, __global double* B, __global double* C, int N)
{
	// Get the work-item's unique ID
	int col = get_global_id(0);
	int row = get_global_id(1);
	if ((row < N) && (col < N))
	{
		double sum = 0.0;
		for (int k=0 ; k<N ; k++)
			sum += A[row*N + k] * B[k*N + col];
		C[row*N + col] = sum;
	}
}
