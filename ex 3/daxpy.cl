#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
void daxpy(double a, __global const double* X, __global const double* Y, int n, __global double* Z)
{
	// Get the work-item's unique ID
	int idx = get_global_id(0);
	if (idx < n)
		Z[idx] = a * X[idx]  +  Y[idx];
}
