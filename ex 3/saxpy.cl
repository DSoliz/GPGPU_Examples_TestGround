__kernel
void saxpy(float a, __global const float* X, __global const float* Y, int n, __global float* Z)
{
	// Get the work-item's unique ID
	int idx = get_global_id(0);
	if (idx < n)
		Z[idx] = a * X[idx]  +  Y[idx];
}
