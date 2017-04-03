__kernel
void vecadd(__global float *A, __global float *B, __global float *C)
{
	// Get the work-item's unique ID
	int idx = get_global_id(0);

	// Add the corresponding locations of
	//  'A' and 'B', and store the result in 'C'.
	C[2*idx] = cos(A[idx]);
	C[2*idx+1] = sin(B[idx]);
}
