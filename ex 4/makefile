all: matrixMultiplyV1

matrixMultiplyV1: matrixMultiplyV1.o readSource.o
	g++ matrixMultiplyV1.o readSource.o -o matrixMultiplyV1 -framework OpenCL

matrixMultiplyV1.o: matrixMultiplyV1.c++
	g++ -std=c++11 -c matrixMultiplyV1.c++

readSource.o: readSource.c++
	g++ -std=c++11 -c readSource.c++
