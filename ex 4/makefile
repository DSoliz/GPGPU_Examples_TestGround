make: matrixMultiplyV1

matrixMultiplyV1: matrixMultiplyV1.o readSource.o
	g++ matrixMultiplyV1.o readSource.o -o matrixMultiplyV1 -lOpenCL

matrixMultiplyV1.o: matrixMultiplyV1.c++
	g++ -c matrixMultiplyV1.c++

readSource.o: readSource.c++
	g++ -c readSource.c++

mac: macmatrixMultiplyV1

macmatrixMultiplyV1: macmatrixMultiplyV1.o macreadSource.o
	g++ macmatrixMultiplyV1.o macreadSource.o -o matrixMultiplyV1 -framework OpenCL

macmatrixMultiplyV1.o: matrixMultiplyV1.c++
	g++ -std=c++11 -c matrixMultiplyV1.c++ -o macmatrixMultiplyV1.o

macreadSource.o: readSource.c++
	g++ -std=c++11 -c readSource.c++ -o macreadSource.o

clean:
	rm ./*.o
	rm ./matrixMultiplyV1
