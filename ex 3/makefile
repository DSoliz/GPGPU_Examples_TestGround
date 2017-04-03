all: daxpy saxpy

daxpy: daxpy.o readSource.o
	g++ daxpy.o readSource.o -o daxpy -framework OpenCL

saxpy: saxpy.o readSource.o
	g++ saxpy.o readSource.o -o saxpy -framework OpenCL

daxpy.o: daxpy.c++
	g++ -std=c++11 -c daxpy.c++

saxpy.o: saxpy.c++
	g++ -std=c++11 -c saxpy.c++

readSource.o: readSource.c++
	g++ -std=c++11 -c readSource.c++
