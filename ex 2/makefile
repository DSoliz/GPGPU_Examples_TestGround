HelloOpenCL: HelloOpenCL.o readSource.o
	g++ HelloOpenCL.o readSource.o -o HelloOpenCL -framework OpenCL

HelloOpenCL.o: HelloOpenCL.c++
	g++ -std=c++11 -c HelloOpenCL.c++

readSource.o: readSource.c++
	g++ -std=c++11 -c readSource.c++
