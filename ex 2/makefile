make: HelloOpenCL.o readSource.o
	g++ HelloOpenCL.o readSource.o -o HelloOpenCL -framework OpenCL

HelloOpenCL.o: HelloOpenCL.c++
	g++ -std=c++11 -c HelloOpenCL.c++

readSource.o: readSource.c++
	g++ -std=c++11 -c readSource.c++

mac: macHelloOpenCL.o macreadSource.o
	g++ macHelloOpenCL.o macreadSource.o -o macHelloOpenCL -framework OpenCL

macHelloOpenCL.o: HelloOpenCL.c++
	g++ -std=c++11 -c HelloOpenCL.c++ -o macHelloOpenCL.o

macreadSource.o: readSource.c++
	g++ -std=c++11 -c readSource.c++ -o macreadSource.o

clean:
	rm ./*.o
	rm ./HelloOpenCL
