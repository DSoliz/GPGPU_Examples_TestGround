all: daxpy saxpy

daxpy: daxpy.o readSource.o
	g++ daxpy.o readSource.o -o daxpy -lOpenCL

daxpy.o: daxpy.c++
	g++ -c daxpy.c++

saxpy: saxpy.o readSource.o
	g++ saxpy.o readSource.o -o saxpy -lOpenCL

saxpy.o: saxpy.c++
	g++ -c saxpy.c++

readSource.o: readSource.c++
	g++ -c readSource.c++

clean:
	rm ./*.o
	rm daxpy
	rm saxpy

macall: macdaxpy macsaxpy

macdaxpy: macdaxpy.o macreadSource.o
	g++ macdaxpy.o macreadSource.o -o macdaxpy -framework OpenCL

macsaxpy: macsaxpy.o macreadSource.o
	g++ macsaxpy.o macreadSource.o -o macsaxpy -framework OpenCL

macdaxpy.o: daxpy.c++
	g++ -std=c++11 -c daxpy.c++ -o macdaxpy.o

macsaxpy.o: saxpy.c++
	g++ -std=c++11 -c saxpy.c++ -o macsaxpy.o

macreadSource.o: readSource.c++
	g++ -std=c++11 -c readSource.c++ -o macreadSource.o

macclean:
	rm ./*.o
	rm macdaxpy
	rm macsaxpy
