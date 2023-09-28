CC = mpicc
NVCC = nvcc
CFLAGS = -fopenmp
NVFLAGS = -gencode arch=compute_61,code=sm_61
EXECUTABLE = run

${EXECUTABLE}: common.o mpi.o scoreCalculation.o main.o
	${CC} $^ -o $@ /usr/local/cuda-11.0/lib64/libcudart_static.a -ldl -lrt

common.o: ../common/common.c
	${CC} -c $< -o $@

mpi.o: mpi/helper.c
	${CC} -c $< -o $@

scoreCalculation.o: cuda/scoreCalculation.cu
	${NVCC} -c ${NVFLAGS} $< -o $@

main.o: main.c
	${CC} -c $< -o $@

clean:
	rm -f *.o ${EXECUTABLE}