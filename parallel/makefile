CC = mpicxx
NVCC = nvcc
CFLAGS = -fopenmp
NVFLAGS = -gencode arch=compute_61,code=sm_61 -Xcompiler -fopenmp
EXECUTABLE = run

${EXECUTABLE}: common.o mpi.o score_calculation.o main.o
	${CC} ${CFLAGS} $^ -o $@ /usr/local/cuda/lib64/libcudart_static.a -ldl -lrt -lgomp

common.o: ../common/common.c
	${CC} ${CFLAGS} -c $< -o $@

mpi.o: mpi/helper.c
	${CC} ${CFLAGS} -c $< -o $@

score_calculation.o: cuda/score_calculation.cu
	${NVCC} ${NVFLAGS} -c $< -o $@

main.o: main.c
	${CC} ${CFLAGS} -c $< -o $@

clean:
	rm -f *.o ${EXECUTABLE}