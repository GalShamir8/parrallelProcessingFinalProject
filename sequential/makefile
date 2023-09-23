CC = gcc
EXECUTABLE = run

${EXECUTABLE}: common.o main.o
	${CC} $^ -o $@

common.o: ../common/common.c
	${CC} -c $< -o $@
	
main.o: main.c
	${CC} -c $< -o $@

clean:
	rm -f *.o ${EXECUTABLE}