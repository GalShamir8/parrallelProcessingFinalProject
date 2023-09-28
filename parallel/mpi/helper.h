#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../../common/common.h"

void mpiInit(int *argc, char **argv, int *numOfProc, int *rank);
void createInitializePayloadType(MPI_Datatype *type, int* blockLengths);
void createWorkerPayloadType(MPI_Datatype *type, int* blockLengths);
void createResultPayloadType(MPI_Datatype *type, int* blockLengths);