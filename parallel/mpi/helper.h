#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../../common/common.h"

typedef struct {
  int scoreMat[SCORE_TABLE_ROWS][SCORE_TABLE_COLS];
  char mainSequence[MAX_CHARACTERS_SEQ1];
} InitializePayload;

typedef struct {
  char sequence[MAX_CHARACTERS_SEQ];
  int index;
} WorkerPayload;

typedef struct {
  int maxScore;
  int offset;
  int k;
  int index;
  char sequence[MAX_CHARACTERS_SEQ];
} ResultPayload;

void mpiInit(int *argc, char **argv, int *numOfProc, int *rank);
void createInitializePayloadType(MPI_Datatype *type, int* blockLengths);
void createWorkerPayloadType(MPI_Datatype *type, int* blockLengths);
void createResultPayloadType(MPI_Datatype *type, int* blockLengths);