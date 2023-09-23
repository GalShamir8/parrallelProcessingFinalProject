#include "helper.h"

void mpiInit(int *argc, char **argv, int *numOfProc, int *rank) {
  MPI_Init(argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, numOfProc);
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
}

void createWorkerPayloadType(MPI_Datatype *type, int* blockLengths) {  
  MPI_Aint displacements[2];
  MPI_Datatype types[2] = { MPI_CHAR, MPI_INT };

  WorkerPayload tmpType;
  MPI_Aint base_address;
  MPI_Get_address(&tmpType, &base_address);
  MPI_Get_address(&tmpType.sequence, &displacements[0]);
  MPI_Get_address(&tmpType.index, &displacements[1]);

  for (int i = 0; i < 2; i++)
    displacements[i] = MPI_Aint_diff(displacements[i], base_address);

  MPI_Type_create_struct(2, blockLengths, displacements, types, type);
  MPI_Type_commit(type);
}

void createInitializePayloadType(MPI_Datatype *type, int* blockLengths) {
  MPI_Aint displacements[2];
  MPI_Datatype types[2] = { MPI_CHAR, MPI_CHAR };

  InitializePayload tmpType;
  MPI_Aint base_address;
  MPI_Get_address(&tmpType, &base_address);
  MPI_Get_address(&tmpType.scoreMat, &displacements[0]);
  MPI_Get_address(&tmpType.mainSequence, &displacements[1]);

  for (int i = 0; i < 2; i++)
    displacements[i] = MPI_Aint_diff(displacements[i], base_address);

  MPI_Type_create_struct(2, blockLengths, displacements, types, type);
  MPI_Type_commit(type);
}

void createResultPayloadType(MPI_Datatype *type, int* blockLengths) {
  MPI_Aint displacements[5];
  MPI_Datatype types[5] = { MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_CHAR };

  ResultPayload tmpType;
  MPI_Aint base_address;
  MPI_Get_address(&tmpType, &base_address);
  MPI_Get_address(&tmpType.maxScore, &displacements[0]);
  MPI_Get_address(&tmpType.offset, &displacements[1]);
  MPI_Get_address(&tmpType.k, &displacements[2]);
  MPI_Get_address(&tmpType.index, &displacements[3]);
  MPI_Get_address(&tmpType.sequence, &displacements[4]);

  for (int i = 0; i < 5; i++)
    displacements[i] = MPI_Aint_diff(displacements[i], base_address);

  MPI_Type_create_struct(5, blockLengths, displacements, types, type);
  MPI_Type_commit(type);
}