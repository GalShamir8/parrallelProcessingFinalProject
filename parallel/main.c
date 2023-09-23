#include "main.h"

int main(int argc, char** argv) {
  int** scoreMat;
  char** sequences;
  int numOfSequences, rank, numOfProc;
  mpiInit(&argc, argv, &numOfProc, &rank);
  if (rank == ROOT_PROCESS_RANK)
    masterProcess(numOfProc, argc, argv);
  else
    workerProcess(rank);
}

void workerProcess(int rank) {
  MPI_Datatype initializePayload;
  int initBlockLengths[2] = { SCORE_TABLE_ROWS * SCORE_TABLE_COLS, MAX_CHARACTERS_SEQ1 };
  createInitializePayloadType(&initializePayload, initBlockLengths);

  InitializePayload initBuffer;
  MPI_Bcast(&initBuffer, 1, initializePayload, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
  
  WorkerPayload receivedBuffer;
  MPI_Datatype workerPayload;
  int workerBlockLengths[2] = { MAX_CHARACTERS_SEQ, 1 };
  createWorkerPayloadType(&workerPayload, workerBlockLengths);

  ResultPayload sendBuffer;
  MPI_Datatype resultPayload;
  int resultBlockLengths[5] = { 1, 1, 1, 1, MAX_CHARACTERS_SEQ };
  createWorkerPayloadType(&workerPayload, resultBlockLengths);

  MPI_Status status;
  int tag;

  do {
    MPI_Recv(
      &receivedBuffer, 1, workerPayload, ROOT_PROCESS_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status
    );
    tag = status.MPI_TAG;
    if (tag == WORK) {
      strcpy(&sendBuffer.sequence, &receivedBuffer.sequence);
      sendBuffer.index = receivedBuffer.index;
      calculateMaxAlignmentScore(initBuffer.mainSequence, receivedBuffer.sequence, &sendBuffer, initBuffer.scoreMat);
      MPI_Send(
        &sendBuffer, 1, resultPayload, ROOT_PROCESS_RANK, FINISH, MPI_COMM_WORLD
      );
    }
  } while(tag != STOP);
}

void masterProcess(int numOfProc, int argc, char** argv) {
  int** scoreMat;
  char** sequences;
  int numOfSequences, taskSent = 0, taskDone = 0;

  switch(argc) {
    case 2:
      scoreMat = buildScoreTable(NULL);
      break;
    case 3:
      scoreMat = buildScoreTable(argv[2]);
      break;
    default:
      printf("Usage: missig input");
      exit(1);
  }

  sequences = readSeqInput(argv[1], &numOfSequences);
  #ifdef DEBUG
    printMatrix(scoreMat, SCORE_TABLE_ROWS, SCORE_TABLE_COLS);
    printSequeneces(sequences, numOfSequences);
  #endif

  MPI_Status status;

  MPI_Datatype initializePayload;
  int initBlockLengths[2] = { SCORE_TABLE_ROWS * SCORE_TABLE_COLS, MAX_CHARACTERS_SEQ1 };
  createInitializePayloadType(&initializePayload, initBlockLengths);

  MPI_Bcast(&(InitializePayload){ scoreMat, sequences[0] }, 1, initializePayload, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
  
  MPI_Datatype workerPayload;
  int workerBlockLengths[2] = { MAX_CHARACTERS_SEQ, 1 };
  createWorkerPayloadType(&workerPayload, workerBlockLengths);

  MPI_Datatype resultPayload;
  int resultBlockLengths[5] = { 1, 1, 1, 1, MAX_CHARACTERS_SEQ };
  createWorkerPayloadType(&workerPayload, resultBlockLengths);
  

  for (int rank = 1; rank < numOfProc; rank++) {
    if (rank <= numOfSequences) {
      MPI_Send(
        &(WorkerPayload){ sequences[rank], rank }, 1, workerPayload, rank, WORK, MPI_COMM_WORLD
      );
      taskSent++;
    }
  }

  ResultPayload results[numOfSequences];
  ResultPayload resultBuffer;

  while(taskDone <= numOfSequences) {  
    MPI_Recv(&resultBuffer, 1, resultPayload, MPI_ANY_SOURCE, FINISH, MPI_COMM_WORLD, &status);
    results[resultBuffer.index] = resultBuffer;
    taskDone++;
    if (taskSent < numOfSequences) {
      MPI_Send(
        &(WorkerPayload){ sequences[taskSent], taskSent }, 1, workerPayload, status.MPI_SOURCE, WORK, MPI_COMM_WORLD
      );
      taskSent++;
    }
  }

  // notify all workers to stop working
  for (int rank = 1; rank < numOfProc; rank++) {
    MPI_Send(&(WorkerPayload){ "", 0 }, 0, workerPayload, rank, STOP, MPI_COMM_WORLD);
  }

  printResults(results, numOfSequences);


  MPI_Type_free(&initializePayload);
  MPI_Type_free(&workerPayload);
  MPI_Type_free(&resultPayload);
  free(scoreMat);
  free(sequences);
  MPI_Finalize();
}

void printResults(ResultPayload results[], int size) {
  for (int i = 0; i < size; i++) {
    printf(
      "%s highest alignment score = %d, offset = %d, k = %d\n",
      results[i].sequence, results[i].maxScore,
      results[i].offset, results[i].k
    );
  }
}