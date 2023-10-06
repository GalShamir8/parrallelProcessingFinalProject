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
  printf("in worker process: %d\n", rank);
  MPI_Datatype initializePayload;
  int initBlockLengths[2] = { SCORE_TABLE_ROWS * SCORE_TABLE_COLS, MAX_CHARACTERS_SEQ1 };
  createInitializePayloadType(&initializePayload, initBlockLengths);

  InitializePayload initBuffer;
  initBuffer.scoreMat = (int*) malloc(SCORE_TABLE_ROWS * SCORE_TABLE_COLS * sizeof(int));
  initBuffer.mainSequence = (char*) malloc( MAX_CHARACTERS_SEQ1);
  MPI_Bcast(&initBuffer, 1, initializePayload, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
  printf("worker: %d recevied brodcast init, master sequence: %s print matrix:\n", rank, initBuffer.mainSequence);
  printMatrix1D(initBuffer.scoreMat, SCORE_TABLE_ROWS, SCORE_TABLE_COLS);
  
  WorkerPayload receivedBuffer;
  MPI_Datatype workerPayload;
  int workerBlockLengths[2] = { MAX_CHARACTERS_SEQ, 1 };
  createWorkerPayloadType(&workerPayload, workerBlockLengths);
  receivedBuffer.sequence = (char*) malloc(MAX_CHARACTERS_SEQ);

  ResultPayload sendBuffer;
  MPI_Datatype resultPayload;
  int resultBlockLengths[5] = { 1, 1, 1, 1, MAX_CHARACTERS_SEQ };
  createResultPayloadType(&resultPayload, resultBlockLengths);
  sendBuffer.sequence = (char*) malloc(MAX_CHARACTERS_SEQ);

  MPI_Status status;
  int tag;

  do {
    MPI_Recv(
      &receivedBuffer, 1, workerPayload, ROOT_PROCESS_RANK, MPI_ANY_TAG, MPI_COMM_WORLD, &status
    );
    printf("rank: %d, recevied new sequence: %s\n", rank, receivedBuffer.sequence);
    tag = status.MPI_TAG;
    if (tag == WORK) {
      strcpy(sendBuffer.sequence, receivedBuffer.sequence);
      printf("rank: %d, processing sequence: %s\n", rank, sendBuffer.sequence);
      sendBuffer.index = receivedBuffer.index;
      calculateMaxAlignmentScore(initBuffer.mainSequence, receivedBuffer.sequence, &sendBuffer, initBuffer.scoreMat);
      MPI_Send(
        &sendBuffer, 1, resultPayload, ROOT_PROCESS_RANK, FINISH, MPI_COMM_WORLD
      );
    }
  } while(tag != STOP);
}

void masterProcess(int numOfProc, int argc, char** argv) {
  printf("in master process\n");
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

  InitializePayload initSendBuffer;
  MPI_Datatype initializePayload;
  int initBlockLengths[2] = { SCORE_TABLE_ROWS * SCORE_TABLE_COLS, MAX_CHARACTERS_SEQ1 };
  createInitializePayloadType(&initializePayload, initBlockLengths);
  initSendBuffer.scoreMat = (int*) malloc(SCORE_TABLE_ROWS * SCORE_TABLE_COLS * sizeof(int));
  initSendBuffer.mainSequence = (char*) malloc( MAX_CHARACTERS_SEQ1);
  initSendBuffer.scoreMat = *scoreMat;

  for (int row = 0; row < SCORE_TABLE_ROWS; row++) {
      for (int col = 0; col < SCORE_TABLE_COLS; col++) {
          *(initSendBuffer.scoreMat + (row * SCORE_TABLE_COLS) + col) = scoreMat[row][col];
      }
  }
  strcpy(initSendBuffer.mainSequence, sequences[0]);
  printf("master sending brodcast message, main sequence: %s, prinit matrix:\n", initSendBuffer.mainSequence);
  printMatrix1D(initSendBuffer.scoreMat, SCORE_TABLE_ROWS, SCORE_TABLE_COLS);
  MPI_Bcast(&initSendBuffer, 1, initializePayload, ROOT_PROCESS_RANK, MPI_COMM_WORLD);
  
  WorkerPayload workerSendBuffer;
  MPI_Datatype workerPayload;
  int workerBlockLengths[2] = { MAX_CHARACTERS_SEQ, 1 };
  createWorkerPayloadType(&workerPayload, workerBlockLengths);
  workerSendBuffer.sequence = (char*) malloc(MAX_CHARACTERS_SEQ);

  ResultPayload results[numOfSequences];
  ResultPayload resultBuffer;
  MPI_Datatype resultPayload;
  int resultBlockLengths[5] = { 1, 1, 1, 1, MAX_CHARACTERS_SEQ };
  createResultPayloadType(&resultPayload, resultBlockLengths);
  
  resultBuffer.sequence = (char*) malloc(MAX_CHARACTERS_SEQ);
  for (int i = 0; i < numOfSequences; i++) {
    results[i].sequence = (char*) malloc(MAX_CHARACTERS_SEQ);
  }

  for (int rank = 1; rank < numOfProc; rank++) {
    if (rank <= numOfSequences) {
      workerSendBuffer.index = rank;
      strcpy(workerSendBuffer.sequence, sequences[rank]);
      printf("static loop sending %s to rank: %d, task sent: %d\n", workerSendBuffer.sequence, rank, taskSent);
      MPI_Send(
        &workerSendBuffer, 1, workerPayload, rank, WORK, MPI_COMM_WORLD
      );
      taskSent++;
    }
  }


  while(taskDone <= numOfSequences) {  
    printf("In dynaamic loop, task sent: %d\n", taskSent);

    MPI_Recv(&resultBuffer, 1, resultPayload, MPI_ANY_SOURCE, FINISH, MPI_COMM_WORLD, &status);
    printf("recevied result from worker: %d, sequence: %s, max score: %d, k: %d, offst: %d tasks done/sent: %d/%d\n",
    status.MPI_SOURCE, resultBuffer.sequence, resultBuffer.maxScore, resultBuffer.offset, resultBuffer.k, taskDone, taskSent);
    results[resultBuffer.index] = resultBuffer;
    taskDone++;
    if (taskSent < numOfSequences) {
      workerSendBuffer.index = taskSent;
      strcpy(workerSendBuffer.sequence, sequences[taskSent]);
      MPI_Send(
        &workerSendBuffer, 1, workerPayload, status.MPI_SOURCE, WORK, MPI_COMM_WORLD
      );
      taskSent++;
    }
  }

  // notify all workers to stop working
  for (int rank = 1; rank < numOfProc; rank++) {
    // dummy value
    workerSendBuffer.index = -1;
    strcpy(workerSendBuffer.sequence, "");
    MPI_Send(&workerSendBuffer, 0, workerPayload, rank, STOP, MPI_COMM_WORLD);
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