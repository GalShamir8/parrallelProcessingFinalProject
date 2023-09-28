#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "../../common/common.h"

typedef struct {
  int score;
  int offset;
  int k;
} ScorePayload;

typedef struct {
  int maxScore;
  int offset;
  int k;
  int index;
  char sequence[MAX_CHARACTERS_SEQ];
} ResultPayload;

// Function declaration
void calculateMaxAlignmentScore(char* mainSequence, char* sequence, ResultPayload *sendBuffer, int** scoreMat);

__global__ void calculate(char* mainSequence, int mainSequenceLength, char* sequence, int sequenceLength,
                          int* scoreMat, int scoreMatLength, ScorePayload* scorePayload
) {
  __shared__ int results[MAX_CHARACTERS_SEQ1];
  int offset = blockIdx.x;
  int k = threadIdx.y;

  int rowIndex = mainSequence[threadIdx.x + offset] - 'A';
  int colIndex;

  if (blockDim.x >= k) {
    colIndex = (sequence[threadIdx.x] + 1) - 'A';
  } else {
    colIndex = sequence[threadIdx.x] - 'A';
  }

  if (colIndex >= SCORE_TABLE_COLS) {
    colIndex -= SCORE_TABLE_COLS;
  }

  int score = scoreMat[(rowIndex * SCORE_TABLE_COLS) + colIndex];
  results[threadIdx.x] = score;

  __syncthreads();
  
  int sum = 0;
  for (int i = 0; i < mainSequenceLength; i++) {
    sum += results[i];
  }

  int scoreIndex = k * blockDim.y + blockIdx.x;
  scorePayload[scoreIndex].score = sum;
  scorePayload[scoreIndex].offset = offset;
  scorePayload[scoreIndex].k = k;
}

void calculateMaxAlignmentScore(char* mainSequence, char* sequence, ResultPayload *sendBuffer, int** scoreMat) {
  int* deviceScoreMat;
  char *deviceMainSequence, *deviceSequence;
  int mainSequenceLength = strlen(mainSequence) + 1,
      sequenceLength= strlen(sequence) + 1,
      scoreMatLength = SCORE_TABLE_ROWS * SCORE_TABLE_COLS;
  int numOfBlocks = (mainSequenceLength - sequenceLength), 
      offset = (mainSequenceLength - sequenceLength),
      k = sequenceLength - 1;
  dim3 threadsPerBlock(sequenceLength, k);
  int numOfScores = offset * k;
  ScorePayload *scorePayload, *deviceScorePayload;

  scorePayload = (ScorePayload*) malloc(numOfScores * sizeof(ScorePayload));
  if (scorePayload == NULL){
    printf("Failed to allocate score payload");
    exit(1);
  } 

  // allocate memory for cuda arguments (pointers mostly)
  cudaMalloc((void**)&deviceScorePayload, numOfScores * sizeof(ScorePayload));
  cudaMalloc((void**)&deviceMainSequence, mainSequenceLength);
  cudaMalloc((void**)&deviceSequence, sequenceLength);
  cudaMalloc((void**)&deviceScoreMat, scoreMatLength * sizeof(int));

  // assign CPU variables into allocated GPU variables
  cudaMemcpy(deviceMainSequence, mainSequence, mainSequenceLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceSequence, sequence, sequenceLength, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceScoreMat, *scoreMat, scoreMatLength, cudaMemcpyHostToDevice);

  calculate<<<numOfBlocks, threadsPerBlock>>>(
    deviceMainSequence, mainSequenceLength, 
    deviceSequence, sequenceLength,
    deviceScoreMat, scoreMatLength,
    deviceScorePayload
  );

  // copy result from GPU to CPU variable
  cudaMemcpy(scorePayload, deviceScorePayload, numOfScores * sizeof(ScorePayload), cudaMemcpyDeviceToHost);

  #pragma omp declare reduction (max_alignment_score: struct ScorePayload :\
            omp_out = (omp_out.score > omp_in.scre ? omp_out : omp_in)\
            initializer(omp_priv = omp_orig)

  ScorePayload maxScore;
  maxScore.score = scorePayload[0].score;

  #pragma omp parallel for reduction (max_alignment_score: vmax)
  for (int i = 0; i < numOfScores; i++)
  {
    maxScore.score = scorePayload[i].score;
    maxScore.offset = scorePayload[i].offset;
    maxScore.k = scorePayload[i].k;
  }

  // enrich sndBuffer with CPU results
  sendBuffer->offset = maxScore.offset;
  sendBuffer->maxScore = maxScore.score;
  sendBuffer->k = maxScore.k;

  // free memory on the GPU side
  cudaFree(deviceMainSequence);
  cudaFree(deviceSequence);
  cudaFree(deviceScoreMat);
}