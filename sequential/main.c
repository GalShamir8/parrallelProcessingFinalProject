#include "main.h"

int main(int argc, char** argv) {
  int** scoreMat;
  char** sequences;
  int numOfSequences;

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
  process(scoreMat, sequences, numOfSequences);
  free(scoreMat);
  free(sequences);
}

void process(int** scoreMat, char** sequences, int numOfSequences) {
  char* mainSequence = sequences[0];
  int mainsequenceLength = (strlen(mainSequence));

  for (int seq = 1; seq <= numOfSequences; seq++) {
    int currentSequenceLength = strlen(sequences[seq]);
    int offset = (mainsequenceLength - currentSequenceLength);

    int maxScore = compare(scoreMat, mainSequence, sequences[seq], currentSequenceLength, currentSequenceLength, 0);
    int maxScoreOffset = 0, maxScoreK = currentSequenceLength;
    for (int currentOffset = 0; currentOffset <= offset; currentOffset++) {
      for (int k = (currentSequenceLength - 1); k >= 0; k--) {
        int score = compare(scoreMat, mainSequence, sequences[seq], currentSequenceLength, k, currentOffset);
        if (score > maxScore) {
          maxScore = score;
          maxScoreOffset = currentOffset;
          maxScoreK = k;
        }
      }
    }
  printf("%s highest alignment score = %d, offset = %d, k = %d\n", sequences[seq], maxScore, maxScoreOffset, maxScoreK);
  }
}

int compare(int** scoreMat, char* str, char* other, int numberOfIterations, int shift, int offset) {
  int scoreSum = 0;
  for (int i = 0; i < numberOfIterations; i++) {
    int rowIndex = str[i + offset] - 'A';
    int colIndex;

    if (i >= shift) {
      colIndex = (other[i] + 1) - 'A';
    } else {
      colIndex = other[i] - 'A';
    }
    if (colIndex >= SCORE_TABLE_COLS) {
      colIndex -= SCORE_TABLE_COLS;
    }

    scoreSum += scoreMat[rowIndex][colIndex];
  }

  return scoreSum;
}