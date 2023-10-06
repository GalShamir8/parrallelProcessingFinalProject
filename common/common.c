#include "common.h"

int** buildScoreTable(char* filePath) {
  int** scoreMat = (int**) calloc(SCORE_TABLE_COLS, sizeof(int*));

  for (int i = 0; i < SCORE_TABLE_ROWS; i++)
    scoreMat[i] = (int*) calloc(SCORE_TABLE_ROWS, sizeof(int));

  if (filePath == NULL) {
    for (int i = 0; i < SCORE_TABLE_ROWS; i++)
      scoreMat[i][i] = 1;
  } else {
    FILE *fptr;
    if ((fptr = fopen(filePath, "r")) == NULL) {
    printf("Error! opening file");
    exit(1);
    }

    for (int row = 0; row < SCORE_TABLE_ROWS; row++) {
      for (int col = 0; col < SCORE_TABLE_COLS; col++) {
        int fileInput;
        // check if there is a file content left to read
        if(fscanf(fptr, "%d", &fileInput) == 1)
          scoreMat[row][col] = fileInput;
      }
    }
    fclose(fptr);
  }
  return scoreMat;
}

void readLine(FILE* fileDescriptor, char* str, int maximumCharacters) {
  fgets(str, maximumCharacters, fileDescriptor);
  str[strlen(str) - 1] = '\0';
}

char** readSeqInput(char* filePath, int* numOfSequences) {
  FILE *fileDescriptor;
  if ((fileDescriptor = fopen(filePath, "r")) == NULL) {
  printf("Error! opening file");
  exit(1);
  }
  char seq1[MAX_CHARACTERS_SEQ1];
  readLine(fileDescriptor, seq1, MAX_CHARACTERS_SEQ1);

  if(fscanf(fileDescriptor, "%d", numOfSequences) != 1) {
    printf("Error! reading sequence number");
    exit(1);
  }
  fgetc(fileDescriptor); // cleaning new line from buffer

  char** sequences = (char**) malloc((*numOfSequences + 1) * sizeof(char*));
  if (sequences == NULL) {
    printf("Error! malloc");
    exit(1);
  }

  sequences[0] = (char*) malloc((strlen(seq1) + 1) * sizeof(char));
  if (sequences[0] == NULL) {
    printf("Error! malloc");
    exit(1);
  }
  strcpy(sequences[0], seq1);

  for (int i = 1; i <= *numOfSequences; i++) {
    char seq[MAX_CHARACTERS_SEQ];
    readLine(fileDescriptor, seq, MAX_CHARACTERS_SEQ);
    if (strlen(seq) >= strlen(seq1)) {
      printf("Usage: sequenece number %d longer than main sequenece", i);
      exit(1);
    }
    sequences[i] = (char*) malloc((strlen(seq) + 1) * sizeof(char));
    if (sequences[i] == NULL) {
      printf("Error! malloc");
      exit(1);
    }
    strcpy(sequences[i], seq);
  }

  fclose(fileDescriptor);

  return sequences;
}

void printMatrix(int** mat, int rows, int cols) {
  for (int row = 0; row < SCORE_TABLE_ROWS; row++) {
    for (int col = 0; col < SCORE_TABLE_COLS; col++) {
      printf("%5d ", mat[row][col]);
    }
    printf("\n");
  }
}

void printMatrix1D(int* mat, int rows, int cols) {
  for (int row = 0; row < SCORE_TABLE_ROWS; row++) {
    for (int col = 0; col < SCORE_TABLE_COLS; col++) {
      int index = (row * cols) + col;
      printf("%5d ", *(mat + index));
    }
    printf("\n");
  }
}

void printSequeneces(char** mat, int numOfSec) {
  for (int i = 0; i < (numOfSec + 1); i++) {
    printf("%s\n", &mat[i][0]);
  }
}