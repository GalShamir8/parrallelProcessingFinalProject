#pragma once
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#define SCORE_TABLE_ROWS 26
#define SCORE_TABLE_COLS 26
#define MAX_CHARACTERS_SEQ1 3000
#define MAX_CHARACTERS_SEQ 2000
#define ROOT_PROCESS_RANK 0

typedef struct {
  int scoreMat[SCORE_TABLE_ROWS * SCORE_TABLE_COLS];
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

/*
  @param filePath optional path to an input score configuration file or NULL for default
  @return matrix with scores for all the alphabetic characters combinations (incasesense 26 x 26)
  @throws IOException
*/
int** buildScoreTable(char* filePath);
/*
  @param filePath path to an inut file defining input sequences
  @param numOfSequences pointer to number of sequences (not included the main sequence)
  @return matrix with seq 1 in the first row, seq 2 in the second row ... (length can be diffrent from one row to another)
  @throws IOException
*/
char** readSeqInput(char* filePath, int* numOfSequences);
/*
  @param fileDescriptor pointer to file descriptor
  @param str the str for the read sequence assignment
  @maximumCharacters limit str capacity for characters
*/
void readLine(FILE* fileDescriptor, char* str, int maximumCharacters);

void printMatrix(int** mat, int rows, int cols);
void printMatrix1D(int* mat, int rows, int cols);
void printSequeneces(char** mat, int numOfSec);

