#include "mpi/helper.h"
#include "cuda/scoreCalculation.cu"
// #define DEBUG

enum tags { WORK, STOP, FINISH };

int main(int argc, char **argv);

void masterProcess(int numOfProc, int argc, char** argv);

void workerProcess(int rank);

void printResults(ResultPayload results[], int size);