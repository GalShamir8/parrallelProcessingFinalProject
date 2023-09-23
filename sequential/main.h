#include "../common/common.h"
// #define DEBUG

int main(int argc, char **argv);

void process(int** scoreMat, char** sequences, int numOfSequences);

int compare(int** scoreMat, char* str, char* other, int numberOfIterations, int shift, int offset);