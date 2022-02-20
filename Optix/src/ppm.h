#ifndef ppm_h
#define ppm_h

#include "stdint.h"

#include <cstdio>

typedef struct PPMFormat {
  int width, height, size;
  uint8_t maxColor;
  uint8_t depth;
  char *tupleType;
  uint8_t *pixmap;
} PPMFormat;

void ppm_ReadInP3(FILE *inFile, PPMFormat inData);
void ppm_ReadInP6(FILE *inFile, PPMFormat inData);
void ppm_ReadInP7(FILE *inFile, PPMFormat inData);
void ppm_WriteOutP3(PPMFormat inData, FILE *outFile);
void ppm_WriteOutP6(PPMFormat inData, FILE *outFile);
void ppm_WriteOutP7(PPMFormat inData, FILE *outFile);
void ppm_AtEndOfFile(FILE *inFile, char *errorMessage);
void ppm_ReadLine(FILE *fileStream, char *outputBuffer);
void ppm_SplitOnSpaces(char *inputBuffer, char *output1, char *output2);
void ppm_Fail(char *s);
uint8_t *ppm_RemoveAlphaChannel(PPMFormat inData);
float ppm_clamp(float value, float lower_bound, float upper_bound);

#endif
