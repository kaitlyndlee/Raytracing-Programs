#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "ppm.h"

const int EOL = 10;


/**
 * Checks to see if the file is at its end. If it is at the end the function, will call ppm_Fail with
 * the given error message.
 * 
 * @param inFile The input file
 * @param errorMessage The error message to send to the screen if the file is at the end of the file
 */
void ppm_AtEndOfFile(FILE *inFile, char *errorMessage){
  int checkFile;
  checkFile = getc(inFile);
  if (checkFile == EOF) {
    ppm_Fail(errorMessage); 
  }
  else {
    ungetc(checkFile, inFile);
  }
}

/**
 * Reads in a P3 PPM file. Takes all the image data and stores it in the PPMFormat data structure's
 * pixmap. This fucntion checks to see if the image is the correct width and height as stated in the
 * PPM file header.
 * 
 * @param inFile The input file
 * @param inData The data structure holding the pixmap and PPM values
 */
void ppm_ReadInP3 (FILE *inFile, PPMFormat inData) {
  int buffer;
  uint8_t num;
  uint8_t *pixel = inData.pixmap;
  
  for (int y = 0; y < inData.height; y++) {
    for (int x = 0; x < inData.width; x++) {
      for (int p = 0; p < 3; p++){
        ppm_AtEndOfFile(inFile, "The input file is too short based on the given height and width.");

        fscanf(inFile, "%d", &buffer);
        num = buffer;

        if (buffer > inData.maxColor){
          ppm_Fail("Pixel exceeds max color.");
        }
        pixel[p] = num;
      }
      pixel[3] = 255;
      pixel += 4;
    }
  }

  while (isspace(getc(inFile)));

  if (getc(inFile) != EOF) {
    ppm_Fail("The input file is too long for the given width and height.");
  }
}

/**
 * Reads in a P6 PPM file. Takes all the image data and stores it in the PPMFormat data structure's
 * pixmap. This fucntion checks to see if the image is the correct width and height as stated in the
 * PPM file header.
 * 
 * @param inFile The input file
 * @param inData The data structure holding the pixmap and PPM values
 */
void ppm_ReadInP6 (FILE *inFile, PPMFormat inData) {
  uint8_t buffer[3];
  uint8_t *pixel = inData.pixmap;
  
  for (int y = 0; y < inData.height; y++) {
    for (int x = 0; x < inData.width; x++) {
      int entriesRead = fread(buffer, 3, 1, inFile);
      if (entriesRead < 1) {
        ppm_Fail("The input file is too short based on the given height and width.");
      }

      if (buffer[0] > inData.maxColor || buffer[1] > inData.maxColor || buffer[2] > inData.maxColor){
        ppm_Fail("Pixel exceeds max color.");
      }

      pixel[0] = buffer[0];
      pixel[1] = buffer[1];
      pixel[2] = buffer[2];
      pixel[3] = 255;
      pixel += 4;
    }
  }

  while (isspace(getc(inFile)));

  if (getc(inFile) != EOF){
    ppm_Fail("The input file is too long for the given width and height.");
  }
}

/**
 * Reads in a P7 PPM file. Takes all the image data and stores it in the PPMFormat data structure's
 * pixmap. This fucntion checks to see if the image is the correct width and height as stated in the
 * PPM file header.
 * 
 * @param inFile The input file
 * @param inData The data structure holding the pixmap and PPM values
 */
void ppm_ReadInP7 (FILE *inFile, PPMFormat inData) {
  uint8_t buffer[4];
  uint8_t *pixel = inData.pixmap;
  
  for (int y = 0; y < inData.height; y++) {
    for (int x = 0; x < inData.width; x++) {
      if (inData.depth == 3) {
        int entriesRead = fread(buffer, 3, 1, inFile);
        if (entriesRead < 1) {
          ppm_Fail("The input file is too short based on the given height and width.");
        }

        if (buffer[0] > inData.maxColor || buffer[1] > inData.maxColor || buffer[2] > inData.maxColor){
        ppm_Fail("Pixel exceeds max color.");
      }
        pixel[0] = buffer[0];
        pixel[1] = buffer[1];
        pixel[2] = buffer[2];
        pixel[3] = 255;
        pixel += 4;
      }
      else {
        int entriesRead = fread(buffer, 4, 1, inFile);
        if (entriesRead < 1) {
          ppm_Fail("The input file is too short based on the given height and width.");
        }

        if (buffer[0] > inData.maxColor || buffer[1] > inData.maxColor 
            || buffer[2] > inData.maxColor || buffer[3] > inData.maxColor){
        ppm_Fail("Pixel exceeds max color.");
      }

        pixel[0] = buffer[0];
        pixel[1] = buffer[1];
        pixel[2] = buffer[2];
        pixel[3] = buffer[3];
        pixel += 4;
      }
    }
  }

  while (isspace(getc(inFile)));

  if (getc(inFile) != EOF){
    ppm_Fail("The input file is too long for the given width and height.");
  }
}

/**
 * Prints the pixmap from the PPMFormat data structure into the output file. This fucntion will print
 * in P3 PPM format, so it will print out in ASCII.
 * 
 * @param inData The data structure holding the pixmap and PPM values
 * @param outFile The output file
 */
void ppm_WriteOutP3 (PPMFormat inData, FILE *outFile) {
  // inData.pixmap = ppm_RemoveAlphaChannel(inData);
  uint8_t *pixel = inData.pixmap;
  fprintf(outFile, "P3\n");
  fprintf(outFile, "%d %d \n%u\n", inData.width, inData.height, inData.maxColor);

  for (int y = 0; y < inData.height; y++){
    for (int x = 0; x < inData.width; x++){
      uint8_t *pixel = &inData.pixmap[3*((y * inData.width) + x)];
      fprintf(outFile,"%d\n%d\n%d\n",
           pixel[0],
           pixel[1],
           pixel[2]);
      pixel += 3;
    }
  }
}

/**
 * Prints the pixmap from the PPMFormat data structure into the output file. This fucntion will print
 * in P6 PPM format, so it will print out in raw binary.
 * 
 * @param inData The data structure holding the pixmap and PPM values
 * @param outFile The output file
 */
void ppm_WriteOutP6 (PPMFormat inData, FILE *outFile) {
  fprintf(outFile, "P6");
  fputc(EOL, outFile);
  fprintf(outFile, "%d %d", inData.width, inData.height);
  fputc(EOL, outFile);
  fprintf(outFile, "%d", inData.maxColor);
  fputc(EOL, outFile);
  inData.pixmap = ppm_RemoveAlphaChannel(inData);
  fwrite(inData.pixmap, 1, 3 * inData.width * inData.height, outFile);
  fputc(EOL, outFile);
}

/**
 * Prints the pixmap from the PPMFormat data structure into the output file. This fucntion will print
 * in P7 PPM format, so it will print out in raw binary.
 * 
 * @param inData The data structure holding the pixmap and PPM values
 * @param outFile The output file
 */
void ppm_WriteOutP7 (PPMFormat inData, FILE *outFile) {
  fprintf(outFile, "P7");
  fputc(EOL, outFile);
  fprintf(outFile, "WIDTH %d", inData.width);
  fputc(EOL, outFile);
  fprintf(outFile, "HEIGHT %d", inData.height);
  fputc(EOL, outFile);
  fprintf(outFile, "DEPTH %d", 4);
  fputc(EOL, outFile);
  fprintf(outFile, "MAXVAL %d", inData.maxColor);
  fputc(EOL, outFile);
  fprintf(outFile, "TUPLTYPE RGB_ALPHA");
  fputc(EOL, outFile);
  fwrite(inData.pixmap, 1, 4 * inData.width * inData.height, outFile);
  fputc(EOL, outFile);
}

/**
 * Prints an error message to the screen and exits the program gracefully.
 * 
 * @param s String containing error message
 */
void ppm_Fail(char *s) {
  fprintf(stderr, "Error: %s\n\n", s);
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "test_pattern CONVERT_TO INPUT_FILE OUTPUT_FILE\n\n");
  exit(1);
}

/**
 * Reads a line from the header, skipping comments that come before and after any tokens.
 * 
 * @param fileStream   Input file stream
 * @param outputBuffer String to store ouptut with token.
 */
void ppm_ReadLine(FILE *fileStream, char *outputBuffer) {
  char currentChar = fgetc(fileStream);

  // skip comments before header line
  if (currentChar == '#') {
    while (1) {
      currentChar = fgetc(fileStream);

      if (currentChar == EOL) {
        currentChar = fgetc(fileStream);
        if (currentChar == '#') {
          continue;
        }
        else {
          break;
        }
      }
    }
  }

  // Grab header tokens
  int count = 0;
  while (currentChar != EOL && currentChar != '#') {
    outputBuffer[count] = currentChar;
    count++;
    outputBuffer[count] = '\0';
    currentChar = fgetc(fileStream);
  }

  // Skip comments after tokens
  if (currentChar == '#') {
    while (currentChar != EOL) {
      currentChar = fgetc(fileStream);
    }
  }
}


/**
 * Takes in a buffer string and stores the tokens in two output strings, skipping whitespace.
 * Used to grab the header name and value, e.g., 'HEADER' and '350'.
 * 
 * @param inputBuffer String containing input to split on
 * @param output1     Header name
 * @param output2     Header value
 */
void ppm_SplitOnSpaces(char *inputBuffer, char *output1, char *output2) {
  int index = 0;
  while (!isspace(inputBuffer[index])) {
    output1[index] = inputBuffer[index];
    index++;
    output1[index] = '\0';
  }

  while (isspace(inputBuffer[index])) {
    index++;
  } 

  // Clear what was already stored in output2
  output2[0] = '\0';

  int outputIndex = 0;
  while (inputBuffer[index] != '\0') {
    output2[outputIndex] = inputBuffer[index];
    index++;
    outputIndex++;
    output2[outputIndex] = '\0';
  }
}

/**
 * Removes the alpha channel from a pixmap array to be exported to P3 or P6.
 * 
 * @param  inData PPMFormat struct containing pixmap that has an alpha channel.
 * @return        Array without the alpha channel.
 */
uint8_t *ppm_RemoveAlphaChannel(PPMFormat inData) {
  uint8_t *tempArray = malloc(inData.width * inData.height * 3);
  uint8_t *pixelWithAlpha = inData.pixmap;
  uint8_t *pixelNoAlpha = tempArray;

  for (int y = 0; y < inData.height; y++) {
    for (int x = 0; x < inData.width; x++) {
      uint8_t *pixel = &inData.pixmap[3*((y * inData.width) + x)];
      pixelNoAlpha[0] = pixelWithAlpha[0];
      pixelNoAlpha[1] = pixelWithAlpha[1];
      pixelNoAlpha[2] = pixelWithAlpha[2];
      pixelWithAlpha += 4;
      pixelNoAlpha += 3;
    }
  }
  return tempArray;
}

/**
 * Takes in a value and clamps the value into a range
 * using a lower and upper bound.
 *
 * @param value - The value to be clamped
 * @param lower_bound - The lower bound
 * @param upper_bound - The upper bound
 * @return - returns the clamped value
 */
float ppm_clamp(float value, float lower_bound, float upper_bound) {
  if (value > upper_bound) {
    value = upper_bound;
  }
  if (value < lower_bound) {
    value = lower_bound;
  }

  return value;
}
