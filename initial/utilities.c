#include <stdlib.h>

float **allocFloatMatrix(int columns, int rows) {
    int i; 
    float **matrix;
    
    matrix = (float**) malloc(columns*sizeof(float *));
    
    float *elements = (float *)calloc(rows*columns, sizeof(float));
    
    for(i=0; i<columns; i++) {
        matrix[i] = &elements[rows*i];
    }
    
    return matrix;
}

char **allocCharMatrix(int columns, int rows) {
    int i; 
    char **matrix;
    
    matrix = (char**) malloc(columns*sizeof(char *));
    
    char *elements = (char *)calloc(rows*columns, sizeof(char));
    
    for(i=0; i<columns; i++) {
        matrix[i] = &elements[rows*i];
    }
    
    return matrix;
}

void freeMatrix(void *matrix) {
    void **elements = (void **) matrix;
    free(elements[0]);
    free(matrix);
}