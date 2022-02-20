#ifndef parse_h
#define parse_h

#include "objects.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct json_data_t {
  float camera_width;
  float camera_height;
  shape_t *shapes_list;
  light_t *lights_list;
  int num_shapes;
  int num_lights;
} json_data_t;

void parse_json(FILE *json, json_data_t *json_data);
int read_field(FILE *json, char *output);
void split_on_colon(char *input, char *name, char *value);
void set_to_black(float *input);

#endif