#include "parse.h"

#include <stdbool.h>
#include <string.h>

/**
 * Parses the input JSON, checks that it has a camera entry, and creates 
 * the shapes linked list to be used in the raycast method.
 *
 * @param json - the input file pointer
 * @param json_data - struct to store the camera width/height and shapes linked list
 */
void parse_json(FILE *json, json_data_t *json_data) {
  // Temp variables
  char buffer[100];
  char name[100];
  char value [100];
  float color[3];
  float diffuse[3];
  float specular[3];
  float reflectivity;
  float refractivity;
  float ior;
  bool diffuse_found;
  bool specular_found;
  float pos[3];
  float norm[3];
  float radius;
  float quadric_params[10];
  float theta;
  bool isSpotlight;
  float radial_coef[3];
  float a0;
  float direction[3];
  int cameras = 0;
  int num_fields;
  int eol_reached;

  while (true) {
    eol_reached = 0;
    num_fields = 0;
    specular_found = false;
    diffuse_found = false;
    read_field(json, buffer);
    if (strcmp(buffer, "camera") == 0) {
      cameras++;
      while (eol_reached == 0) {
        eol_reached = read_field(json, buffer);
        split_on_colon(buffer, name, value);
        if (strcmp(name, "width") == 0) {
          sscanf(value, "%f", &json_data->camera_width);
          num_fields++;
        }
        else if (strcmp(name, "height") == 0) {
          sscanf(value, "%f", &json_data->camera_height);
          num_fields++;
        }
      } 
      if (num_fields != 2) {
        fprintf(stderr, "Error: A camera width or height was not given.\n");
        exit(1);
      }
    }
    else if (strcmp(buffer, "sphere") == 0) {
      while (eol_reached == 0) {
        eol_reached = read_field(json, buffer);
        split_on_colon(buffer, name, value);
        if (strcmp(name, "diffuse_color") == 0) {
          sscanf(value, "[%f, %f, %f]", &diffuse[0], &diffuse[1], &diffuse[2]);
          num_fields++;
          diffuse_found = true;
        }
        else if (strcmp(name, "specular_color") == 0) {
          sscanf(value, "[%f, %f, %f]", &specular[0], &specular[1], &specular[2]);
          num_fields++;
          specular_found = true;
        }
        else if (strcmp(name, "position") == 0) {
          sscanf(value, "[%f, %f, %f]", &pos[0], &pos[1], &pos[2]);
          num_fields++;
        }
        else if (strcmp(name, "radius") == 0) {
          sscanf(value, "%f", &radius);
          num_fields++;
        }
        else if (strcmp(name, "reflectivity") == 0) {
          sscanf(value, "%f", &reflectivity);
          num_fields++;
        }
        else if (strcmp(name, "refractivity") == 0) {
          sscanf(value, "%f", &refractivity);
          num_fields++;
        }
        else if (strcmp(name, "ior") == 0) {
          sscanf(value, "%f", &ior);
          num_fields++;
        }
      }
      if (!specular_found) {
        set_to_black(specular);
        num_fields++;
      }
      if (!diffuse_found) {
        set_to_black(diffuse);
        num_fields++;
      }

      if (num_fields != 7) {
        fprintf(stderr, "Error: A field for a sphere was missing.\n");
        exit(1);
      }

      if (reflectivity < 0 || reflectivity > 1) {
        fprintf(stderr, "Error: invalid reflectivity for a sphere. Must be between 0 and 1.\n");
        exit(1);
      }
      if (refractivity < 0 || refractivity > 1) {
        fprintf(stderr, "Error: invalid refractivity for a sphere. Must be between 0 and 1.\n");
        exit(1);
      }

      json_data->shapes_list = add_new_sphere(json_data->shapes_list, diffuse, specular, pos, radius, 
                                              reflectivity, refractivity, ior);
      json_data->num_shapes += 1;
    }
    else if (strcmp(buffer, "plane") == 0) {
      while(eol_reached == 0) {
        eol_reached = read_field(json, buffer);
        split_on_colon(buffer, name, value);
        if (strcmp(name, "diffuse_color") == 0) {
          sscanf(value, "[%f, %f, %f]", &diffuse[0], &diffuse[1], &diffuse[2]);
          num_fields++;
          diffuse_found = true;
        }
        else if (strcmp(name, "specular_color") == 0) {
          sscanf(value, "[%f, %f, %f]", &specular[0], &specular[1], &specular[2]);
          num_fields++;
          specular_found = true;
        }
        else if (strcmp(name, "position") == 0) {
          sscanf(value, "[%f, %f, %f]", &pos[0], &pos[1], &pos[2]);
          num_fields++;
        }
        else if (strcmp(name, "normal") == 0) {
          sscanf(value, "[%f, %f, %f]", &norm[0], &norm[1], &norm[2]);
          num_fields++;
        }
        else if (strcmp(name, "reflectivity") == 0) {
          sscanf(value, "%f", &reflectivity);
          num_fields++;
        }
      }
      if (!specular_found) {
        set_to_black(specular);
        num_fields++;
      }
      if (!diffuse_found) {
        set_to_black(diffuse);
        num_fields++;
      }

      if (num_fields != 5) {
        fprintf(stderr, "Error: A field for a plane was missing.\n");
        exit(1);
      }

      if (reflectivity < 0 || reflectivity > 1) {
        fprintf(stderr, "Error: invalid reflectivity for a sphere. Must be between 0 and 1.\n");
        exit(1);
      }

      json_data->shapes_list = add_new_plane(json_data->shapes_list, diffuse, specular, pos, norm, reflectivity);
      json_data->num_shapes += 1;
    }
    else if (strcmp(buffer, "quadric") == 0) {
      while (eol_reached == 0) {
        eol_reached = read_field(json, buffer);
        split_on_colon(buffer, name, value);
        if (strcmp(name, "diffuse_color") == 0) {
          sscanf(value, "[%f, %f, %f]", &diffuse[0], &diffuse[1], &diffuse[2]);
          num_fields++;
          diffuse_found = true;
        }
        else if (strcmp(name, "specular_color") == 0) {
          sscanf(value, "[%f, %f, %f]", &specular[0], &specular[1], &specular[2]);
          num_fields++;
          specular_found = true;
        }
        else if (strcmp(name, "reflectivity") == 0) {

          sscanf(value, "%f", &reflectivity);
          num_fields++;
        }
        else if (strcmp(name, "a") == 0) {
          sscanf(value, "%f", &quadric_params[0]);
          num_fields++;
        }
        else if (strcmp(name, "b") == 0) {
          sscanf(value, "%f", &quadric_params[1]);
          num_fields++;
        }
        else if (strcmp(name, "c") == 0) {
          sscanf(value, "%f", &quadric_params[2]);
          num_fields++;
        }
        else if (strcmp(name, "d") == 0) {
          sscanf(value, "%f", &quadric_params[3]);
          num_fields++;
        }
        else if (strcmp(name, "e") == 0) {
          sscanf(value, "%f", &quadric_params[4]);
          num_fields++;
        }
        else if (strcmp(name, "f") == 0) {
          sscanf(value, "%f", &quadric_params[5]);
          num_fields++;
        }
        else if (strcmp(name, "g") == 0) {
          sscanf(value, "%f", &quadric_params[6]);
          num_fields++;
        }
        else if (strcmp(name, "h") == 0) {
          sscanf(value, "%f", &quadric_params[7]);
          num_fields++;
        }
        else if (strcmp(name, "i") == 0) {
          sscanf(value, "%f", &quadric_params[8]);
          num_fields++;
        }
        else if (strcmp(name, "j") == 0) {
          sscanf(value, "%f", &quadric_params[9]);
          num_fields++;
        }
      }
      if (!specular_found) {
        set_to_black(specular);
        num_fields++;
      }
      if (!diffuse_found) {
        set_to_black(diffuse);
        num_fields++;
      }

      if (num_fields != 13) {
        fprintf(stderr, "Error: A field for a quadric was missing.\n");
        exit(1);
      }

      if (reflectivity < 0 || reflectivity > 1) {
        fprintf(stderr, "Error: invalid reflectivity for a sphere. Must be between 0 and 1.\n");
        exit(1);
      }

      json_data->shapes_list = add_new_quadric(json_data->shapes_list, diffuse, specular, 
                                               quadric_params[0], quadric_params[1], 
                                               quadric_params[2], quadric_params[3], 
                                               quadric_params[4], quadric_params[5], 
                                               quadric_params[6], quadric_params[7], 
                                               quadric_params[8], quadric_params[9], reflectivity);
      json_data->num_shapes += 1;
    }
    else if (strcmp(buffer, "light") == 0) {
      isSpotlight = false;
      while (eol_reached == 0) {
        eol_reached = read_field(json, buffer);
        split_on_colon(buffer, name, value);
        if (strcmp(name, "color") == 0) {
          sscanf(value, "[%f, %f, %f]", &color[0], &color[1], &color[2]);
          num_fields++;
        }
        else if (strcmp(name, "position") == 0) {
          sscanf(value, "[%f, %f, %f]", &pos[0], &pos[1], &pos[2]);
          num_fields++;
        }
        else if (strcmp(name, "theta") == 0) {
          sscanf(value, "%f", &theta);
          if (theta == 0.0) { // is a point light
            continue;
          }

          // convert to radians
          theta = theta * (180.0 / PI);
          num_fields++;
          isSpotlight = true;
        }
        else if (strcmp(name, "radial-a0") == 0) {
          sscanf(value, "%f", &radial_coef[0]);
          num_fields++;
        }
        else if (strcmp(name, "radial-a1") == 0) {
          sscanf(value, "%f", &radial_coef[1]);
          num_fields++;
        }
        else if (strcmp(name, "radial-a2") == 0) {
          sscanf(value, "%f", &radial_coef[2]);
          num_fields++;
        }
        else if (strcmp(name, "angular-a0") == 0) {
          sscanf(value, "%f", &a0);
          num_fields++;
        }
        else if (strcmp(name, "direction") == 0) {
          sscanf(value, "[%f, %f, %f]", &direction[0], &direction[1], &direction[2]);
          num_fields++;
        }
      }
      if ( (!isSpotlight && num_fields != 5) || (isSpotlight && num_fields != 8 ) ) {
        fprintf(stderr, "Error: A field for a light was missing.\n");
        exit(1);
      }
      if (isSpotlight) {
        json_data->lights_list = add_new_spot_light(json_data->lights_list, color, pos, theta, a0, 
                                                    direction, radial_coef);
      }
      else {
        json_data->lights_list = add_new_point_light(json_data->lights_list, color, pos, radial_coef);
      }

      json_data->num_lights += 1; 
    }

    char tempChar = fgetc(json);
    if(tempChar == EOF) {
      break;
    }
    else {
      ungetc(tempChar, json);
    }
  }

  if (cameras != 1) {
    fprintf(stderr, "Error: The scene must have one and only one camera.\n");
    exit(1);
  }

  /*shape_t object_array[json_data->num_shapes];
  shape_t *current_object = json_data->shapes_list;
  light_t light_array[json_data->num_lights];
  light_t *current_light = json_data->lights_list;

  for(int index = 0;index<json_data->num_shapes; index++)
  {
    object_array[index] = *current_object;
    current_object = current_object->next;
  }

  for(int index = 0;index<json_data->num_lights; index++)
  {
    light_array[index] = *current_light;
    current_light = current_light->next;
  }

  //printf("getting here");
  json_data->lights_list = free_light_list(json_data->lights_list);
  json_data->shapes_list = free_shape_list(json_data->shapes_list);
  json_data->shapes_list = object_array;
  json_data->lights_list = light_array;*/
}

/**
 * Reads the input file until a comma is reached. If a tuple is part of the field, it will
 * read the tuple with the commas and stop when it reaches the comma after the tuple ends.
 * For example: position: [0, 0, 0] could be returned.
 *
 * @param json - the input file pointer
 * @param output - output buffer to store the field
 */
int read_field(FILE *json, char *output) {
  char currentChar = fgetc(json);
  int count = 0;
  output[count] = '\0';
  while (currentChar != ',' && currentChar != '\n' && currentChar != EOF) {
    if (currentChar == '[') {
      while (currentChar != ']') {
        output[count] = currentChar;
        count++;
        output[count] = '\0';
        currentChar = fgetc(json);
      }
    }

    output[count] = currentChar;
    count++;
    output[count] = '\0';
    currentChar = fgetc(json);
  }

  if (currentChar == '\n' || currentChar == EOF) {
    return 1;
  }
  return 0;
}


/**
 * Splits a string on a colon and stores the two strings in name and value.
 *
 * @param input - the string to split
 * @param name - output buffer for the left string
 * @param value - output buffer for the right string
 */
void split_on_colon(char *input, char *name, char *value) {
  int index = 0;
  int name_index = 0;
  while (input[index] != ':') {
    while(input[index] == ' ') {
      index++;
    }
    name[name_index] = input[index];
    index++;
    name_index++;
    name[name_index] = '\0';
  }

  // skip colon
  index++;

  // skip extra spaces
  while(input[index] == ' ') {
    index++;
  }

  int val_index = 0;
  while (input[index] != '\0') {
    value[val_index] = input[index];
    index++;
    val_index++;
    value[val_index] = '\0';
  }
}

void set_to_black(float *input) {
  input[0] = 0;
  input[1] = 0;
  input[2] = 0;
}