#ifndef objects_h
#define objects_h

typedef enum {
  SPHERE,
  PLANE,
  QUADRIC
} shape_type_t;

typedef enum {
  POINT,
  SPOTLIGHT
} light_type_t;

typedef struct shape_t {
  float diffuse_color[3];
  float specular_color[3];
  float position[3];
  float reflectivity;
  float refractivity;
  float ior;
  union {
    // Plane 
    struct {
      float normal[3];
    };

    // Sphere
    struct {
      float radius;
    };

    // Quadric
    struct {
      float a;
      float b;
      float c;
      float d;
      float e;
      float f;
      float g;
      float h;
      float i;
      float j;
    };
  };
  shape_type_t type;
  struct shape_t *next;
} shape_t;

typedef struct light_t {
  float position[3];
  float color[3];
  float radial_coef[3];
  float theta;
  float cos_theta;
  float a0;
  float direction[3];
  light_type_t type;
  struct light_t *next;
} light_t;

shape_t *add_new_sphere(shape_t *head, float *diffuse, float *specular, float *position, 
                       float radius, float reflectivity, float refractivity, float ior);
shape_t *add_new_plane(shape_t *head, float *diffuse, float *specular, float *position, 
                       float *normal, float reflectivity);
shape_t *add_new_quadric(shape_t *head, float *diffuse, float *specular, float a, float b, float c, 
                        float d, float e, float f, float g, float h, float i, float j, float reflectivity);
shape_t *free_shape_list(shape_t *head);
light_t *free_light_list(light_t *head);
light_t *add_new_spot_light(light_t *head, float *color, float *position, 
                             float theta, float a0, float *direction, float *radial_coef);
light_t *add_new_point_light(light_t *head, float *color, float *position, float *radial_coef);

#endif
