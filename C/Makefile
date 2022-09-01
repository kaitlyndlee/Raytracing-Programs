all: raytrace

raytrace: raycast.c v3math.c ppm.c parse.c objects.c
	gcc -O3 raycast.c v3math.c ppm.c parse.c objects.c -o raytrace -lm

clean:
	rm -f raytrace
