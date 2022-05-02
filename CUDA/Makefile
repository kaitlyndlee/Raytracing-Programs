CC = nvcc
GENCODE = -gencode=arch=compute_75,code=sm_75

all: raytrace

raytrace: raycast.cu v3math.cu ppm.cu parse.cu objects.cu
	$(CC) -O3 -rdc=true $(GENCODE) raycast.cu v3math.cu ppm.cu parse.cu objects.cu -o raytrace

clean:
	rm -f raytrace
