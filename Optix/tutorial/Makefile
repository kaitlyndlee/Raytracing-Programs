CC = nvcc
GENCODE = -gencode=arch=compute_75,code=sm_75

all: raytrace

raytrace: main.cpp v3math.cu ppm.cu SampleRenderer.cpp devicePrograms.cu
	$(CC) -rdc=true $(GENCODE) main.cpp v3math.cu ppm.cu SampleRenderer.cpp devicePrograms.cu -o raytrace -I/home/klee/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64/include

clean:
	rm -f raytrace
