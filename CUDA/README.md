# Authors
Kaitlyn Lee kdl222@nau.edu
Jacob Kaufman jmk478@nau.edu

# Usage

This progrgam, raytrace, runs on the GPU with CUDA. In order to run the program,
you will need an NVIDIA GPU.
raytrace takes in a scene file and outputs an image in ppm format.
It currently supports spheres, planes, and quadrics. When using quadrics,
make sure the position of the quadric is calculated into the parameters.
You can find example scenes in the `examples` directory.
To run the program:
`raytrace WIDTH HEIGHT INPUT_SCENE OUTPUT_IMAGE`

# Known Issues
No known issues