# Boids-Meet-CUDA
Flocking simulation on the GPU using CUDA, written in C++. Code partially derived from my bachelor's thesis in computer engineering at Chalmers University of Technology. 

Thanks to Oskar Lyrstrand ([oskery](https://github.com/oskery)) for most of the OpenGL related code. 
## Dependencies

- [GLFW](https://www.glfw.org/): for creating windows and retrieving input.
- [GLAD](https://glad.dav1d.de/): for cross-platform support.
- [GLM](https://glm.g-truc.net/): for vector and matrix transformations etc.
- [CUB](https://nvlabs.github.io/cub/): for parallel sorting algorithms. 

## Demo
Watch a simulation with 500 000 boids on youtube:

[![Flocking simulation: 500 000 boids using C++/CUDA](https://img.youtube.com/vi/cKETvurFrbQ/0.jpg)](https://www.youtube.com/watch?v=cKETvurFrbQ)
