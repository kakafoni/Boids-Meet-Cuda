#ifndef boid_h
#define boid_h

#include "glm/glm.hpp"
#include <vector>

#define PREDATOR_FLAG 1
#define IS_ALIVE_FLAG 2

#if defined(__CUDACC__) // NVCC
   #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MY_ALIGN(n) alignas(n)
#elif defined(_MSC_VER) // MSVC
  #define MY_ALIGN(n) alignas(n)
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif


// TODO: Try different align values, can affect performance according to CUDA docs 9.2.1.2. A Sequential but Misaligned Access Pattern
struct MY_ALIGN(32) Boid {
	glm::vec3 velocity, position;
	char status;

};

#endif