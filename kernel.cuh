#ifndef kernel_h
#define kernel_h

#include "boid.h"

#define NR_BOIDS 500000

// MAX_COORD 700.f
#define MAX_COORD 1300.f // The same as the width/height/depth of the cube the boids live in
#define CELL_SIZE 10.f // The world is divided in cubic cells  
#define BOID_SCOPE 10.f // This is how far boids look for neighbours. Should always be == CELL_SIZE

// Boid attributes
#define PLANE_SOFTNESS 20.f
#define PLANE_AVOID_DISTANCE 40.f
#define DEATH_DISTANCE 3.5f
#define MAX_ACCELERATION 0.07f
#define MAX_ACCELERATION_PREDATOR 0.05f
#define SEPARATION_SOFTNESS 2.0f

#define SEPARATION_FACTOR 8.f
#define ALIGN_FACTOR 1.f
#define COHESION_FACTOR 0.1f 
#define AVOID_FACTOR 30.0f
#define REPELLATION_FACTOR 20.f
#define PREY_ATTRACT_FACTOR 2.0f
#define PRED_AVOID_FACTOR 12.f

#define AVAILABLE_ACCELERATION 2.8f

#define MIN_SPEED 0.8f
#define MAX_SPEED 1.8f

extern glm::vec3 cameraDir;
extern glm::vec3 cameraPos;
extern bool isLaserActive;

Boid** initBoidsOnGPU(Boid*);
void deinitBoidsOnGPU(void);
void step();
void prepareBoidRender(Boid* boids, glm::vec3* renderBoids, glm::mat4 projection, glm::mat4 view);
void printCUDAError();

void mapBufferObjectCuda( struct cudaGraphicsResource** positionsVBO_CUDA, size_t* num_bytes, glm::vec3** positions);

#endif