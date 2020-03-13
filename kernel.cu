#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <windows.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "kernel.h"
#include "boid.h"

// Uncomment to enable timing 
// #define TIMING

// Uncomment to use coordinate concatentation instead of Z-order for the uniform grid
//#define BIT_CONCATENATION

/*
TODO: Replacing glm datatypes with CUDA's can improve performance,
for example vec3 -> float3, distance() -> norm3df()
*/

using namespace cub;

// A useful macro for displaying CUDA errors 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = false)	
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


struct ObstaclePlane {
	glm::vec3 point, normal;

	ObstaclePlane(float p1, float p2, float p3, float n1, float n2, float n3)
		: point(p1, p2, p3), normal(n1, n2, n3) {}
};

// Array for the walls
ObstaclePlane* walls = NULL;

// These arrays hold the actual boids
extern Boid* boids;
Boid* boidsAlt = NULL;

// These arrays hold the (Z-order/morton encoded) cell ids
uint64_t* cellIDs = NULL;
uint64_t* cellIDsAlt = NULL;

// These arrays hold the boid IDs. boidsAlt is a alternate array needed for the radixSort
int* boidIDs = NULL;
int* boidIDsAlt = NULL;

// Doublebuffers containing boidIDs and cellIDs, these are used by the radix sort function
DoubleBuffer<uint64_t> cellIDsBuf;
DoubleBuffer<int> boidIDsBuf;

// The maximum (x/y/z) index a cell can have
const int MAX_CELL_INDEX = (int)MAX_COORD / CELL_SIZE;

// Calculate the maximum value of a Morton encoded (Z-ordered) cell ID
// Number of bits needed to represent the maximum coordinate
#define shiftBitK(x, k) (int) ((x&(1<<k)) << k*2+2 | (x&(1<<k)) << k*2+1 | (x&(1<<k)) << k*2)
#ifdef BIT_CONCATENATION
const int NR_CELLS = MAX_CELL_INDEX << MIN_SHIFT * 2 | MAX_CELL_INDEX << MIN_SHIFT | MAX_CELL_INDEX;
#else
const int NR_CELLS = shiftBitK(MAX_CELL_INDEX, 10)
| shiftBitK(MAX_CELL_INDEX, 9)
| shiftBitK(MAX_CELL_INDEX, 8)
| shiftBitK(MAX_CELL_INDEX, 7)
| shiftBitK(MAX_CELL_INDEX, 6)
| shiftBitK(MAX_CELL_INDEX, 5)
| shiftBitK(MAX_CELL_INDEX, 5)
| shiftBitK(MAX_CELL_INDEX, 4)
| shiftBitK(MAX_CELL_INDEX, 3)
| shiftBitK(MAX_CELL_INDEX, 2)
| shiftBitK(MAX_CELL_INDEX, 1)
| shiftBitK(MAX_CELL_INDEX, 0);
#endif

// When using bit concatenation, we shift coordinates using this value. It is calculated as log2(MAX_COORD)+1
#define MIN_SHIFT 7 
//static const int MIN_SHIFT = log2(MAX_COORD) + 1;

// These parameters are used by the CUDA kernels
int blockSize = 256;
int numBlocksBoids = (NR_BOIDS + blockSize - 1) / blockSize;
int numBlocksCells = (NR_CELLS + blockSize - 1) / blockSize;

// Time variable
float t = 0.f;

// A tempory storage for new velocities allows parallel processing of the boids velocities 
glm::vec3* newVelocities;

// These arrays hold the start and end indices the boids contained in each cell
int* cellStartIndex;
int* cellEndIndex;

// Get the cell based on the boids position
// TODO: floorf() on the individual coordinates might be faster!?
inline __device__ glm::vec3 getCell(glm::vec3 pos) {
	return glm::floor(pos * (1.0f / CELL_SIZE));
}

// The boids are given an ID, which simply is the index it has in the intial boid array before they're sorted
__global__ void initBoidIDs(int BoidIDs[], int nrBoids) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i >= nrBoids) return;
	BoidIDs[i] = i;
}

// Helper function that spreads out bits with two zeroes in between. Used to morton encode cell coordinates 
// Credit: mweerden (https://stackoverflow.com/questions/1024754/)
__device__ uint64_t spreadOutByThree(uint64_t i) {
	i = (i | (i << 16)) & 0x030000FF;
	i = (i | (i << 8)) & 0x0300F00F;
	i = (i | (i << 4)) & 0x030C30C3;
	i = (i | (i << 2)) & 0x09249249;
	return i;
}

// Hash cell coords to morton code with "magic numbers"
__device__ uint64_t mortonEncode(int x, int y, int z) {
	return spreadOutByThree((uint64_t)x) | (spreadOutByThree((uint64_t)y) << 1) | (spreadOutByThree((uint64_t)z) << 2);
}

// Hash cell coords to morton code with simple bit concatenation
__device__ uint64_t bitConcatenation(int x, int y, int z) {
	return ((uint64_t)x & 0x1FFFFF) << (MIN_SHIFT * 2) | ((uint64_t)y & 0x1FFFFF) << MIN_SHIFT | ((uint64_t)y & 0x1FFFFF);
}

// This function is used when scanning the sorted boids' cell-ids to see were cells starts and ends 
__global__ void detectCellIndexChange(int cellStarts[], int cellEnds[], uint64_t cellIDs[], int nrBoids) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i >= nrBoids) return;

	int cellID = cellIDs[i];
	int nextCellID = cellIDs[i + 1];

	if (i == 0) {
		// This is the case for the first element in the boid array 
		cellStarts[cellID] = i;
	}
	else if (i == nrBoids - 1) {
		// This is the case for the last element in the boid array
		cellEnds[cellID] = i;
		return;
	}
	if (cellID != nextCellID) {
		// A change in cell index was detected!
		cellStarts[nextCellID] = i + 1;
		cellEnds[cellID] = i;
	}
}

// Helper function for adding acceleration to a boid's velocity according to a force, IF there's any acceleration available
__device__ inline void extractAcceleration(float &acc, glm::vec3 force, glm::vec3 &newVel) {
	float magnitude = fminf(glm::length(force), acc);
	if (magnitude > 0.f) {
		newVel += magnitude * normalize(force);
	}
	acc -= magnitude;
}


// Update boid with index n
// TODO: Right now we use this function for both prey and predators, which makes it quite verbose
__global__ void computeVelocities(Boid boids[], int cellStarts[], int cellEnds[], uint64_t cellIDs[]
	, int nrBoids, Boid boidsUpdated[], ObstaclePlane walls[], glm::vec3 cameraPos, glm::vec3 cameraDir, bool isLaserActive) {

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	if (idx >= nrBoids) return;

	// Nr of neighbours
	float nrPrey = 0;
	float nrPredators = 0;

	// Current boid whose neighbours we're checking
	Boid b = boids[idx]; 

	// Decide which cell current boid is in
	glm::vec3 cell = getCell(b.position);

	// In order to avoid possible thread divergence caused by if/else clauses, we cast bools to floats
	// TODO: Not sure if this is necessary, the compiler can probably avoid thread divergence anyway
	float bIsPredator = (bool)(b.status & PREDATOR_FLAG);
	float bIsPrey = (bool)!(b.status & PREDATOR_FLAG);

	glm::vec3 oldVel = b.velocity;
	glm::vec3 newVel = glm::vec3(0);

	glm::vec3 avgPreyPos = glm::vec3(0);
	glm::vec3 avgPreyVel = b.velocity;
	glm::vec3 avgPredatorPos = glm::vec3(0);

	// The "forces" acting on a boid
	glm::vec3 separation = glm::vec3(0);
	glm::vec3 cohesion = glm::vec3(0);
	glm::vec3 alignment = glm::vec3(0);
	glm::vec3 predatorAvoidance = glm::vec3(0);
	glm::vec3 wallAvoidance = glm::vec3(0.f);
	glm::vec3 repellation = glm::vec3(0.f);

	// A predator follows the closest available prey
	glm::vec3 closestPreyPos = glm::vec3(0.0);
	bool preyFound = false;
	float epsilon = 0.f;

	// Start checking all 27 neighbouring cells
	// TODO: find out a clever way to iterate over cells in order of the morton code to get 
	// more coherent memory accesses
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {

				#ifdef BIT_CONCATENATION
				uint64_t cellID = bitConcatenation((int)cell.x + i, (int)cell.y + j, (int)cell.z + k);
				#else
				// calculate the (Morton encoded/Z-order) cell id based on cell coordinates
				uint64_t cellID = mortonEncode((int)cell.x + i, (int)cell.y + j, (int)cell.z + k);
				#endif

				if (cellStarts[cellID] == -1) {
					// cell is empty if it's start index is unchanged since last reset
					continue; 
				}
				// Iterate over all boids in neighbouring cell
				for (int l = cellStarts[cellID]; l <= cellEnds[cellID]; l++) {
					Boid n = boids[l];

					glm::vec3 deltaPos = n.position - b.position;
					float distance = glm::length(deltaPos);

					// Exclude neighbours that are outside boid's scope 
					float validNeighbour = (distance > epsilon && distance < BOID_SCOPE);
					
					// Keep count of number of predators and prey
					float nIsPredator = validNeighbour * (bool)(n.status & PREDATOR_FLAG);
					float nIsPrey = validNeighbour * (!(bool)(n.status & PREDATOR_FLAG));
					nrPrey += nIsPrey;
					nrPredators += nIsPredator;

					// Check average position of neighbours
					avgPreyPos += nIsPrey * n.position;
					avgPreyVel += nIsPrey * n.velocity;
					avgPredatorPos += nIsPredator * n.position;

					// Prey behavior "forces"
					if (validNeighbour > 0) {
						separation -= (bIsPrey * nIsPrey) * deltaPos / (distance*distance);
						predatorAvoidance -= (nIsPredator) * deltaPos / sqrt(distance);
					}

					// Predator beavior "forces" 
					if (nIsPrey > 0 && (distance < glm::distance(b.position, closestPreyPos) || !preyFound)) {
						closestPreyPos = n.position;
						preyFound = true;
					}
				}
			}
		}
	}

	if (isLaserActive) {
		glm::vec3 point = cameraPos + dot(b.position - cameraPos, cameraDir) * cameraDir;
		float distance = glm::distance(b.position, point);
		repellation = (float)(distance < 30.f) * normalize(b.position - point) / distance;
	}

	// Avoid walls
	for (int m = 0; m < 6; m++) {
		ObstaclePlane o = walls[m];

		// Calculate a vector from plane orig point to the boid
		glm::vec3 v = b.position - o.point;

		// Take the dot product of that vector with the unit normal vector n
		float dist = glm::dot(v, o.normal);

		// Inverse distance from plane
		float inverse_dist = 1.f / dist;
		
		// Square the inverse (power law!)
		inverse_dist *= inverse_dist;
		
		// Avoid wall if it's withing plane avoid distance
		wallAvoidance += PLANE_SOFTNESS * inverse_dist * o.normal * (float)(dist < PLANE_AVOID_DISTANCE);
	}



	avgPreyPos /= nrPrey;
	avgPreyVel /= nrPrey;
	separation /= nrPrey;
	cohesion += (avgPreyPos - b.position) * bIsPrey;
	alignment += (avgPreyVel - b.velocity) * bIsPrey;
	if (nrPredators > 0 && bIsPrey) {
		avgPredatorPos /= nrPredators;
		predatorAvoidance += b.position - avgPredatorPos;
	}
	glm::vec3 preyAttraction = (avgPreyPos - b.position) * bIsPredator;

	// Arbitrate available acceleration to the different "forces" 
	float acc = AVAILABLE_ACCELERATION;
	extractAcceleration(acc, wallAvoidance * AVOID_FACTOR, newVel);
	extractAcceleration(acc, predatorAvoidance * PRED_AVOID_FACTOR, newVel);
	extractAcceleration(acc, repellation* REPELLATION_FACTOR, newVel);
	extractAcceleration(acc, separation* SEPARATION_FACTOR, newVel);
	extractAcceleration(acc, cohesion* COHESION_FACTOR, newVel);
	extractAcceleration(acc, alignment* ALIGN_FACTOR, newVel);
	extractAcceleration(acc, preyAttraction * PREY_ATTRACT_FACTOR, newVel);



	// Limit acceleration. Prey has some "extra" acceleration available when predators are close
	glm::vec3 acceleration = newVel - oldVel;
	float maxAcceleration = MAX_ACCELERATION * (1 + 0.7f * (nrPredators > 0) * bIsPrey);
	float accelerationMag = glm::clamp(glm::length(acceleration), 0.f, maxAcceleration);

	//Also limit velocity. We need to check if acceleration > 0 since we can't normalize a zero vector
	if (accelerationMag > epsilon) {
		newVel = oldVel + glm::normalize(acceleration)*accelerationMag;
		float speed = glm::clamp(glm::length(newVel), MIN_SPEED, MAX_SPEED * (1 + (nrPredators > 0)*bIsPrey));
		newVel = glm::normalize(newVel) * speed;
	}
	else {
		newVel = oldVel;
	}

	// Update position 
	glm::vec3 newPos = b.position + newVel;

	// TODO: Right now we wrap the boids around a cube
	newPos.x = newPos.x < CELL_SIZE ? MAX_COORD - CELL_SIZE : newPos.x;
	newPos.y = newPos.y < CELL_SIZE ? MAX_COORD - CELL_SIZE : newPos.y;
	newPos.z = newPos.z < CELL_SIZE ? MAX_COORD - CELL_SIZE : newPos.z;

	newPos.x = newPos.x > MAX_COORD - CELL_SIZE ? CELL_SIZE : newPos.x;
	newPos.y = newPos.y > MAX_COORD - CELL_SIZE ? CELL_SIZE : newPos.y;
	newPos.z = newPos.z > MAX_COORD - CELL_SIZE ? CELL_SIZE : newPos.z;

	boidsUpdated[idx].position = newPos;
	boidsUpdated[idx].velocity = newVel;
	boidsUpdated[idx].status = b.status;
}


// Sets all the cell start/end indices to -1, so no old values is left
// TODO: only reset the ones that actually has had boids in it?
__global__ void resetCellRanges(int cellStarts[], int cellEnds[], int nrCells) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i < nrCells) {
		cellStarts[i] = -1;
		cellEnds[i] = -1;
	}
}

// Stores the Morton code/Z-order value for each boid, based on the coordinates of the 
// cell which the boid is currently in
__global__ void calculateCellID(int n, uint64_t cellIDs[], Boid b[], int nrBoids) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	glm::vec3 cell = getCell(b[i].position);
	#ifdef BIT_CONCATENATION
	cellIDs[i] = bitConcatenation((int)cell.x, (int)cell.y, (int)cell.z);
	#else
	cellIDs[i] = mortonEncode((int)cell.x, (int)cell.y, (int)cell.z);
	#endif
}

// After boid IDs are sorted, the array with the actual boid structs are sorted accordingly with this function
__global__ void rearrangeBoids(int boidIDs[], Boid boids[], Boid boidsAlt[], int nrBoids) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i >= nrBoids) return;
	boidsAlt[i] = boids[boidIDs[i]]; // copy over boids to the boidsAlt array, which in the end will be sorted
}

// Calculate
__global__ void prepareBoidRenderKernel(Boid* boids, glm::vec3* renderBoids, glm::mat4 projection, glm::mat4 view, float t) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	int j = i * 54;
	if (i >= NR_BOIDS) return;
	Boid b = boids[i];

	// This makes the wings flap, sort of
	float wingHeight = cosf(t/5.f + b.position.x) - 0.6f;

	// One vector for each vertex
	glm::vec3 p1(0.0f, 1.5f, -1.73205080757f);
	glm::vec3 p0(-1.0f, -1.0f, wingHeight);
	glm::vec3 p2(0.0f, -1.0f, -1.73205080757f);
	glm::vec3 p3(1.0f, -1.0f, wingHeight);
	glm::vec3 p4(0.0f, -1.0f, -1.33205080757f); // Last point is less away than back point p3

	// Default boid color. Dirty quickfix
	glm::vec3 color(198.f / 255.f, 231.f / 255.f, 1.0f);

	// Create model matrix from boid position
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, b.position);
	glm::vec3 v = glm::vec3(b.velocity.z, 0, -b.velocity.x);
	float angle = acosf(b.velocity.y / glm::length(b.velocity));
	model = glm::rotate(model, angle, v);

	// Color predators red
	if (b.status & PREDATOR_FLAG) {
		color = glm::vec3(133.f / 255.f, 30.f / 255.f, 62.f / 255.f);
		model = glm::scale(model, glm::vec3(3.0f));
	}

	glm::vec3 v0 = view * model * glm::vec4(p0, 1.0f);
	glm::vec3 v1 = view * model * glm::vec4(p1, 1.0f);
	glm::vec3 v2 = view * model * glm::vec4(p2, 1.0f);
	glm::vec3 v3 = view * model * glm::vec4(p3, 1.0f);
	glm::vec3 v4 = view * model * glm::vec4(p4, 1.0f);

	glm::vec3 n0 = glm::mat3(glm::transpose(glm::inverse(model))) * glm::cross(p2 - p0, p1 - p0); // Left wing top
	glm::vec3 n1 = glm::mat3(glm::transpose(glm::inverse(model))) * glm::cross(p1 - p0, p4 - p0); // Left wing bottom 
	glm::vec3 n2 = glm::mat3(glm::transpose(glm::inverse(model))) * glm::cross(p3 - p2, p1 - p2); // Right wing top
	glm::vec3 n3 = glm::mat3(glm::transpose(glm::inverse(model))) * glm::cross(p1 - p4, p3 - p4); // Right wing bottom
	glm::vec3 n4 = glm::mat3(glm::transpose(glm::inverse(model))) * glm::cross(p4 - p0, p2 - p0); // Left back
	glm::vec3 n5 = glm::mat3(glm::transpose(glm::inverse(model))) * glm::cross(p2 - p3, p4 - p3); // Right back

	// Left wing top
	renderBoids[j + 0] = v0;
	renderBoids[j + 1] = color;
	renderBoids[j + 2] = n0;
	renderBoids[j + 3] = v1;
	renderBoids[j + 4] = color;
	renderBoids[j + 5] = n0;
	renderBoids[j + 6] = v2;
	renderBoids[j + 7] = color;
	renderBoids[j + 8] = n0;

	// Left wing bottom
	renderBoids[j + 9] = v0;
	renderBoids[j + 10] = color;
	renderBoids[j + 11] = n1;
	renderBoids[j + 12] = v4;
	renderBoids[j + 13] = color;
	renderBoids[j + 14] = n1;
	renderBoids[j + 15] = v1;
	renderBoids[j + 16] = color;
	renderBoids[j + 17] = n1;

	// Right wing top
	renderBoids[j + 18] = v1;
	renderBoids[j + 19] = color;
	renderBoids[j + 20] = n2;
	renderBoids[j + 21] = v3;
	renderBoids[j + 22] = color;
	renderBoids[j + 23] = n2;
	renderBoids[j + 24] = v2;
	renderBoids[j + 25] = color;
	renderBoids[j + 26] = n2;

	// Right wing bottom
	renderBoids[j + 27] = v1;
	renderBoids[j + 28] = color;
	renderBoids[j + 29] = n3;
	renderBoids[j + 30] = v4;
	renderBoids[j + 31] = color;
	renderBoids[j + 32] = n3;
	renderBoids[j + 33] = v3;
	renderBoids[j + 34] = color;
	renderBoids[j + 35] = n3;

	// Left back
	renderBoids[j + 36] = v3;
	renderBoids[j + 37] = color;
	renderBoids[j + 38] = n4;
	renderBoids[j + 39] = v4;
	renderBoids[j + 40] = color;
	renderBoids[j + 41] = n4;
	renderBoids[j + 42] = v2;
	renderBoids[j + 43] = color;
	renderBoids[j + 44] = n4;

	// Right back
	renderBoids[j + 45] = v2;
	renderBoids[j + 46] = color;
	renderBoids[j + 47] = n5;
	renderBoids[j + 48] = v4;
	renderBoids[j + 49] = color;
	renderBoids[j + 50] = n5;
	renderBoids[j + 51] = v0;
	renderBoids[j + 52] = color;
	renderBoids[j + 53] = n5;
}

void prepareBoidRender(Boid* boids, glm::vec3* renderBoids, glm::mat4 projection, glm::mat4 view) {
	prepareBoidRenderKernel << < numBlocksBoids, blockSize >> > (boids, renderBoids, projection, view, t);
}

void printCUDAInfo() {
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
}

void printCUDAError() {
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
		printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
		printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

// Allocates memory on the GPU for the boids, walls and cells. 
// Returns a pointer to the boid array so the CPU can populate it with data
__host__ Boid** initBoidsOnGPU(Boid* boidsArr) {
	
	printCUDAInfo();
	boids = boidsArr; // TODO: clean up all "boids" pointers

	// Allocate memory for the walls
	gpuErrchk(cudaMallocManaged((void**)&walls, sizeof(ObstaclePlane) * 6));
	
	// Allocate memory for the cell index arrays
	gpuErrchk(cudaMallocManaged((void**)&cellStartIndex, sizeof(int) * NR_CELLS));
	gpuErrchk(cudaMallocManaged((void**)&cellEndIndex, sizeof(int) * NR_CELLS));
	
	// Allocate memory for the temp storage of new velocities
	gpuErrchk(cudaMallocManaged((void**)&newVelocities, sizeof(glm::vec3) * NR_BOIDS));
	
	// Allocate memory for the boids
	gpuErrchk(cudaMallocManaged((void**)&boidsAlt, sizeof(Boid) * NR_BOIDS));
	gpuErrchk(cudaMallocManaged((void**)&boids, sizeof(Boid) * NR_BOIDS));
	
	// Allocate memory for the buffer arrays
	gpuErrchk(cudaMallocManaged((void**)&cellIDs, sizeof(*cellIDs) * NR_BOIDS));
	gpuErrchk(cudaMallocManaged((void**)&cellIDsAlt, sizeof(*cellIDsAlt) * NR_BOIDS));
	gpuErrchk(cudaMallocManaged((void**)&boidIDs, sizeof(*boids) * NR_BOIDS));
	gpuErrchk(cudaMallocManaged((void**)&boidIDsAlt, sizeof(*boidIDsAlt) * NR_BOIDS));

	cellIDsBuf = DoubleBuffer<uint64_t>(cellIDs, cellIDsAlt);
	boidIDsBuf = DoubleBuffer<int>(boidIDs, boidIDsAlt);

	float max = MAX_COORD;
	float min = 0;
	
	// Right wall
	walls[0] = ObstaclePlane(max, max/2.f, max / 2.f, -1, 0, 0);
	// Left wall
	walls[1] = ObstaclePlane(min, max / 2.f, max / 2.f, 1, 0, 0);
	// Ceiling
	walls[2] = ObstaclePlane(max / 2.f, max, max / 2.f, 0, -1, 0);
	// Floor
	walls[3] = ObstaclePlane(max / 2.f, min, max / 2.f, 0, 1, 0);
	// Back wall
	walls[4] = ObstaclePlane(max / 2.f, max / 2.f, max, 0, 0, -1);
	// Front wwall
	walls[5] = ObstaclePlane(max / 2.f, max / 2.f, min, 0, 0, 1);

	return &boids;
}

__host__ void deinitBoidsOnGPU() {
	// Free memory
	cudaFree(cellStartIndex);
	cudaFree(cellEndIndex);
	cudaFree(cellIDsBuf.d_buffers[0]);
	cudaFree(cellIDsBuf.d_buffers[1]);
	cudaFree(boidIDsBuf.d_buffers[0]);
	cudaFree(boidIDsBuf.d_buffers[1]);
	cudaFree(newVelocities);
	cudaFree(boids);
	cudaFree(boidsAlt);
}


void mapBufferObjectCuda(struct cudaGraphicsResource** positionsVBO_CUDA, size_t* num_bytes, glm::vec3** positions) {
	cudaGraphicsMapResources(1, positionsVBO_CUDA, 0);
	cudaGraphicsResourceGetMappedPointer((void**)positions, num_bytes, *positionsVBO_CUDA);
}

// Perform one step in the simulation
void step() {
	
	t += 1.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	#ifdef TIMING
	cudaEventRecord(start);
	#endif 

	// Initialize boid id's
	initBoidIDs << < numBlocksBoids, blockSize >> > (boidIDsBuf.Current(), NR_BOIDS);
	
	#ifdef TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("initBoids: %f\n", milliseconds);
	#endif 


	#ifdef TIMING
	cudaEventRecord(start);
	#endif 

	// Calculate cell IDs for every boid
	calculateCellID << < numBlocksBoids, blockSize >> > (NR_BOIDS, cellIDsBuf.Current(), boids, NR_BOIDS);

	#ifdef TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("calculateCellID: %f\n", milliseconds);
	#endif 


	#ifdef TIMING
	cudaEventRecord(start);
	#endif 

	// Reset cell ranges
	resetCellRanges << < numBlocksCells, blockSize >> > (cellStartIndex, cellEndIndex, NR_CELLS);

	#ifdef TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("resetCellRanges: %f\n", milliseconds);
	#endif 

	// Determine temporary device storage requirements
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;

	#ifdef TIMING
	cudaEventRecord(start);
	#endif 
	
	// Determine temporary storage need
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, cellIDsBuf, boidIDsBuf, NR_BOIDS);

	// Allocate temporary storage
	// TODO: cudaMalloc is expensive, is it possible to do this particular allocation only once and reuse it? 
	gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

	// Run sorting operation
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, cellIDsBuf, boidIDsBuf, NR_BOIDS);

	cudaFree(d_temp_storage);
	
	#ifdef TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Sorting: %f\n", milliseconds);
	#endif 

	#ifdef TIMING
	cudaEventRecord(start);
	#endif 

	// Rearrange the actual boids based on the sorted boidIDs
	rearrangeBoids << < numBlocksBoids, blockSize >> > (boidIDsBuf.Current(), boids, boidsAlt, NR_BOIDS);
	// After rearranging the boids, we now work on the boidsAlt

	#ifdef TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("rearrangeBoids: %f\n", milliseconds);
	#endif 


	#ifdef TIMING
	cudaEventRecord(start);
	#endif 

	// Check were cellID changes occurs in the sorted boids array
	detectCellIndexChange << < numBlocksBoids, blockSize >> > (cellStartIndex, cellEndIndex, cellIDsBuf.Current(), NR_BOIDS);

	#ifdef TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("detectCellIndexChange: %f\n", milliseconds);
	#endif 

	#ifdef TIMING
	cudaEventRecord(start);
	#endif 

	// Update boid velocities based on the rules
	computeVelocities << < numBlocksBoids, blockSize >> > (boidsAlt, cellStartIndex, cellEndIndex, cellIDsBuf.Current(), NR_BOIDS, boids, walls, cameraPos, cameraDir, isLaserActive);

	#ifdef TIMING
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("computeVelocities: %f\n\n", milliseconds);
	#endif 

	// Swap the boids array pointer, so 'boids' now points to a sorted array
	cudaDeviceSynchronize(); 

}