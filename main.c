#include <algorithm>
#include <iostream>
#include <vector>
#include <list>
#include <random>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>  
#include <glm/gtc/type_ptr.hpp>

#include <cuda_gl_interop.h>


#include "Shader.h"
#include "boid.h"
#include "kernel.h"



// Uncomment to enable timing
//#define TIMING
const int FRAMES_MEASURED = 500;
int frame = 0;

GLFWwindow* window;

// Define screen size
const unsigned int screenWidth = 1920, screenHeight = 1080;

// Camera variables
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraDir = glm::vec3(0.0f, 0.0f, 2.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
float fov = 45.0f;

// Boids avoid a "laser" ray when right clicking
bool isLaserActive = false;

// Timing
unsigned int currentFrame = 0;
unsigned int lastFrame = 0;
unsigned int deltaTime = 0;	// time between current frame and last frame

// How many boids on screen
const int nrBoids = NR_BOIDS;
const int nrPredators = 1000;
Boid* boids;

// Define the light sources position
glm::vec3 lightPos1(350.0f, 700.0f, 350.0f);
glm::vec3 lightPos2(50.0f, 700.0f, 350.0f);


// Process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	isLaserActive = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;

	float cameraSpeed = 150 * deltaTime;
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraDir;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraDir;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraDir, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraDir, cameraUp)) * cameraSpeed;

}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	static bool firstMouse = true;
	static float yaw = -90.0f;	// Yaw is initialized to -90.0 degrees since a yaw of 0.0 results in a direction vector pointing to the right so we initially rotate a bit to the left.
	static float pitch = 0.0f;
	static float lastX = 800.0f / 2.0;
	static float lastY = 600.0 / 2.0;

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
	lastX = xpos;
	lastY = ypos;

	float sensitivity = 0.1f; // change this value to your liking
	xoffset *= sensitivity;
	yoffset *= sensitivity;

	yaw += xoffset;
	pitch += yoffset;

	// make sure that when pitch is out of bounds, screen doesn't get flipped
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;

	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
	cameraDir = glm::normalize(front);
}

// GLFW: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

int initGLFW()
{
	// Initialize and configure
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Needed for OS X, and possibly Linux
	glfwWindowHint(GLFW_SAMPLES, 8); // Smoothen edges

	// Window creation
	window = glfwCreateWindow(screenWidth, screenHeight, "BoidSim", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouseCallback);

	// GLFW catches the cursor
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// GLAD: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	return 1;
}

// Initialize boids with random positions and velocities
void initBoidsPosAndVel(Boid* boids) {
	static std::random_device rd;
	static std::uniform_real_distribution<double> dist(CELL_SIZE + 1.f, MAX_COORD - CELL_SIZE);
	static std::uniform_real_distribution<double> dist2(-MAX_SPEED, MAX_SPEED);

	for (int i = 0; i < nrBoids; i++) {

		boids[i] = Boid();

		boids[i].position.x = dist(rd);
		boids[i].position.y = dist(rd);
		boids[i].position.z = dist(rd);
		boids[i].velocity.x = dist2(rd);
		boids[i].velocity.y = dist2(rd);
		boids[i].velocity.z = dist2(rd);

	}
	
	// Set predator flag for predators
	for (int i = 0; i < nrPredators; i++) {
		boids[i].status |= PREDATOR_FLAG;
	}
}




int main()
{

	double updateTime = 0.;
	double renderTime = 0.;

    boids = *(initBoidsOnGPU(boids));
    // GLFW: initialize and configure
	initGLFW();

    // Build and compile shader programs
    Shader shader("vert.shader", "frag.shader");
	Shader lampShader("lamp.vert", "lamp.frag");

	// Initialize boids' (random) positions and velocities
	initBoidsPosAndVel(boids);

	// Create a vertex array object
    unsigned int VAO;
	glGenVertexArrays(1, &VAO);

    // Explicitly set device 0
	cudaSetDevice(0);

    // Create buffer object and register it with CUDA
	GLuint positionsVBO;
	struct cudaGraphicsResource* positionsVBO_CUDA;
    glGenBuffers(1, &positionsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    unsigned int size = nrBoids * sizeof(glm::vec3) * 54;
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsRegisterFlagsWriteDiscard);

    // Use the shader created earlier so we can attach matrices
    shader.use();

    // Instantiate transformation matrices
    glm::mat4 projection, view;
    
	// Projection will always be the same: define FOV, aspect ratio and view frustum (near & far plane)
    projection = glm::perspective(glm::radians(45.0f), (float)screenWidth / screenHeight, 0.1f, 1500.0f);
    
	// Set projection matrix as uniform (attach to bound shader)
    shader.setMatrix("projection", projection);
	
	// Set the light sources positions
	shader.setVec3("lightPos1", lightPos1);
	shader.setVec3("lightPos2", lightPos2);

	// Draw the light sources
	lampShader.use();
	lampShader.setMatrix("projection", projection);
	
	unsigned int lightVAO;
	glGenVertexArrays(1, &lightVAO);
	glBindVertexArray(lightVAO);


	// Enable z-test 
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE); // smoother edges

    // Render loop
    while (!glfwWindowShouldClose(window))
    {
		currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
        
		// Keyboard and mouse input
        processInput(window);

        // update camera direction, rotation
		projection = glm::perspective(glm::radians(fov), (float)screenWidth / screenHeight, 0.1f, 1000.0f);
		shader.use();
		shader.setMatrix("projection", projection);
		
		// set view matrix
		view = glm::lookAt(cameraPos, cameraPos + cameraDir, cameraUp);
		shader.setMatrix("view", view);
        
		// clear whatever was on screen last frame
		glm::vec3 bgColor(5.f/255.f, 30.f/255.f, 62.f/255.f);
        glClearColor(bgColor.x, bgColor.y, bgColor.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

		// Time one step of the simulation
		double lastTime = glfwGetTime();
        
		// Perform one step of the simulation on the GPU
		step();

		updateTime += glfwGetTime()- lastTime;
		lastTime = glfwGetTime();

        // Map buffer object so CUDA can access OpenGl buffer
        glm::vec3* renderBoids;
        size_t num_bytes;
        mapBufferObjectCuda(&positionsVBO_CUDA, &num_bytes, &renderBoids);

        //Execute kernel HERE 
        prepareBoidRender(boids, renderBoids, projection, view);
        printCUDAError();

        // Unmap buffer object so OpenGl can access the buffer again
		cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);

        // Render from buffer object
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Bind buffer object and boid array
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
		glEnableVertexAttribArray(2);

        // Draw 3 * nrBoids vertices
		shader.setVec3("color", glm::vec3(1.0f, 1.0f, 1.0f));
		shader.setVec3("bgColor", bgColor);
		shader.setVec3("cameraPos", cameraPos);
        glDrawArrays(GL_TRIANGLES, 0, nrBoids * 24);


		// Draw light source
		lampShader.use();
		lampShader.setMatrix("view", view);
		
		glm::mat4 lightModel = glm::scale(glm::translate(glm::mat4(1.0f), lightPos1), glm::vec3(20.f));
		lampShader.setMatrix("model", lightModel);
		glBindVertexArray(lightVAO);

		glm::mat4 lightModel2 = glm::scale(glm::translate(glm::mat4(1.0f), lightPos2), glm::vec3(20.f));
		lampShader.setMatrix("model", lightModel2);
		glBindVertexArray(lightVAO);

        // unbind buffer and vertex array
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        glfwSwapBuffers(window);
        glfwPollEvents();

		#ifdef TIMING
		renderTime += glfwGetTime() - lastTime;
		if (++frame > FRAMES_MEASURED) break;
		#endif	
    }

	cudaGraphicsUnregisterResource(positionsVBO_CUDA);
    glDeleteBuffers(1, &positionsVBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    glfwTerminate();
    deinitBoidsOnGPU();
	
	#ifdef TIMING
	printf("Update time: %f\n", updateTime / (double)FRAMES_MEASURED);
	printf("Render time: %f\n", renderTime / (double)FRAMES_MEASURED);
	while (1);
	#endif
	
    return 0;
}

