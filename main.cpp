#define GLM_ENABLE_EXPERIMENTAL
#include <GL/glew.h>
#include <GL/gl.h>
#include <glm/gtx/string_cast.hpp>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

#include "ModelLoader.h"
#include "Mesh.h"
#include "Camera.h"
#include "shaderClass.h"
#include "Scene.h"

struct Landmark {
    float x, y, z;
};

int main()
{
    const char* SHM_NAME = "landmarks_shm";
    const size_t BUFFER_SIZE = 65536;  // Must match the Python buffer size

    // Open the shared memory
    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (shm_fd == -1) {
        std::cerr << "Failed to open shared memory." << std::endl;
        return -1;
    }

    // Map the shared memory
    void* shm_ptr = mmap(0, BUFFER_SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory." << std::endl;
        return -1;
    }

	GLFWwindow* window;

	//Initialize the library
	if (!glfwInit())
		return -1;

	//initiliaze monitor/window/mode
	GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	if (!monitor) {
		std::cerr << "Failed to get primary monitor!" << std::endl;
		glfwTerminate();
		return -1;
	}
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);
	if (!mode) {
		std::cerr << "Failed to get video mode!" << std::endl;
		glfwTerminate();
		return -1;
	}
	window = glfwCreateWindow(mode->width, mode->height, "OpenGL Window", monitor, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glViewport(0, 0, mode->width, mode->height);

	//initilize GLEW
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW!" << std::endl;
		return -1;
	}

	Mesh box("box.obj");
	Shader shader("face.vert", "face.frag");
	Camera camera(mode->width, mode->height, glm::vec3(0.0f, 0.0f, -10.0f), 90.0f, 1.0f, 1000.0f);
	Scene scene(&camera);
	scene.addObject(box, shader, glm::vec2(1.0f));

	glm::vec3 lightPos = glm::vec3(1.0f, 1.0f, 1.0f);
	glm::vec4 lightColor = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);

	float scale = 0.05f;

	glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
	glEnable(GL_DEPTH_TEST);


	// Loop for returning frames from the webcam
    try
    {
        while (!(glfwWindowShouldClose(window)))
        {
            GLfloat positions[468 * 3];
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            float* data = static_cast<float*>(shm_ptr);

            // Read landmarks
            std::vector<Landmark> landmarks;
            size_t i = 0;
            // while (data[i] != 0.0f) {  // Null-terminated data
                // positions[i] = data[i];
                // std::cout << positions[i] << std::endl;
                // i += 1;
            // }
            for (int i = 0; i < 468 * 3; i++)
            {
                positions[i] = data[i];
            }
            
            shader.Activate();
            glUniform1f(glGetUniformLocation(shader.ID, "scale"), scale);
            glUniform4f(glGetUniformLocation(shader.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
            glUniform3f(glGetUniformLocation(shader.ID, "lightPos"), lightPos.x, lightPos.y, lightPos.z);

            camera.Inputs(window);
            camera.updateMatrix();
            camera.Matrix(shader, "camMatrix");

            box.vbo.resetInstances(positions);
            box.Draw();
            glfwSwapBuffers(window);
            glfwPollEvents();
            // usleep(30000);
        }
    } catch (...) {
        std::cerr << "Error reading from shared memory." << std::endl;
    }

	// Release the webcam and close all OpenCV windows
    munmap(shm_ptr, BUFFER_SIZE);
    close(shm_fd);
	glfwTerminate();

	return 0;
}




