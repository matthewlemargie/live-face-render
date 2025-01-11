#ifndef MESH_CLASS_H
#define MESH_CLASS_H

#include <string>
#include <vector>
#include <glm/glm.hpp>

#include "ModelLoader.h"
#include "VAO.h"
#include "EBO.h"
#include "Camera.h"

class Mesh
{
public:
    glm::mat4 model = glm::mat4(1.0f);
	glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 Orientation = glm::vec3(0.0f, 0.0f, 1.0f);

    float speed = 0.025f;
    float sensitivity = 100.0f;

    bool firstClick = true;

    std::vector<Vertex> vertices;
    std::vector<GLuint> indices;
    std::vector<glm::mat4> instanceMats;
    std::vector<glm::vec2> instanceVecs;

    VAO vao;
    VBO vbo;

    int numInstances = 0;

    Mesh(const std::string& filename);

    void addInstance(glm::vec2 instanceVecs);
    void resetInstances(float* instances);
    void Draw();
    void Inputs(GLFWwindow* window);
};

#endif

