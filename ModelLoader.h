#define GLM_ENABLE_EXPERIMENTAL
#ifndef MODEL_CLASS_H
#define MODEL_CLASS_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <string>
#include "VBO.h"

class Model
{
private:
    std::vector<Vertex> vertices;
    std::vector<GLuint> indices;
public:
    Model() = default;

    bool load(const std::string& filename);

    std::vector<Vertex>& getVertices()  { return vertices; }
    std::vector<unsigned int>& getIndices() { return indices; }
};

#endif
