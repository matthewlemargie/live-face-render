#pragma once
#include <vector>
#include <tuple>
#include "ModelLoader.h"
#include "Mesh.h"
#include "shaderClass.h"
#include "Camera.h"
#include "VAO.h"
#include <glm/mat4x4.hpp>
#include <map>
#include <set>

class Scene {
public:
    std::map<Shader*, std::set<Mesh*>> shaderMap;
    Camera* sceneCam;

    Scene(Camera* camera);
    ~Scene();

    void addObject(Mesh& mesh, Shader& shader, glm::vec2 modelMatrix);
    void Render(glm::vec3 lightPos, glm::vec4 lightColor, float scale);
};
