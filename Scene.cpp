#include "Scene.h"

Scene::Scene(Camera* camera) : sceneCam(camera) {}

Scene::~Scene() {}

void Scene::addObject(Mesh& mesh, Shader& shader, glm::vec2 modelMatrix) {
    mesh.addInstance(modelMatrix);

    Mesh* meshPtr = &mesh;
    Shader* shaderPtr = &shader;

    if (shaderMap.find(shaderPtr) != shaderMap.end()) {
        if (shaderMap[shaderPtr].find(meshPtr) == shaderMap[shaderPtr].end()) {
            shaderMap[shaderPtr].insert(meshPtr);
        }
    }
    else {
		shaderMap[shaderPtr].insert(meshPtr);
        shaderMap[shaderPtr] = { meshPtr };
    }
}

	void Scene::Render(glm::vec3 lightPos, glm::vec4 lightColor, float scale) {
    for (auto it = shaderMap.begin(); it != shaderMap.end(); ++it) {
        Shader* shader = it->first;
        std::set<Mesh*> meshes = it->second;

        shader->Activate();
		glUniform1f(glGetUniformLocation(shader->ID, "scale"), scale);
		glUniform4f(glGetUniformLocation(shader->ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
		glUniform3f(glGetUniformLocation(shader->ID, "lightPos"), lightPos.x, lightPos.y, lightPos.z);
        sceneCam->Matrix(*shader, "camMatrix");

        for (auto& mesh : meshes) {
            mesh->Draw();
        }
    }
}

