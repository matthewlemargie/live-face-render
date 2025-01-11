#include "VBO.h"

VBO::VBO()
{
	glGenBuffers(1, &ID);
	glGenBuffers(1, &instanceID);
}

void VBO::addVertices(std::vector<Vertex>& vertices)
{
	glBindBuffer(GL_ARRAY_BUFFER, ID);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, instanceID);
	glBufferData(GL_ARRAY_BUFFER, 468 * 3 * sizeof(float), nullptr, GL_STATIC_DRAW);
}

void VBO::addInstance(std::vector<glm::mat4>& instanceMats)
{
	glBindBuffer(GL_ARRAY_BUFFER, instanceID);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * (numInstances % 100), sizeof(glm::mat4) * instanceMats.size(), instanceMats.data());
	numInstances += (instanceMats.size());
}

void VBO::resetInstances(void* instances)
{
	glBindBuffer(GL_ARRAY_BUFFER, instanceID);
	glBufferSubData(GL_ARRAY_BUFFER, 0, 468 * 3 * sizeof(float), instances);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VBO::Bind()
{
	glBindBuffer(GL_ARRAY_BUFFER, ID);
}

void VBO::bindInstance()
{
	glBindBuffer(GL_ARRAY_BUFFER, instanceID);
}

void VBO::Unbind()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VBO::Delete()
{
	glDeleteBuffers(1, &ID);
}
