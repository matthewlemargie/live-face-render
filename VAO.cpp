#include "VAO.h"

VAO::VAO()
{
	glGenVertexArrays(1, &ID);
}

void VAO::LinkAttrib(VBO& VBO, GLuint layout, GLuint numComponents, GLenum type, GLenum stride, void* offset)
{
	VBO.Bind();
	glVertexAttribPointer(layout, numComponents, type, GL_FALSE, stride, offset);
	glEnableVertexAttribArray(layout);
	glVertexAttribDivisor(layout, 0);
	VBO.Unbind();
}

void VAO::LinkInstance(VBO& VBO)
{
	//VBO.bindInstance();
	//for (GLuint i = 0; i < 4; ++i) {
	//	glVertexAttribPointer(2 + i, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(i * sizeof(glm::vec4)));
	//	glEnableVertexAttribArray(2 + i);
	//	glVertexAttribDivisor(2 + i, 1);
	//}
	//VBO.Unbind();
	VBO.bindInstance();
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(2);
	glVertexAttribDivisor(2, 1);
	VBO.Unbind();
}

void VAO::Bind()
{
	glBindVertexArray(ID);
}

void VAO::Unbind()
{
	glBindVertexArray(0);
}

void VAO::Delete()
{
	glDeleteVertexArrays(1, &ID);
}
