#ifndef	VAO_CLASS_H
#define	VAO_CLASS_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include "VBO.h"

class VAO
{
public:
	GLuint ID;
	int numInstances = 0;
	VAO();

	void LinkAttrib(VBO& VBO, GLuint layout, GLuint numComponents, GLenum type, GLenum stride, void* offset);
	void LinkInstance(VBO& VBO);
	void Bind();
	void Unbind();
	void Delete();
};

#endif
