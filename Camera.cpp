#include"Camera.h"

Camera::Camera(int width, int height, glm::vec3 position, float FOVdeg, float nearPlane, float farPlane)
{
	Camera::width = width;
	Camera::height = height;
	Camera::Position = position;
	Camera::initialPosition = position;
	Camera::mouseX = 0.0f;
	Camera::mouseY = 0.0f;
	Camera::FOVdeg = FOVdeg;
	Camera::nearPlane = nearPlane;
	Camera::farPlane = farPlane;

	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, 72 * sizeof(GLfloat), nullptr, GL_STATIC_DRAW); // Allocate space
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);
}

void Camera::updateMatrix()
{
	view = glm::lookAt(Position, Position + Orientation, Up);
	float Width = width;
	float Height = height;
	float aspectRatio = Width / Height;
	projection = glm::perspective(glm::radians(FOVdeg), aspectRatio, nearPlane, farPlane);

	cameraMatrix = projection * view;
}

void Camera::Matrix(Shader& shader, const char* uniform)
{
	glUniformMatrix4fv(glGetUniformLocation(shader.ID, uniform), 1, GL_FALSE, glm::value_ptr(cameraMatrix));
}

void Camera::drawFrustum(Shader rayShader, glm::mat4 proj, glm::mat4 vw) {
	hNear = nearPlane * tan(glm::radians(FOVdeg / 2.0f));
	wNear = hNear * ((float)width / height);
	hFar = farPlane * tan(glm::radians(FOVdeg / 2.0f));
	wFar = hFar * ((float)width / height);

	GLfloat vertices[] = {
		// Near Plane
		-wNear,  hNear, -nearPlane,
		 wNear,  hNear, -nearPlane,

		 wNear,  hNear, -nearPlane,
		 wNear, -hNear, -nearPlane,

		 wNear, -hNear, -nearPlane,
		-wNear, -hNear, -nearPlane,

		-wNear, -hNear, -nearPlane,
		-wNear,  hNear, -nearPlane,

		// Far Plane
		-wFar,  hFar, -farPlane,
		 wFar,  hFar, -farPlane,

		 wFar,  hFar, -farPlane,
		 wFar, -hFar, -farPlane,

		 wFar, -hFar, -farPlane,
		-wFar, -hFar, -farPlane,

		-wFar, -hFar, -farPlane,
		-wFar,  hFar, -farPlane,

		// Connect Near and Far Planes
		-wNear,  hNear, -nearPlane,
		-wFar,   hFar,  -farPlane,

		 wNear,  hNear, -nearPlane,
		 wFar,   hFar,  -farPlane,

		 wNear, -hNear, -nearPlane,
		 wFar,  -hFar,  -farPlane,

		-wNear, -hNear, -nearPlane,
		-wFar,  -hFar,  -farPlane,
	};
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), &vertices);
	rayShader.Activate();
	glm::mat4 camMat = proj * vw;
	glUniformMatrix4fv(glGetUniformLocation(rayShader.ID, "camMatrix"), 1, GL_FALSE, glm::value_ptr(camMat));

	// Pass the inverseView matrix to the shader
	glm::mat4 inverseView = glm::inverse(view);
	glUniformMatrix4fv(glGetUniformLocation(rayShader.ID, "inverseView"), 1, GL_FALSE, glm::value_ptr(inverseView));

	// Draw the frustum
	glBindVertexArray(VAO);
	glLineWidth(1.0f);
	glDrawArrays(GL_LINES, 0, 24);
}

void Camera::Inputs(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		Position += speed * Orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		Position += speed * -glm::normalize(glm::cross(Orientation, Up));
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		Position += speed * -Orientation;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		Position += speed * glm::normalize(glm::cross(Orientation, Up));
	}
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		Position += speed * Up;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
	{
		Position += speed * -Up;
	}
	if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
	{
		Position = initialPosition;
		Orientation = initialOrientation;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
	{
		speed = 1.0f;
	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE)
	{
		speed = 0.5f;
	}
	double lastX = width / 2.0, lastY = height / 2.0;

	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	{
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

		if (firstClick)
		{
			// Set the cursor position to the center of the screen only once
			glfwSetCursorPos(window, width / 2, height / 2);
			firstClick = false;
		}
		
		// Get the current mouse position
		glfwGetCursorPos(window, &mouseX, &mouseY);

		// Calculate the difference in mouse movement since the last frame
		float rotX = sensitivity * (float)(mouseY - lastY) / height;
		float rotY = sensitivity * (float)(mouseX - lastX) / width;  // Consider the width for X rotation

		// Update the rotation based on the movement
		glm::vec3 newOrientation = glm::rotate(Orientation, glm::radians(-rotX), glm::normalize(glm::cross(Orientation, Up)));
		if (!(glm::angle(newOrientation, Up) <= glm::radians(5.0f) || glm::angle(newOrientation, -Up) <= glm::radians(5.0f)))
		{
			Orientation = newOrientation;
		}

		// Apply Y-axis rotation
		Orientation = glm::rotate(Orientation, glm::radians(-rotY), Up);

		// Save the current mouse position for the next frame
		lastX = mouseX;
		lastY = mouseY;

		// Reset cursor position to center for the next frame
		glfwSetCursorPos(window, (width / 2), (height / 2));
	}
	else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
	{
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		firstClick = true;
	}
}
