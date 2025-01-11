#include "shaderClass.h"

std::string get_file_contents(const char* filename)
{
	std::ifstream in(filename, std::ios::binary);
	if (in)
	{
		std::string contents;
		in.seekg(0, std::ios::end);
		contents.resize(in.tellg());
		in.seekg(0, std::ios::beg);
		in.read(&contents[0], contents.size());
		in.close();
		return(contents);
	}
	throw(errno);
}

Shader::Shader(const char* vertexFile, const char* fragmentFile)
{
	std::string vertexCode = get_file_contents(vertexFile);
	std::string fragmentCode = get_file_contents(fragmentFile);

	const char* vertexSource = vertexCode.c_str();
	const char* fragmentSource = fragmentCode.c_str();

	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);

	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);

	GLint frag_success;
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &frag_success);
	if (!frag_success) {
		// Get the error message if the shader failed to compile
		GLint logLength;
		glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &logLength);
		char* log = new char[logLength];
		glGetShaderInfoLog(fragmentShader, logLength, nullptr, log);
		std::cerr << "Shader compilation failed: " << log << std::endl;
		delete[] log;
	}
	GLint vert_success;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vert_success);
	if (!vert_success) {
		// Get the error message if the shader failed to compile
		GLint logLength;
		glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &logLength);
		char* log = new char[logLength];
		glGetShaderInfoLog(vertexShader, logLength, nullptr, log);
		std::cerr << "Shader compilation failed: " << log << std::endl;
		delete[] log;
	}

	ID = glCreateProgram();

	glAttachShader(ID, vertexShader);
	glAttachShader(ID, fragmentShader);

	glLinkProgram(ID);

	GLint success;
	glGetProgramiv(ID, GL_LINK_STATUS, &success);
	if (!success) {
		// Get the error message if linking failed
		GLint logLength;
		glGetProgramiv(ID, GL_INFO_LOG_LENGTH, &logLength);
		char* log = new char[logLength];
		glGetProgramInfoLog(ID, logLength, nullptr, log);
		std::cerr << "Program linking failed: " << log << std::endl;
		delete[] log;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

void Shader::Activate()
{
	glUseProgram(ID);
}

void Shader::Delete()
{
	glUseProgram(0);
}