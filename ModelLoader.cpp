#include "ModelLoader.h"


bool Model::load(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    std::vector<glm::vec3> positions, normals;
    std::map<std::tuple<unsigned int, unsigned int, unsigned int>, unsigned int> indexMap;
	unsigned int pos = 0;

    
    while (std::getline(file, line)) {
        std::istringstream sstream(line);
        std::string firstWord;
        sstream >> firstWord;

        if (firstWord == "v") {
            float x, y, z;
            sstream >> x >> y >> z;
            positions.push_back(glm::vec3(x,y,z));
        }
        else if (firstWord == "vn") {
            float x, y, z;
            sstream >> x >> y >> z;
            normals.push_back(glm::vec3(x,y,z));
        }
        else if (firstWord == "f") {
            Vertex vertex;
            unsigned int vertexIndex[3], texIndex[3], normIndex[3];
            char slash;
            for (int i = 0; i < 3; ++i) {
                sstream >> vertexIndex[i] >> slash >> texIndex[i] >> slash >> normIndex[i];
                auto key = std::make_tuple(vertexIndex[i], texIndex[i], normIndex[i]);
                if (indexMap.find(key) == indexMap.end())
                {
                    indices.push_back(pos);
                    indexMap[key] = pos;
                    pos += 1;
                    vertex.position = positions[vertexIndex[i] - 1];
                    vertex.normal = normals[normIndex[i] - 1];
                    vertices.push_back(vertex);
                }
                else
                {
                    indices.push_back(indexMap[key]);
                }
            }
        }
    };

    return true;
};