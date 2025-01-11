#version 460 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 instance;

uniform float scale;
uniform mat4 camMatrix;

out vec3 fragNormal;
out vec3 crntPos;

void main()
{
    vec3 inst = vec3(20.0 * instance);
    mat4 instanceModel = mat4(
        scale, 0.0,   0.0,   0.0,
        0.0,   scale, 0.0,   0.0,
        0.0,   0.0,   scale, 0.0,
        -inst.x, -inst.y, inst.z, 1.0
    );

    vec4 worldPos = instanceModel * vec4(aPos, 1.0f);
    crntPos = worldPos.xyz;

    fragNormal = normalize(mat3(instanceModel) * aNormal);

    gl_Position = camMatrix * worldPos;
}
