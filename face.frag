#version 460 core

in vec3 fragNormal;
in vec3 crntPos;
out vec4 FragColor;

uniform vec4 lightColor;
uniform vec3 lightPos;
//uniform vec3 camPos;


void main()
{
    vec3 normal = normalize(fragNormal);
    vec3 lightDirection = normalize(lightPos - crntPos);
    float dist = distance(lightPos, crntPos);
    //float strength = 0.3f  / (log(dist) / log(1000.0f));
    float strength = 5.0f * min(1.0f  / log(dist), 0.9f);

    // Calculate diffuse lighting
    float diffuse = max(dot(normal, lightDirection), 0.0f);

    // Set the fragment color
    FragColor = lightColor;
}
