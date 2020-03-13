#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;

out vec3 ourColor;
out vec3 Normal;
out vec3 FragPos;

uniform vec3 bgColor;
uniform vec3 cameraPos;

uniform vec3 color;
uniform mat4 projection;
uniform mat4 view;

void main()
{
    Normal = aNormal;
    gl_Position = projection * vec4(aPos, 1.0);

    float a = length(aPos);
    float ratio = sqrt(a / 8000.0);
    //float ratio = 0;
    ourColor = max(aColor *(1 - ratio) + bgColor*ratio, bgColor);
    FragPos = mat3(transpose(inverse(view))) * aPos;
}