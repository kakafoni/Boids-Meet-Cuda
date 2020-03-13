#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec3 ourColor;
uniform vec3 lightPos;
uniform vec3 lightPos2;
uniform vec3 bgColor;

void main()
{
	// ambient
	// vec3 ambient = 0.2 * vec3(1.0, 1.0, 1.0);
	vec3 ambient = bgColor + vec3(1.0, 1.0, 1.0)*0.3;


	// diffuse
	vec3 norm = normalize(Normal);
	vec3 lightDir = normalize(lightPos2 - FragPos);
	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

	// diffuse 2
	vec3 norm2 = normalize(Normal);
	vec3 lightDir2 = normalize(lightPos - FragPos);
	float diff2 = max(dot(norm, lightDir), 0.0);
	vec3 diffuse2 = diff * vec3(1.0, 1.0, 1.0);

	vec3 result = (0.5*diffuse + 0.5*diffuse2 + ambient) * ourColor;
    FragColor = vec4(result, 1.0f);
}