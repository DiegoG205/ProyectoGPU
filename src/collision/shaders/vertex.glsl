#version 330 core

layout (location = 0) in vec3 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform bool outline;

void main() {
  if (outline) {
    vec3 pos = aPos * 1.1f;
    gl_Position = projection * view * model * vec4(pos, 1.0);
  } else {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
  } 
}
