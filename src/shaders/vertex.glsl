#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aVel;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vel;

void main() {
  gl_PointSize = 5.0;
  gl_Position = projection * view * model * vec4(aPos, 1.0);
  vel = aVel;
}
