#version 330 core
out vec4 fragColor;

in vec3 vel;

uniform vec3 color;

void main() {
  fragColor = vec4(vel, 1.0);
}
