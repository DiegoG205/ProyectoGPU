#version 330 core
out vec4 fragColor;

in vec3 vel;

uniform vec3 color;

void main() {

  float MAX_SPEED = 25.f;
  vec4 MAX_COLOR = vec4(1.0, 0.3, 0.1, 1.0);
  vec4 MIN_COLOR = vec4(0.1, 0.3, 1.0, 1.0);

  float mag = length(vel);
  float val = mag/MAX_SPEED;
  
  if (val > 1) {
    fragColor = MAX_COLOR;
  }
  else {
    fragColor = mix(MIN_COLOR, MAX_COLOR, val);
  }
}
