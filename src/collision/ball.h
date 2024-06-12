#ifndef _EXAMPLE_BALL_H_
#define _EXAMPLE_BALL_H_

#include "glm/fwd.hpp"
#include <glm/glm.hpp>

class Ball {
public:
  glm::vec3 position;
  glm::vec3 velocity;
  glm::vec3 color;
  float radius;
  float mass;

  Ball();
  ~Ball();
  glm::mat4 model();
  void draw();
  void update(float deltaTime);
  bool check(const glm::vec2& point);
  bool check(const Ball& ball);
};

#endif // _EXAMPLE_BALL_H_
