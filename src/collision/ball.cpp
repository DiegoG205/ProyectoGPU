#include "ball.h"
#include "glcore/buffer.h"
#include "glm/ext/matrix_transform.hpp"
#include <vector>
#include <numbers>
#include <memory>

#define DEFINITION 20

 static std::unique_ptr<Buffer> buff = nullptr;


std::vector<float> generate_circle_vertex(unsigned int n) {
  std::vector<float> points;
  float dTheta = 2 * std::numbers::pi / n;
  for (size_t i = 0; i < n; i++) {
    float theta = i * dTheta;
    points.push_back(glm::cos(theta)); // x
    points.push_back(glm::sin(theta)); // y
    points.push_back(0.f);             // z
  }
  points.push_back(0.f); // x
  points.push_back(0.f); // y
  points.push_back(0.f); // z
  return points;
}

std::vector<unsigned int> generate_circle_index(unsigned int n) {
  std::vector<unsigned int> indices;
  for (size_t i = 0; i < n - 2; i++) {
    indices.push_back(n - 1);
    indices.push_back(i);
    indices.push_back(i + 1);
  }
  indices.push_back(n - 1);
  indices.push_back(n - 2);
  indices.push_back(0);

  return indices;
}


Ball::Ball(): position(0.f), velocity(0.f), radius(1.f), mass(100.f) {
  if (buff == nullptr) {
    buff = std::make_unique<BasicBuffer>(generate_circle_vertex(DEFINITION), generate_circle_index(DEFINITION));
    buff->build();
  }
}

Ball::~Ball() {
}

glm::mat4 Ball::model() {
  glm::mat4 m(1.f);
  return glm::scale(glm::translate(m, position), glm::vec3(radius, radius, 0.f));
}

void Ball::draw() {
  buff->draw();
}

void Ball::update(float deltaTime) {
  position += velocity * deltaTime;
}

bool Ball::check(const glm::vec2& point) {
  auto dist = glm::vec3(point, 0.0f) - position;
  return glm::length(dist) <= radius;
}

bool Ball::check(const Ball& ball) {
  auto dist = ball.position - position;
  return glm::length(dist) <= radius + ball.radius;
}
