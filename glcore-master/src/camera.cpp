#include "glm/common.hpp"
#include "glm/ext/matrix_transform.hpp"
#include "glm/geometric.hpp"
#include <glcore/camera.h>

using glm::mat4;
using glm::vec3;

Camera::Camera(glm::vec3 position, float distance, float yaw, float pitch,
               glm::vec3 up)
    : position(position), distance(distance), yaw(yaw), pitch(pitch),
      worldUp_(up), direction_(0.f) {
  updateVectors();
}

void Camera::update(float deltaTime) {
  vec3 dir = direction_;
  if (dir != vec3(0))
    dir = glm::normalize(dir);
  position += speed * deltaTime * dir;
  updateVectors();
}

mat4 Camera::view() {
  return glm::lookAt(position, position + (distance * front_), up_);
}

void Camera::key_callback(float xoffset, float yoffset) {
  direction_ = xoffset * right_ + yoffset * front_;
}

void Camera::cursor_position_callback(float xoffset, float yoffset) {

  xoffset *= sensitibity;
  yoffset *= sensitibity;

  yaw = xoffset;
  pitch = yoffset;
  pitch = glm::max(-89.f, glm::min(89.f, pitch));
}
void Camera::scroll_callback(float yoffset) {
  zoom -= yoffset;
  zoom = glm::max(1.f, glm::min(45.f, zoom));
}

void Camera::updateVectors() {
  // calculate the new Front vector
  glm::vec3 front;
  front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
  front.y = sin(glm::radians(pitch));
  front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
  front_ = glm::normalize(front);
  right_ = glm::normalize(glm::cross(
      front_, worldUp_)); // normalize the vectors, because their length gets
                          // closer to 0 the more you look up or down which
                          // results in slower movement.
  up_ = glm::cross(right_, front_);
}
