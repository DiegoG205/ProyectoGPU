#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <glm/glm.hpp>

class Camera {
public:

  glm::vec3 position;
  float distance;
  float yaw;
  float pitch;
  float sensitibity = .1f;
  float zoom = 45.f;
  float speed = 1.f;

  Camera(glm::vec3 position = glm::vec3(0.f), float distance = 1.f, float yaw = 0.0f, float pitch = 0.0f, glm::vec3 up = glm::vec3(0.f, 1.f, 0.f));
  Camera(Camera &&) = default;
  Camera(const Camera &) = default;
  Camera &operator=(Camera &&) = default;
  Camera &operator=(const Camera &) = default;
  ~Camera() = default;

  void update(float deltaTime = 0);
  glm::mat4 view();

  void key_callback(float xoffset, float yoffset);
  void cursor_position_callback(float xoffset, float yoffset);
  void scroll_callback(float yoffset);
  
private:
  glm::vec3 up_;
  glm::vec3 front_;
  glm::vec3 right_;
  glm::vec3 worldUp_;
  glm::vec3 direction_;

  void updateVectors();
};

#endif // _CAMERA_H_
