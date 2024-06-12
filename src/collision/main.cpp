#include "ball.h"
#include "glcore/camera.h"
#include "glcore/shader.h"
#include "glm/ext/matrix_clip_space.hpp"
#include <cstdlib>
#include <glcore/app.h>
#include <glcore/buffer.h>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <vector>

#define N 10

using std::make_unique;
using std::unique_ptr;
using std::vector;

class MyApp : public App {
public:
  vector<Ball> balls;
  unique_ptr<Shader> shader;
  unique_ptr<Camera> cam;
  unique_ptr<Buffer> quad;
  glm::mat4 projection;
  struct {
    float xmouse, ymouse;
    bool showMenu = false;
    float speed = 1.0f;
    Ball *target = nullptr;
  } settings;

  MyApp() : App(3, 3, 800, 600), projection(1.0f) {
    for (size_t i = 0; i < N; i++) {
      balls.emplace_back();
      balls[i].radius = glm::mix(0.05, 0.1, (double)rand()/RAND_MAX);
      balls[i].velocity = glm::vec3((double)rand()/RAND_MAX * 2 - 1, (double)rand()/RAND_MAX * 2 - 1, 0.f);
      balls[i].position = glm::vec3((double)rand()/RAND_MAX * 2 - 1, (double)rand()/RAND_MAX * 2 - 1, 0.f);
      balls[i].color = glm::vec3((double)rand()/RAND_MAX, (double)rand()/RAND_MAX, (double)rand()/RAND_MAX);
    }

    vector<float> qv{1.0f,  1.0f,  0.0f, 1.0f,  -1.0f, 0.0f,
                     -1.0f, -1.0f, 0.0f, -1.0f, 1.0f,  0.0f};
    vector<unsigned int> qi{0, 1, 2, 2, 3, 0};
    quad = make_unique<BasicBuffer>(qv, qi);
    quad->build();
    shader = make_unique<Shader>(SHADER_PATH "vertex.glsl",
                                 SHADER_PATH "fragment.glsl");
    cam = make_unique<Camera>(glm::vec3(0.f, 0.f, 3.f), 1.f, -90.f);
  }

  void render() override {
    App::render();
    glDisable(GL_DEPTH_TEST); // solo 2D
    if (settings.showMenu) {
      show_main_menu();
    }

    if (settings.target) {
      ImGui::Begin("Información"); // Create a window called "Hello,
                                   // world!" and append into it.
      ImGui::Text("Pelota en (%.2f, %.2f)", settings.target->position.x,
                  settings.target->position.y);
      ImGui::SliderFloat("Radio", &settings.target->radius, 0.01f, 0.5f);
      ImGui::SliderFloat("Mass", &settings.target->mass, 0.1f, 1.0f);
      ImGui::ColorPicker3(
          "Color",
          glm::value_ptr(
              settings.target->color)); // Edit 3 floats representing a color

      ImGui::End();
    }

    shader->use();
    glm::mat4 view = cam->view();
    shader->set("view", cam->view());
    float aspect = (float)width_ / (float)height_;
    projection = glm::ortho(-aspect, aspect, -1.0f, 1.0f, 0.1f, 100.0f);
    // glm::perspective(glm::radians(cam->zoom),
    //                  ((float)width_) / ((float)height_), 0.1f, 100.0f);
    shader->set("projection", projection);
    shader->set("outline", 0);

    shader->set("model", glm::mat4(1.0f));
    shader->set("color", glm::vec3(0.1f, 0.12f, 0.12f));
    quad->draw();

    if (settings.target) {
      shader->set("model", settings.target->model());
      shader->set("color", {1.f, 1.f, 1.f});
      shader->set("outline", 1);
      settings.target->draw();
      shader->set("outline", 0);
    }

    for (auto &b : balls) {
      shader->set("model", b.model());
      shader->set("color", b.color);
      b.draw();
    }
  }

  void update(float deltaTime) override {
    deltaTime *= settings.speed;
    if (settings.showMenu)
      return;

    // collisions
    for (int i = 0; i < balls.size(); i++) {
      for (int j = i + 1; j < balls.size(); j++) {
        auto &b1 = balls[i];
        auto &b2 = balls[j];
        if (b1.check(b2)) {
          auto collision = b1.position - b2.position;
          float distance = glm::length(collision);
          glm::vec3 plane = collision / distance;
          float b1i = glm::dot(b1.velocity, plane);
          float b2i = glm::dot(b2.velocity, plane);

          float b1f = b2i;
          float b2f = b1i;

          b1.velocity += (b1f - b1i) * plane;
          b2.velocity += (b2f - b2i) * plane;

          // position correction
          float delta = b1.radius + b2.radius - distance;
          b1.position += 0.5f * delta * plane;
          b2.position -= 0.5f * delta * plane;
        }
      }
    }

    // wall collisions
    for (auto &b : balls) {
      if (b.position.x - b.radius < -1.0) {
        b.velocity.x *= -1;
        float delta = b.position.x - b.radius + 1.0;
        b.position.x -= delta;
      }

      if (b.position.x + b.radius > 1.0) {
        b.velocity.x *= -1;
        float delta = b.position.x + b.radius - 1.0;
        b.position.x -= delta;
      }

      if (b.position.y - b.radius < -1.0) {
        b.velocity.y *= -1;
        float delta = b.position.y - b.radius + 1.0;
        b.position.y -= delta;
      }
      if (b.position.y + b.radius > 1.0) {
        b.velocity.y *= -1;
        float delta = b.position.y + b.radius - 1.0;
        b.position.y -= delta;
      }
    }

    // movement
    for (auto &b : balls) {
      b.position += b.velocity * deltaTime;
    }
  }

  void show_main_menu() {
    constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                       ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoSavedSettings;
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);
    if (ImGui::Begin("Pausa", NULL, flags)) {
      ImGui::Text("Menu de Pausa");
      ImGui::SliderFloat("Simulation Speed", &settings.speed, 0.1f, 2.0f);
      if (ImGui::Button("Salir"))
        glfwSetWindowShouldClose(window, 1);
    }
    ImGui::End();
  }

  void key_callback(int key, int scancode, int action, int mods) override {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      settings.showMenu = !settings.showMenu;
    }
  }

  void cursor_position_callback(double xpos, double ypos) override {
    if (io->WantCaptureMouse)
      return;
    settings.xmouse = xpos;
    settings.ymouse = ypos;
  }

  void mouse_button_callback(int button, int action, int mods) override {
    if (io->WantCaptureMouse)
      return;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
      settings.target = nullptr;
      float x = (2.0f * settings.xmouse) / width_ - 1.0f;
      float y = 1.0f - (2.0f * settings.ymouse) / height_;

      // from viewport to world
      glm::vec4 pos(x, y, 0.0f, 1.0f);
      glm::mat4 toWorld = glm::inverse(projection * cam->view());
      glm::vec4 realPos = toWorld * pos;
      for (auto &b : balls) {
        if (b.check(realPos)) {
          settings.target = &b;
        }
      }
    }
  }
};

int main() {
  MyApp app;
  app.run();
  return 0;
}
