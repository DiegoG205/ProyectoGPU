#include "imgui.h"
#include <format>
#include <glcore/app.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>

using std::string;

static App *current = nullptr;

static void global_framebuffer_size_callback(GLFWwindow *window, int w, int h) {
  if (window == current->window) {
    current->framebuffer_size_callback(w, h);
  }
}
static void global_key_callback(GLFWwindow *window, int key, int scancode,
                                int action, int mods) {
  if (window == current->window) {
    current->key_callback(key, scancode, action, mods);
  }
}

static void global_character_callback(GLFWwindow *window,
                                      unsigned int codepoint) {
  if (window == current->window) {
    current->character_callback(codepoint);
  }
}
static void global_cursor_position_callback(GLFWwindow *window, double xpos,
                                            double ypos) {
  if (window == current->window) {
    current->cursor_position_callback(xpos, ypos);
  }
}

static void global_mouse_button_callback(GLFWwindow *window, int button,
                                         int action, int mods) {
  if (window == current->window) {
    current->mouse_button_callback(button, action, mods);
  }
}
static void global_cursor_enter_callback(GLFWwindow *window, int entered) {
  if (window == current->window) {
    current->cursor_enter_callback(entered);
  }
}

static void global_scroll_callback(GLFWwindow *window, double xoffset,
                                   double yoffset) {
  if (window == current->window) {
    current->scroll_callback(xoffset, yoffset);
  }
}

static void global_error_callback(int error, const char* desc) {
  std::cerr << "GFLW Error " << error << ": " << desc << "\n";
}

App::App(unsigned int glmajor, unsigned int glminor, int width, int height,
         const string &title) {
  glfwSetErrorCallback(global_error_callback);
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, glmajor);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, glmajor);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
  if (width <= 0 || height <= 0)
    glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
  window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
  if (window == NULL) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    exit(-1);
  }
  glfwGetWindowSize(window, &width_, &height_);

  glfwMakeContextCurrent(window);
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    glfwTerminate();
    exit(-1);
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  io = &ImGui::GetIO();
  (void)io;
  io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsDark();
  // ImGui::StyleColorsLight();
  
  glfwSetFramebufferSizeCallback(window, global_framebuffer_size_callback);
  glfwSetKeyCallback(window, global_key_callback);
  glfwSetMouseButtonCallback(window, global_mouse_button_callback);
  glfwSetCursorPosCallback(window, global_cursor_position_callback);
  glfwSetScrollCallback(window, global_scroll_callback);

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(std::format("#version {}{}0", glmajor, glminor).c_str());
  glEnable(GL_DEPTH_TEST);
}

App::~App() { 
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate(); 
}

bool App::isFinished() { return glfwWindowShouldClose(window); }

// Dibujar la simulacion
void App::render() {
  glViewport(0, 0, width_, height_);
  glClearColor(.0f, .0f, .0f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

// Hacer calculos de la simulacion
void App::update(float deltaTime) {} 

void App::framebuffer_size_callback(int width, int height) {
  height_ = height;
  width_ = width;
}
void App::key_callback(int key, int scancode, int action, int mods) {}
void App::character_callback(unsigned int codepoint) {}
void App::cursor_position_callback(double xpos, double ypos) {}
void App::mouse_button_callback(int button, int action, int mods) {}
void App::cursor_enter_callback(int entered) {}
void App::scroll_callback(double xoffset, double yoffset) {}

void App::run() {
  current = this;
  float lastFrame = 0.f;
  while (!isFinished()) {
    float currentFrame = glfwGetTime();
    update(currentFrame - lastFrame);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    render();
    ImGui::Render();
    lastFrame = currentFrame;
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  current = nullptr;
}
