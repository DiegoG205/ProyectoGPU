#include "glcore/camera.h"
#include "glcore/shader.h"
#include "glm/ext/matrix_clip_space.hpp"
#include <cstdlib>
#include <glcore/app.h>
#include <glcore/buffer.h>
#include <glm/gtc/type_ptr.hpp>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#include "fluid_sim.cu"

#define N 256

using std::make_unique;
using std::unique_ptr;
using std::vector;

class CudaBuffer : public VertexBuffer {
  public:
    struct cudaGraphicsResource *VBO;
    int size;

    CudaBuffer(int n) {

      glGenVertexArrays(1, &vao_);
      glGenBuffers(1, &vbo_);
      
      size = n;
    };

    void build() override {
      bind();
      glBufferData(GL_ARRAY_BUFFER, 6* sizeof(float) * size,
                  nullptr, GL_DYNAMIC_DRAW);
      glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
      glEnableVertexAttribArray(1);

      cudaGraphicsGLRegisterBuffer(&VBO, vbo_, cudaGraphicsMapFlagsWriteDiscard);
    };

    void draw() override {
      bind();
      glDrawArrays(GL_POINTS, 0, size);
    }
};


class FluidApp : public App {
public:
  unique_ptr<Shader> shader;
  unique_ptr<Camera> cam;
  unique_ptr<Buffer> quad;
  unique_ptr<CudaBuffer> particles;
  glm::mat4 projection;

  float3 *auxDev;
  float *densDev;

  struct {
    float xmouse, ymouse;
    bool showMenu = true;
    bool stop = true;

    float sRadius = 3.f;
    float dt = 0.01f;

    float targetDensity = 0.5f;
    float PressureMultiplier = 0.5f;

  } settings;

  FluidApp() : App(3, 3, 1000, 1000, "Fluid Simulation"), projection(1.0f) {

    // Crear datos partículas

    std::size_t size = 2 * sizeof(float3) * N;

    cudaMalloc(&auxDev, size);
    cudaMalloc(&densDev, sizeof(float)*N);

    createData(auxDev);

    particles = make_unique<CudaBuffer>(N);
    particles->build();

    vector<float> qv{20.0f,  20.0f,  0.0f, 1.0f, 1.0f, 1.0f,
                     20.0f,  -20.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                     -20.0f, -20.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                     -20.0f, 20.0f,  0.0f, 1.0f, 1.0f, 1.0f};
    vector<unsigned int> qi{0, 1, 2, 2, 3, 0};
    quad = make_unique<BasicBuffer>(qv, qi);
    quad->build();

    shader = make_unique<Shader>(SHADER_PATH "vertex.glsl", SHADER_PATH "fragment.glsl");
    cam = make_unique<Camera>(glm::vec3(0.f, 0.f, 3.f), 1.f, -90.f);

    glEnable( GL_PROGRAM_POINT_SIZE );
  }

  void render() override {
    App::render();
    glClearColor(0.7, 0.7, 0.7, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST); // solo 2D
    if (settings.showMenu) {
      show_main_menu();
    }

    shader->use();
    glm::mat4 view = cam->view();
    shader->set("view", cam->view());
    float aspect = (float)width_ / (float)height_;
    projection = glm::ortho(-20.f, 20.f, -20.0f, 20.0f, 0.1f, 100.0f);
    shader->set("projection", projection);

    shader->set("model", glm::mat4(1.0f));
    shader->set("color", glm::vec3(0.1f, 0.12f, 0.12f));
    //quad->draw();

    shader->set("color", glm::vec3(0.f));
    particles->draw();
  }

  void update(float deltaTime) override {
    //deltaTime *= settings.speed;
    if (settings.stop)
      return;

    runCuda(&(particles->VBO), auxDev);
  }

  void createData(float3 *auxDev) {

  float3 aux[2*N];

  int rowSize = (int)std::sqrt(N);
  int colSize = (N - 1) / rowSize + 1;
  float size = 18.f;

  for (int i = 0; i < colSize; i++) {
    for(int j = 0; j < rowSize; j++) {
      int index = i*rowSize + j;
      std::cout << index << "\n";
      if (index >= N) break;
      aux[2*index].x = j * size/rowSize - size/2;
      aux[2*index].y = i * size/colSize - size/2;
      aux[2*index].z = 0;
      aux[2*index + 1] = {0,0,0};
    }
  }
  //for(int i = 0; i < N; i++) std::cout << aux[2*i].x << " " << aux[2*i].y << " " << aux[2*i].z << "\n" << aux[2*i+1].x << " " << aux[2*i+1].y << " " << aux[2*i+1].z << "\n";
  cudaMemcpy(auxDev, aux, 2*N*sizeof(float3), cudaMemcpyHostToDevice);
};

  void runCuda(struct cudaGraphicsResource **cudaVBOResourcePointer, float3 *auxDev) {
    // Map OpenGL buffer object for writing from CUDA
    float3 *dptr;
    cudaGraphicsMapResources(1, cudaVBOResourcePointer, 0);
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**) &dptr, &numBytes, *cudaVBOResourcePointer);

    // Block size
    int blockSize = 64;

    // Round up in case N is not a multiple of blockSize
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Execute the kernel
    cudaMemcpy(dptr, auxDev, 2*N*sizeof(float3), cudaMemcpyDeviceToDevice);

    //std::cout << "Start kernel\n";
    updateDensities<<<numBlocks, blockSize>>>(N, dptr, densDev, settings.sRadius);
    fluid_kernel<<<numBlocks, blockSize>>>(N, dptr, auxDev, densDev, settings.dt, 
                                          settings.sRadius, settings.targetDensity, settings.PressureMultiplier);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, cudaVBOResourcePointer, 0);
  }

  void show_main_menu() {
    constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                       ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoSavedSettings;
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x/3, viewport->Size.y/3));
    if (ImGui::Begin("Opciones", NULL, flags)) {
      ImGui::Text("Opciones");
      ImGui::SliderFloat("Target Density", &settings.targetDensity, 0, 2.f);
      ImGui::SliderFloat("Pressure Multiplier", &settings.PressureMultiplier, 0, 100.f);
      ImGui::SliderFloat("Smoothing Radius", &settings.sRadius, 1.f, 4.f);
      ImGui::SliderFloat("Delta Time", &settings.dt, 0, 0.02f);
      ImGui::Checkbox("Stop", &settings.stop);
    }
    ImGui::End();
  }

  void key_callback(int key, int scancode, int action, int mods) override {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      settings.showMenu = !settings.showMenu;
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
      settings.stop = !settings.stop;
    }
    if (key == GLFW_KEY_O && action == GLFW_PRESS) {
      settings.sRadius += 0.5f;
      std::cout << "New radius: " << settings.sRadius << "\n";
    }
    if (key == GLFW_KEY_I && action == GLFW_PRESS) {
      settings.sRadius -= 0.5f;
      std::cout << "New radius: " << settings.sRadius << "\n";
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
      //settings.target = nullptr;
      float x = (2.0f * settings.xmouse) / width_ - 1.0f;
      float y = 1.0f - (2.0f * settings.ymouse) / height_;

      // from viewport to world
      glm::vec4 pos(x, y, 0.0f, 1.0f);
      glm::mat4 toWorld = glm::inverse(projection * cam->view());
      glm::vec4 realPos = toWorld * pos;
      float3 p = {realPos.x, realPos.y, 0};
      float3 *dptr;
      size_t numBytes;
      cudaGraphicsMapResources(1, &(particles->VBO), 0);
      cudaGraphicsResourceGetMappedPointer((void**) &dptr, &numBytes, *&(particles->VBO));
      
      float3 auxP[2*N];
      cudaMemcpy(auxP, dptr, 2*N*sizeof(float3), cudaMemcpyDeviceToHost);
      //for (int i = 0; i < N; i++) std::cout << auxP[i].x << " " << auxP[i].y << " " << auxP[i].z << "\n";
      float res = calculateDensityHost(N, p, auxP, settings.sRadius);
      std::cout << res << "\n";
      cudaGraphicsUnmapResources(1, &(particles->VBO), 0);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
      // from viewport to world
      float3 *dptr;
      size_t numBytes;
      cudaGraphicsMapResources(1, &(particles->VBO), 0);
      cudaGraphicsResourceGetMappedPointer((void**) &dptr, &numBytes, *&(particles->VBO));
      
      float3 auxP[2*N];
      cudaMemcpy(auxP, dptr, 2*N*sizeof(float3), cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; i++) std::cout << i << ": " << auxP[i].x << " " << auxP[i].y << " " << auxP[i].z << "\n";
      cudaGraphicsUnmapResources(1, &(particles->VBO), 0);
    }
  }
};

int main() {
  FluidApp app;
  app.run();
  return 0;
}
