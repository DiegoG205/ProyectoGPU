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

#define N 8192

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
    uint3 *hashDev;
    uint *spatialIndexDev;

    int n_part;

    struct {
      float xmouse, ymouse;
      float realXpos, realYpos;
      int mouseAction = 0;
      bool showMenu = true;
      bool stop = true;

      float sRadius = 3.f;
      float dt = 0.01f;

      float targetDensity = 0.5f;
      float PressureMultiplier = 0.5f;
      float ViscosityStr = 0.5f;
      float gravity = 10.f;

    } settings;

    double prev_time = 0.0;
    double current_time = 0.0;
    double time_diff = 0.0;
    unsigned int counter = 0;

  FluidApp(int n) : App(3, 3, 1000, 1000, "Fluid Simulation"), projection(1.0f), n_part(n) {

    // Crear datos partículas

    std::size_t size = 2 * sizeof(float3) * n_part;

    cudaMalloc(&auxDev, size);
    cudaMalloc(&densDev, sizeof(float)*n_part);
    cudaMalloc(&hashDev, sizeof(uint3)*n_part);
    cudaMalloc(&spatialIndexDev, sizeof(uint)*n_part);

    createData(auxDev);

    particles = make_unique<CudaBuffer>(n_part);
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
    projection = glm::ortho(-50.f, 50.f, -50.0f, 50.0f, 0.1f, 100.0f);
    shader->set("projection", projection);

    shader->set("model", glm::mat4(1.0f));
    shader->set("color", glm::vec3(0.1f, 0.12f, 0.12f));
    //quad->draw();

    shader->set("color", glm::vec3(0.f));
    particles->draw();
  }

  void update(float deltaTime) override {
    // FPS counter
    current_time = glfwGetTime();
    time_diff = current_time - prev_time;
    counter++;
    if (time_diff >= 0.5) {
      std::string FPS = std::to_string((1.0/time_diff) * counter);
      std::string newTitle = "Fluid Simulation - " + FPS + " FPS";
      glfwSetWindowTitle(window, newTitle.c_str());
      prev_time = current_time;
      counter = 0;
    }
    if (settings.stop)
      return;
    if(settings.mouseAction) {
      update_real_mouse_pos();
    }
    runCuda(&(particles->VBO), auxDev);
  }

  void createData(float3 *auxDev) {

    // float3 aux[2*n_part];

    vector<float3> aux(2*n_part);

    int rowSize = (int)std::sqrt(n_part);
    int colSize = (n_part - 1) / rowSize + 1;
    float size = 18.f;

    for (int i = 0; i < colSize; i++) {
      for(int j = 0; j < rowSize; j++) {
        int index = i*rowSize + j;
        if (index >= n_part) break;
        aux[2*index].x = j * size/rowSize - size/2;
        aux[2*index].y = i * size/colSize - size/2;
        aux[2*index].z = 0;
        aux[2*index + 1] = {0,0,0};
      }
    }
    cudaMemcpy(auxDev, aux.data(), 2*n_part*sizeof(float3), cudaMemcpyHostToDevice);
  };

  void runCuda(struct cudaGraphicsResource **cudaVBOResourcePointer, float3 *auxDev) {
    // Map OpenGL buffer object for writing from CUDA
    float3 *dptr;
    cudaGraphicsMapResources(1, cudaVBOResourcePointer, 0);
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**) &dptr, &numBytes, *cudaVBOResourcePointer);

    // Block size
    int blockSize = 64;

    // Round up in case n_part is not a multiple of blockSize
    int numBlocks = (n_part + blockSize - 1) / blockSize;

    // Execute the kernel
    cudaMemcpy(dptr, auxDev, 2*n_part*sizeof(float3), cudaMemcpyDeviceToDevice);

    // Crear el hash
    calcHash<<<numBlocks, blockSize>>>(n_part, dptr, hashDev, spatialIndexDev, settings.dt);

    // Hacer el sort 
    int k,j;
    for (k = 2; k <= n_part; k <<=1) {
      for (j = k>>1; j>0; j=j>>1) {
        bitonicSortStep<<<numBlocks, blockSize>>>(hashDev, j, k);
      }
    }
    
    // Rellenar los spatialIndex
    // En el arreglo spatialIndexDev, quedara en la posicion k la posicion en hashDev donde empieza la celda con key k
    findCellStart<<<numBlocks, blockSize>>>(n_part, hashDev, spatialIndexDev);

    updateDensitiesHash<<<numBlocks, blockSize>>>(n_part, dptr, densDev, hashDev, spatialIndexDev, settings.sRadius, settings.dt);

    fluid_kernel<<<numBlocks, blockSize>>>(n_part, dptr, auxDev, densDev, hashDev, spatialIndexDev, settings.dt, 
                                          settings.sRadius, settings.targetDensity, settings.PressureMultiplier, 
                                          settings.gravity, settings.ViscosityStr, 
                                          settings.realXpos, settings.realYpos, settings.mouseAction);

    // Unmap buffer object
    cudaGraphicsUnmapResources(1, cudaVBOResourcePointer, 0);
  }

  void show_main_menu() {
    constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                                       ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoSavedSettings;
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(ImVec2(3 * viewport->Size.x /7, viewport->Size.y/4));
    if (ImGui::Begin("Opciones", NULL, flags)) {
      ImGui::Text("Opciones");
      ImGui::SliderFloat("Target Density", &settings.targetDensity, 0, 5.f);
      ImGui::SliderFloat("Pressure Multiplier", &settings.PressureMultiplier, 0, 2.f);
      ImGui::SliderFloat("Viscosity", &settings.ViscosityStr, 0, 1.f);
      ImGui::SliderFloat("Gravity", &settings.gravity, 0, 20.f);
      ImGui::SliderFloat("Delta Time", &settings.dt, 0, 0.02f);
      ImGui::Checkbox("Stop", &settings.stop);
    }
    ImGui::End();
  }

  void update_real_mouse_pos() {
    float x = (2.0f * settings.xmouse) / width_ - 1.0f;
    float y = 1.0f - (2.0f * settings.ymouse) / height_;

    // from viewport to world
    glm::vec4 pos(x, y, 0.0f, 1.0f);
    glm::mat4 toWorld = glm::inverse(projection * cam->view());
    glm::vec4 realPos = toWorld * pos;
    settings.realXpos = realPos.x;
    settings.realYpos = realPos.y;
  };

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
    
      settings.mouseAction = 1;

      update_real_mouse_pos();
    }
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
      settings.mouseAction = 0;
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
      settings.mouseAction = -1;
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
      settings.mouseAction = 0;
    }
  }

  ~FluidApp() {

    cudaFree(hashDev);
    cudaFree(spatialIndexDev);
    cudaFree(auxDev);
    cudaFree(densDev);

  }
};

int main(int argc, char* argv[]) {
  if (argc == 1) {
    FluidApp app(N);
    app.run();
  }
  else if (argc == 2) {
    int n = std::atoi(argv[1]);

    if (n < 64) {
      std::cerr << "Particle count must be bigger or equal than 64\n";
      return 2;
    }

    if (!((n & (n - 1)) == 0)) {
      std::cerr << "Particle count must be a power of 2\n";
      return 3;
    }

    FluidApp app(n);
    app.run();
  }
  else {
    std::cerr << "Use: fluidsim.exe <particle_count (optional)>\n";
    return 1;
  }
  return 0;
}
