#include <iostream>
#include <string>
#include <fstream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <random>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>


// Mesh width and height
const unsigned int meshWidth = 512;
 
// Smoothing radius
float sRadius = 0.1;
#define PI 3.14159265358979323846

bool stop = false;
bool test = false;

// Cuda graphics resource pointer
struct cudaGraphicsResource *cudaVBOResource;

// Animation time
float animTime = 0.0;

float dt = 0.002;

// ========================================================
// ======== Cuda Kernel to modify vertex positions ========
// ========================================================

__device__ float calculateDensity(float3 pos, float3* positions, float radius) {
  
  float density = 0;
  const float mass = 1;

  for (int i = 0; i < meshWidth; i++) {

    float3 other = positions[i];

    float dx = pos.x - other.x;
    float dy = pos.y - other.y;
    float dz = pos.z - other.z;
    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    float value = max(0.0f, radius - dist);
    float volume = PI * std::pow(radius, 4) / 2;

    float influence = value * value * value / volume;

    density += influence * mass;
  }

  return density;

};

__global__ void fluid_kernel(int n, float3 *posData, float3* posAux, float3 *velData, float3 *velAux, float dt, float radius) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;


  float3 pos = posData[index];
  float3 vel = velData[index];

  // Gravity
  vel.y -= 9.8 * dt;

  // Border Collisions
  if (std::abs(pos.y + 0.01) >= 0.95) vel.y = -vel.y*0.7;
  if (std::abs(pos.x + 0.01) >= 0.95) vel.x = -vel.x*0.7;

  float density = calculateDensity(pos, posData, radius);
  
  pos.y += vel.y * dt;

  posAux[index] = pos;
  velAux[index] = vel;

};

float calculateDensityHost(float3 pos, float3* positions, float radius) {
  
  float density = 0;
  const float mass = 1;
  float volume = PI * std::pow(radius, 4) / 2;

  std::cout << volume << '\n';

  for (int i = 0; i < meshWidth; i++) {

    float3 other = positions[i];

    float dx = pos.x - other.x;
    float dy = pos.y - other.y;
    float dz = pos.z - other.z;
    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    float value = max(0.0f, radius - dist);

    float influence = value * value * value / volume;

    density += influence * mass;

    //std::cout << i << ": " << density << '\n';
  }


  return density;
};

void createData(float3 *auxPos, float3 *auxVel) {

  float3 auxP[meshWidth];
  float3 auxV[meshWidth];

  int rowSize = (int)std::sqrt(meshWidth);
  int colSize = (meshWidth - 1) / rowSize + 1;
  float size = 1.6;

  for (int i = 0; i < colSize; i++) {
    for(int j = 0; j < rowSize; j++) {
      if (i*rowSize + j >= meshWidth) break;
      auxP[i*rowSize + j].x = j * size/rowSize - size/2;
      auxP[i*rowSize + j].y = i * size/colSize - size/2;
      auxP[i*rowSize + j].z = 0;
      auxV[i*rowSize + j] = {0,0,0};
    }
  }

  cudaMemcpy(auxPos, auxP, meshWidth*sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(auxVel, auxV, meshWidth*sizeof(float3), cudaMemcpyHostToDevice);
};

void createQuad(float x, float y) {//, float r, float g, float b) {

    float vertexData[24] = {-x, y, 0.0,// r, g, b,
                            x, y, 0.0,// r, g, b,
                            x, -y, 0.0,// r, g, b,
                            x, -y, 0.0,// r, g, b,
                            -x, -y, 0.0,//, r, g, b};
                            -x, y, 0.0};//, r, g, b};

    // int indexData[6] = {0, 1, 2,
    //                     2, 3, 0};

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 24*sizeof(float), vertexData, GL_STATIC_DRAW);

    glBindVertexArray(vao);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glDrawArrays(GL_TRIANGLES, 0, 6);

};

void runCuda(struct cudaGraphicsResource **cudaVBOResourcePointer, float3 *velDev, float3 *auxPos, float3 *auxVel)
{
  // Map OpenGL buffer object for writing from CUDA
  float3 *dptr;
  cudaGraphicsMapResources(1, cudaVBOResourcePointer, 0);
  size_t numBytes;
  cudaGraphicsResourceGetMappedPointer((void**) &dptr, &numBytes, *cudaVBOResourcePointer);

  // Block size
  int blockSize = 64;

  // Round up in case N is not a multiple of blockSize
  int numBlocks = (meshWidth + blockSize - 1) / blockSize;

  // Execute the kernel
  cudaMemcpy(dptr, auxPos, meshWidth*sizeof(float3), cudaMemcpyDeviceToDevice);
  cudaMemcpy(velDev, auxVel, meshWidth*sizeof(float3), cudaMemcpyDeviceToDevice);
  fluid_kernel<<<numBlocks, blockSize>>>(meshWidth, dptr, auxPos, velDev, auxVel, dt, sRadius);

  // Unmap buffer object
  cudaGraphicsUnmapResources(1, cudaVBOResourcePointer, 0);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    stop = !stop;
  if (key == GLFW_KEY_E && action == GLFW_PRESS)
    sRadius += 0.1;
  if (key == GLFW_KEY_Q && action == GLFW_PRESS && sRadius > 0.1)
    sRadius -= 0.1;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
      test = true;
    }
}
      

int main()
{

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  // Creating a GLFW window    
  GLFWwindow* glfwWindow = glfwCreateWindow(1000, 1000, "Test OpenGL", NULL, NULL);

  if (glfwWindow == NULL)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    exit(1);
  } else {
    std::cout << "GLFW window created" << std::endl;
  }

  glfwMakeContextCurrent(glfwWindow);

  // Loading all OpenGL function pointers with glad
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    exit(1);
  } else {
    std::cout << "GLAD initialized successfully" << std::endl;
  }

  glfwSetKeyCallback(glfwWindow, key_callback);
  glfwSetMouseButtonCallback(glfwWindow, mouse_button_callback);

  // Shaders
  std::string vertexShaderCode = R"(
  #version 330 core
  in vec3 position;

  void main() {
      gl_Position = vec4(position, 1.0f);
  }
  )";
  
  std::string fragmentShaderCode = R"(
  #version 330 core
  out vec4 outColor;

  void main() {
      outColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
  }
  )";

  std::string geometryShaderCode = R"(
  #version 330 core

  layout ( points ) in;
  layout ( triangle_strip, max_vertices = 4 ) out;
  in vec3 geomVelocity[];
  out float fragMass;
  uniform bool useQuads;
  const float size = 0.01;

  void main(void){
    vec3 vel = geomVelocity[0];
    float speed = length(vel);
    vec3 dir = normalize(vel);

    if (vel == vec3(0.0))
      dir = vec3(0.0, 1.0, 0.0);
    
    vec3 up = vec3(0.0, 1.0, 0.0);
    // Calcula el producto cruz entre los vectores normalizados
    vec3 crossProduct = cross(dir, up);
    // Calcula el ángulo en radianes utilizando la función acos()
    float angleRadians = acos(dot(dir, up));

    // Ajusta el ángulo según la dirección de giro
    if (dot(crossProduct, vec3(0.0, 0.0, 1.0)) < 0.0) {
      angleRadians = -angleRadians;
    }

    float angle = angleRadians;
    mat3 rot = mat3(
      cos(angle), -sin(angle), 0.0,
      sin(angle), cos(angle), 0.0,
      0.0, 0.0, 1.0
      );
    vec3 offset = vec3(-1.0, -1.0, 0.0) * size;
    offset = rot*offset;
    vec4 vertexPos = vec4(offset, 0.0) + gl_in[0].gl_Position;

    gl_Position = vertexPos;
    fragMass = speed;
    EmitVertex();

    offset = vec3(1.0, -1.0, 0.0) * size;
    offset = rot*offset;
    vertexPos = vec4(offset, 0.0) + gl_in[0].gl_Position;
    gl_Position = vertexPos;
    fragMass = speed;
    EmitVertex();

    offset = vec3(0.0, 1.0, 0.0) * size;
    offset = rot*offset;
    vertexPos = vec4(offset, 0.0) + gl_in[0].gl_Position;
    gl_Position = vertexPos;
    fragMass = speed;
    EmitVertex();

    EndPrimitive();
  }
  )";


  const char* vertexShaderSource = vertexShaderCode.c_str();
  const char* fragmentShaderSource = fragmentShaderCode.c_str();
  const char* geometryShaderSource = geometryShaderCode.c_str();

  // Create Vertex Shader Object and get its reference
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  // Attach Vertex Shader source to the Vertex Shader Object
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  // Compile the Vertex Shader into machine code
  glCompileShader(vertexShader);

  // Create Fragment Shader Object and get its reference
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  // Attach Fragment Shader source to the Vertex Shader Object
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  // Compile the Fragment Shader into machine code
  glCompileShader(fragmentShader);

  GLuint geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
  glShaderSource(geometryShader, 1, &geometryShaderSource, NULL);
  glCompileShader(geometryShader);

  // Create Shader Program Object and get its reference
  GLuint shaderProgram = glCreateProgram();
  // Attach the Vertex and Fragment Shaders to the Shader Program
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glAttachShader(shaderProgram, geometryShader);
  // Wrap-up / link all the shaders together into the Shader Program
  glLinkProgram(shaderProgram);

  // Delete the now useless Vertex and Fragment Shader Objects
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);
  glDeleteShader(geometryShader);

  // Use the shader program
  glUseProgram(shaderProgram);

  // Create reference containers for the Vertex Array Object and the Vertex Buffer Object
  GLuint VAO, VBO;

  // Generate the VAO and VBO with only 1 object each
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  // Make the VAO the current Vertex Array Object by binding it
  glBindVertexArray(VAO);

  // Bind the VBO specifying it's a GL_ARRAY_BUFFER
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  // Introduce the vertices into the VBO
  glBufferData(GL_ARRAY_BUFFER, meshWidth * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

  // Configure the Vertex Attribute so that OpenGL knows how to read the VBO
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  // Enable the Vertex Attribute so that OpenGL knows to use it
  glEnableVertexAttribArray(0);

  // Bind both the VBO and VAO to 0 so that we don't accidentally modify the VAO and VBO
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // Register this buffer object with CUDA
  cudaGraphicsGLRegisterBuffer(&cudaVBOResource, VBO, cudaGraphicsMapFlagsWriteDiscard);

  // Specify the color of the background
  glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
  // Clean the back buffer and assing the new color to it
  glClear(GL_COLOR_BUFFER_BIT);
  // Swap the back buffer with the front buffer
  glfwSwapBuffers(glfwWindow);

  std::cout << "Opening GLFW window" << std::endl;

  std::size_t size = sizeof(float3) * meshWidth;

  float3 *velDev;
  float3 *auxPosDev;
  float3 *auxVelDev;
  cudaMalloc(&auxPosDev, size);
  cudaMalloc(&velDev, size);
  cudaMalloc(&auxVelDev, size);

  createData(auxPosDev, auxVelDev);

  // float vertexData[24] = {-0.95, 0.95, 0.0,
  //                           0.95, 0.95, 0.0,
  //                           0.95, -0.95, 0.0,
  //                           0.95, -0.95, 0.0,
  //                           -0.95, -0.95, 0.0,
  //                           -0.95, 0.95, 0.0};

  // GLuint vao, vbo;
  // glGenVertexArrays(1, &vao);
  // glGenBuffers(1, &vbo);

  // glBindBuffer(GL_ARRAY_BUFFER, vbo);
  // glBufferData(GL_ARRAY_BUFFER, 24*sizeof(float), vertexData, GL_DYNAMIC_DRAW);

  // glBindVertexArray(vao);

  // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

  while (!glfwWindowShouldClose(glfwWindow))
  {
    // Using GLFW to check and process input events internally
    glfwPollEvents();

    if(test) {
      double xpos, ypos;
      glfwGetCursorPos(glfwWindow, &xpos, &ypos);
      float3 pos = {2*xpos/1000 - 1, 2*ypos/1000 - 1, 0};
      float3 auxP[meshWidth];
      cudaMemcpy(auxP, auxPosDev, meshWidth*sizeof(float3), cudaMemcpyDeviceToHost);
      //calculateDensityHost(pos, auxP, sRadius);
      std::cout << "Density in (" << xpos << ", " << ypos << ") = " << calculateDensityHost(pos, auxP, sRadius) << "\n";
      test = false;
    }

    if (stop) continue;
    
    // Run CUDA kernel to generate vertex positions
    runCuda(&cudaVBOResource, velDev, auxPosDev, auxVelDev);

    glClear(GL_COLOR_BUFFER_BIT);

    // Draw the triangle using the GL_POINTS primitive
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindVertexArray(VAO);
    glDrawArrays(GL_POINTS, 0, meshWidth);

    // glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // glBindVertexArray(vao);
    // glDrawArrays(GL_POINTS, 0, 6);

    glfwSwapBuffers(glfwWindow);

    // Update animation
    animTime += 0.04f;
  }

  cudaFree(auxPosDev);
  cudaFree(velDev);
  cudaFree(auxVelDev);
  
  std::cout << "GLFW window closed" << std::endl;

  return 0;
}