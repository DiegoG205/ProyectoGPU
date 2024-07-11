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

#define PI 3.14159265358979323846

__device__ float smoothingKernel(float radius, float dist) {
  float volume = 2 * PI * std::pow(radius, 3) / 3;
  float value = max(0.0f, radius - dist);
  return (value * value / volume);
};

__device__ float smoothingKernelDerivative(float radius, float dist) {
  float scale = 4/3 * PI * std::pow(radius, 3);
  float value = max(0.f, radius - dist);
  return value * scale;
};

__device__ float calculateDensity(int n, float3 pos, float3* positions, float radius) {
  
  float density = 0;
  const float mass = 1;

  for (int i = 0; i < n; i++) {
    
    float3 other = positions[2*i];

    float dx = pos.x - other.x;
    float dy = pos.y - other.y;
    float dz = pos.z - other.z;
    //float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    float dist = norm3d(dx, dy, dz);

    float influence = smoothingKernel(radius, dist);

    density += influence * mass;
  }

  return density;
};

__device__ float densityToPressure(float density, float targetDensity, float pressureMultiplier) {
  float delta = density - targetDensity;
  return (abs(delta) * pressureMultiplier);
};

__device__ float calculateSharedPressure(float density1, float density2, float targetDensity, float pressureMultiplier) {
  float pressure1 = densityToPressure(density1, targetDensity, pressureMultiplier);
  float pressure2 = densityToPressure(density2, targetDensity, pressureMultiplier);
  return (pressure1+pressure2)/2;
};

__device__ float3 calculatePressure(int n, float3 pos, float3* positions, float* densities, float radius, float trgDen, float pressMult) {

  float3 pressure = {0,0,0};
  float mass = 1;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  float selfDensity = densities[index];

  for(int i = 0; i < n; i++) {
    
    if (i == index) continue;

    float3 other = positions[2*i];

    float dx = pos.x - other.x;
    float dy = pos.y - other.y;
    float dz = pos.z - other.z;
    float dist = norm3d(dx, dy, dz);
    if (dist <= 0.01) continue;
    float3 dir = {-dx/dist, -dy/dist, -dz/dist};

    float slope = smoothingKernelDerivative(radius, dist);

    float density = densities[i];
    if (density == 0.0) {
      continue;
    };
    // float val = densityToPressure(density, trgDen, pressMult) * slope * mass / density;
    float val = calculateSharedPressure(density, selfDensity, trgDen, pressMult) * slope * mass / density;

    pressure.x -= dir.x * val;
    pressure.y -= dir.y * val;
    pressure.z -= dir.z * val;
  }

  return pressure;
}

__global__ void updateDensities(int n, float3 *posData, float *densities, float radius) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  densities[index] = calculateDensity(n, posData[2*index], posData, radius);
  //printf("%d: %f\n", index, densities[index]);
};

__global__ void fluid_kernel(int n, float3 *data, float3* dataAux, float *densities, float dt, 
                             float radius, float trgDen, float pressMult) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;


  float3 pos = data[2*index];
  float3 vel = data[2*index + 1];

  // Gravity
  // vel.y -= 9.8 * dt;

  // Pressure force
  float3 pressure = calculatePressure(n, pos, data, densities, radius, trgDen, pressMult);
  float3 pressureAcc = {pressure.x / densities[index], pressure.y / densities[index], pressure.z / densities[index]};

  // TODO: sumar la aceleracion, no asignarla directamente
  vel.x += pressureAcc.x * dt;
  vel.y += pressureAcc.y * dt;
  vel.z += pressureAcc.z * dt;

  // Border Collisions
  if (std::abs(pos.y + 0.05) >= 19.95){ 
    vel.y = -vel.y*0.85;
    pos.y = 19.9*copysign(1.0, pos.y);
  }
  if (std::abs(pos.x + 0.05) >= 19.95) {
    vel.x = -vel.x*0.85;
    pos.x = 19.9*copysign(1.0, pos.x);
  }
  
  pos.x += vel.x * dt;
  pos.y += vel.y * dt;
  pos.z += vel.z * dt;

  dataAux[2*index] = pos;
  dataAux[2*index + 1] = vel;

};

float calculateDensityHost(int n, float3 pos, float3* positions, float radius) {
  
  float density = 0;
  const float mass = 1;
  float volume = PI * std::pow(radius, 4) / 6;

  for (int i = 0; i < n; i++) {

    float3 other = positions[2*i];

    float dx = pos.x - other.x;
    float dy = pos.y - other.y;
    float dz = pos.z - other.z;
    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    float value = max(0.0f, radius - dist);

    float influence = (value * value / volume);

    density += influence * mass;
  }

  return density;
};