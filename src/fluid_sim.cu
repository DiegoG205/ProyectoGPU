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

#define PRIME1 9439
#define PRIME2 17977

typedef unsigned int uint;


__device__ uint hashCell(uint xid, uint yid) {
  return (xid*PRIME1 + yid*PRIME2);
}

__device__ uint hashCell(uint2 cell){
  return (cell.x*PRIME1 + cell.y*PRIME2);
}

__device__ uint keyFromHash(uint hash, uint tableSize) {
  return hash % tableSize;
}

__device__ uint2 pos2Cell(float3 pos, float cellSide) {
  return uint2{static_cast<uint>(floor(pos.x/cellSide)), static_cast<uint>(floor(pos.y/cellSide))};
};

/*
( 1,-1) ( 1, 0) ( 1, 1) 
( 0,-1) ( 0, 0) ( 0, 1)
(-1,-1) (-1, 0) (-1, 1)

(1, 1) -> 95683
sort basado en 95683
[(95683, 0), (95683, 53), (3284854, 12), ...]
*/


__global__ void calcHash(int n, float3 *posData, uint2 *hashData, float dt) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  float3 pos = posData[2*index];
  float3 vel = posData[2*index + 1];
  float3 predPos = {0,0,0};
  predPos.x = pos.x + vel.x * dt;
  predPos.y = pos.y + vel.y * dt;
  predPos.z = pos.z + vel.z * dt;

  float radius = 3.0;
  float cellSide = radius;

  uint2 cell = pos2Cell(predPos, cellSide);

  hashData[index].x = hashCell(cell);
  hashData[index].y = index;
};

__global__ void bitonicSortStep(uint2* hashData, int j, int k) {
  unsigned int ind = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ixj = ind^j;

  uint2 pair0 = hashData[ind];
  uint2 pair1 = hashData[ixj];


  if ((ixj) > ind) {
    if ((ind&k == 0)){
      if (pair0.x > pair1.x) {
        hashData[ind] = pair1;
        hashData[ixj] = pair0;
      }
    }
    if ((ind&k) != 0){
      if (pair0.x < pair1.x) {
        hashData[ind] = pair1;
        hashData[ixj] = pair0;
      }
    }
  }
};

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

__device__ float calculateDensity(int n, float3 pos, float3* positions, float radius, float dt) {
  
  float density = 0;
  const float mass = 1;
  

  for (int i = 0; i < n; i++) {
    
    float3 other = positions[2*i];
    float3 vel = positions[2*i+1];
    other.x += vel.x*dt;
    other.y += vel.y*dt;
    other.z += vel.z*dt;
  
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

// Actualizar esta
__device__ float calculateDensityHash(int n, float3 pos, float3* positions, uint2* hashData, float radius, float dt) {
  
  float density = 0;
  const float mass = 1;
  
  //Obtener la celda y el hash de la celda
  uint2 cell = pos2Cell(pos, radius);
  uint hash = hashCell(cell);
  uint index = 0;
  

  for (int i = -1; i < 2; i++){
    for (int j = -1; j < 2; j++){

      uint2 cell = {i + cell.x, j + cell.y};
      uint key = keyFromHash(hashCell(cell), n);
      uint index = hashData[key].y;

      float3 other = positions[2*index];
      float3 vel = positions[2*index+1];
      other.x += vel.x*dt;
      other.y += vel.y*dt;
      other.z += vel.z*dt;
  
      float dx = pos.x - other.x;
      float dy = pos.y - other.y;
      float dz = pos.z - other.z;
      //float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
      float dist = norm3d(dx, dy, dz);

      float influence = smoothingKernel(radius, dist);

      density += influence * mass;
  
    }
  }
  return density;
};

__device__ float densityToPressure(float density, float targetDensity, float pressureMultiplier) {
  float delta = density - targetDensity;
  return (delta * pressureMultiplier);
};

__device__ float calculateSharedPressure(float density1, float density2, float targetDensity, float pressureMultiplier) {
  float pressure1 = densityToPressure(density1, targetDensity, pressureMultiplier);
  float pressure2 = densityToPressure(density2, targetDensity, pressureMultiplier);
  return (pressure1+pressure2)/2;
};

// Actualizar esta
__device__ float3 calculatePressure(int n, float3 pos, float3* data, float* densities, float radius, float trgDen, float pressMult) {

  float3 pressure = {0,0,0};
  float mass = 1;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  float selfDensity = densities[index];

  for(int i = 0; i < n; i++) {
    
    if (i == index) continue;

    float3 other = data[2*i];

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

// Actualizar esta
__device__ float3 calculateViscosity(int n, float3 pos, float3* data, float radius, float viscStr) {

  float3 viscosity = {0, 0, 0};
  float mass = 1;
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  float3 selfVelocity = data[2*index+1];

  for (int i = 0; i < n; i++){
    if (i == index) continue;

    float3 other = data[2*i];
    float3 otherVel = data[2*i+1];

    float dx = pos.x - other.x;
    float dy = pos.y - other.y;
    float dz = pos.z - other.z;
    float dist = norm3d(dx, dy, dz);
    if (dist <= 0.01) continue;
    float3 dir = {-dx/dist, -dy/dist, -dz/dist};

    float influence = smoothingKernel(radius, dist);

    viscosity.x += (otherVel.x - selfVelocity.x)*influence;
    viscosity.y += (otherVel.y - selfVelocity.y)*influence;
    viscosity.z += (otherVel.z - selfVelocity.z)*influence;
    
  }

  viscosity.x *= viscStr;
  viscosity.y *= viscStr;
  viscosity.z *= viscStr;

  return viscosity;
}

__global__ void updateDensities(int n, float3 *posData, float *densities, float radius, float dt) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  float3 predPos;
  float3 pos = posData[2*index];
  float3 vel = posData[2*index + 1];
  predPos.x = pos.x + vel.x * dt;
  predPos.y = pos.y + vel.y * dt;
  predPos.z = pos.z + vel.z * dt;
  densities[index] = calculateDensity(n, predPos, posData, radius, dt);
  //printf("%d: %f\n", index, densities[index]);
};

__global__ void fluid_kernel(int n, float3 *data, float3* dataAux, float *densities, float dt, 
                             float radius, float trgDen, float pressMult, float grav, float viscStr,
                             float mouseX, float mouseY, int mouseAction) {

  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;


  float3 pos = data[2*index];
  float3 vel = data[2*index + 1];

  // Gravity
  vel.y -= grav * dt;

  // Pressure force
  float3 pressure = calculatePressure(n, pos, data, densities, radius, trgDen, pressMult);
  float3 pressureAcc = {pressure.x / densities[index], pressure.y / densities[index], pressure.z / densities[index]};

  vel.x += pressureAcc.x * dt;
  vel.y += pressureAcc.y * dt;
  vel.z += pressureAcc.z * dt;

  // Viscosity force
  float3 viscosity = calculateViscosity(n, pos, data, radius, viscStr);
  float3 viscosityAcc = {viscosity.x / densities[index], viscosity.y / densities[index], viscosity.z / densities[index]};

  vel.x += viscosityAcc.x * dt;
  vel.y += viscosityAcc.y * dt;
  vel.z += viscosityAcc.z * dt;

  // Border Collisions
  if (std::abs(pos.y + 0.05) >= 49.95){ 
    vel.y = -vel.y*0.7;
    pos.y = 49.9*copysign(1.0, pos.y);
  }
  if (std::abs(pos.x + 0.05) >= 49.95) {
    vel.x = -vel.x*0.7;
    pos.x = 49.9*copysign(1.0, pos.x);
  }
  
  if (mouseAction != 0) {
    float dx = pos.x - mouseX;
    float dy = pos.y - mouseY;
    float dist = sqrt(dx*dx + dy*dy);
    if (dist <= 6.f) {
      float2 dir = {-dx/dist, -dy/dist};
      vel.x = dir.x * 20 * mouseAction;
      vel.y = dir.y * 20 * mouseAction;
    }
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
  float volume = 2 * PI * std::pow(radius, 3) / 3;

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