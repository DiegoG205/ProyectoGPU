#define STB_IMAGE_IMPLEMENTATION
#include <glad/glad.h>
#include <glcore/texture.h>
#include <iostream>
#include <stb_image.h>

using vec = std::vector<unsigned char>;


Texture::Texture(const vec &data, int width, int height, int nrChannels)
    : width_(width), height_(height), nrChannels_(nrChannels) {
  build(data.data());
}

Texture::Texture(const std::string &path) {
  stbi_set_flip_vertically_on_load(true);
  unsigned char *data =
      stbi_load(path.c_str(), &width_, &height_, &nrChannels_, 0);
  if (data) {
    build(data);
  } else {
    std::cerr << "Failed to load texture (" << path << ")" << std::endl;
  }
  stbi_image_free(data);
}

Texture::~Texture() {
  glDeleteTextures(1, &id_); 
}

void Texture::build(const unsigned char *data) {
  unsigned int format;
  switch (nrChannels_) {
    case 3:
      format = GL_RGB;
      break;
    case 4:
      format = GL_RGBA;
      break;
    default:
      format = 0;
      std::cerr << "Unknown format (" << nrChannels_ << ")" << std::endl;
      break;
  }
  glGenTextures(1, &id_);
  glBindTexture(GL_TEXTURE_2D, id_);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, format, width_, height_, 0, format,
               GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);
}

void Texture::bind() { glBindTexture(GL_TEXTURE_2D, id_); }

