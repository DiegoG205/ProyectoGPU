#ifndef _BUFFER_H_
#define _BUFFER_H_

#include "glcore/texture.h"
#include <vector>

class Buffer {
public:
  virtual void draw() = 0;
  virtual void build() = 0;
  virtual ~Buffer() = 0;
};

class VertexBuffer: public Buffer {
public:
  VertexBuffer();
  VertexBuffer(const std::vector<float> &vertices);
  VertexBuffer(VertexBuffer &&) = default;
  VertexBuffer(const VertexBuffer &) = default;
  VertexBuffer &operator=(VertexBuffer &&) = default;
  VertexBuffer &operator=(const VertexBuffer &) = default;
  ~VertexBuffer() override;

  void draw() override;
  void build() override;

protected:
  void bind();
  unsigned int vao_, vbo_;
  std::vector<float> vertices_;
};

class BasicBuffer: public VertexBuffer {
public:
  BasicBuffer();
  BasicBuffer(const std::vector<float> &vertices);
  BasicBuffer(const std::vector<float> &vertices,
              const std::vector<unsigned int> &indices);
  BasicBuffer(BasicBuffer &&) = default;
  BasicBuffer(const BasicBuffer &) = default;
  BasicBuffer &operator=(BasicBuffer &&) = default;
  BasicBuffer &operator=(const BasicBuffer &) = default;
  ~BasicBuffer() override;

  void draw() override;
  void build() override;

protected:
  void bind();
  unsigned int ebo_;
  std::vector<unsigned int> indices_;
};

class TextureBuffer : public BasicBuffer {
public:
  using BasicBuffer::BasicBuffer; // inherit constructors

  void draw() override;
  void addTexture(Texture *texture);
  void build() override;

private:
  std::vector<Texture *> textures_;
};
#endif // _BUFFER_H_
