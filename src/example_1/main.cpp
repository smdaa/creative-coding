#include "cinder/Rand.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

using namespace ci;
using namespace ci::app;

#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080
#define NPARTICLES 100
#define MAX_RADIUS 2.0f
#define MIN_RADIUS 2.0f
#define MAX_VELOCITY 10.0f
#define MIN_VELOCITY -10.0f
#define WALL_BOUNCE_FACTOR 0.2f
#define FORCE_FEILD_FACTOR 2.0f
#define DT 0.9f
#define GRID_RESOLUTION 5
#define DRAW_GRID false
#define DRAW_PARTICLES false

#define BG_COLOR ColorA(220.0 / 255.0, 242.0 / 255.0, 241.0 / 255.0)

const ColorA PALETTE[] = {
    ColorA(15.0 / 255.0, 16.0 / 255.0, 53.0 / 255.0, 0.5f),
    ColorA(127.0 / 255.0, 199.0 / 255.0, 217.0 / 255.0, 0.5f),
    ColorA(54.0 / 255.0, 84.0 / 255.0, 134.0 / 255.0, 0.5f),

};

class Particle {
public:
  vec2 position;
  vec2 velocity;
  float radius;
  Color color;

  Particle(const vec2 &position, const vec2 &velocity, float radius,
           Color color)
      : position(position), velocity(velocity), radius(radius), color(color) {}

  void draw() {
    gl::color(color);
    gl::drawSolidCircle(position, radius);
  }

  void checkEdgeCollision() {
    if (position.x - radius < 0) {
      position.x = radius;
      velocity.x = -WALL_BOUNCE_FACTOR * velocity.x;
    } else if (position.x + radius > WINDOW_WIDTH) {
      position.x = WINDOW_WIDTH - radius;
      velocity.x = -WALL_BOUNCE_FACTOR * velocity.x;
    }

    if (position.y - radius < 0) {
      position.y = radius;
      velocity.y = -WALL_BOUNCE_FACTOR * velocity.y;
    } else if (position.y + radius > WINDOW_HEIGHT) {
      position.y = WINDOW_HEIGHT - radius;
      velocity.y = -WALL_BOUNCE_FACTOR * velocity.y;
    }
  }

  void checkParticleCollision(Particle &other) {
    vec2 relativePosition = other.position - position;
    vec2 relativeVelocity = other.velocity - velocity;
    float distance = glm::length(relativePosition);
    float combinedRadius = radius + other.radius;

    if (distance < combinedRadius) {
      // Particles are colliding
      float penetration = combinedRadius - distance;
      vec2 collisionNormal = glm::normalize(relativePosition);
      float relativeVelocityAlongNormal =
          glm::dot(relativeVelocity, collisionNormal);
      if (relativeVelocityAlongNormal > 0) {
        return;
      }
      float j = -relativeVelocityAlongNormal;
      vec2 impulse = j * collisionNormal;
      velocity -= impulse;
      other.velocity += impulse;
      position -= 0.5f * penetration * collisionNormal;
      other.position += 0.5f * penetration * collisionNormal;
    }
  }

  void updateVelocity(vec2 force, float dt) { velocity += force * dt; }
  void updatePosition(float dt) { position += velocity * dt; }
};

class World {
public:
  float dt;
  int gridNumRows;
  int gridNumCols;
  std::vector<std::vector<float>> grid;
  std::vector<Particle> particles;

  World() {
    dt = DT;
    gridNumRows = WINDOW_HEIGHT / GRID_RESOLUTION;
    gridNumCols = WINDOW_WIDTH / GRID_RESOLUTION;
    grid = std::vector<std::vector<float>>(
        gridNumRows, std::vector<float>(gridNumCols, 0.0f));
    particles = std::vector<Particle>();
  }

  void addParticle(const Particle &particle) { particles.push_back(particle); }

  void updateGrid(const vec2 &mousePos) {
    for (int i = 0; i < gridNumRows; ++i) {
      for (int j = 0; j < gridNumCols; ++j) {
        float x = static_cast<float>(j) * static_cast<float>(GRID_RESOLUTION);
        float y = static_cast<float>(i) * static_cast<float>(GRID_RESOLUTION);

        grid[i][j] = atan2(mousePos.y - y, mousePos.x - x);
      }
    }
  }

  void updateParticles(float dt) {
    for (int i = 0; i < particles.size(); ++i) {
      int rowIndex = static_cast<int>(
          std::min(std::max(particles[i].position.y /
                                static_cast<float>(GRID_RESOLUTION),
                            0.0f),
                   static_cast<float>(gridNumRows - 1)));
      int columnIndex = static_cast<int>(
          std::min(std::max(particles[i].position.x /
                                static_cast<float>(GRID_RESOLUTION),
                            0.0f),
                   static_cast<float>(gridNumCols - 1)));
      float gridValue = grid[rowIndex][columnIndex];
      vec2 force = FORCE_FEILD_FACTOR * vec2(cos(gridValue), sin(gridValue));
      particles[i].updateVelocity(force, dt);
      particles[i].updatePosition(dt);
      for (auto &other : particles) {
        if (&particles[i] != &other) {
          other.checkParticleCollision(particles[i]);
        }
      }
      particles[i].checkEdgeCollision();
    }
  }

  void update(const vec2 &mousePos) {
    updateGrid(mousePos);
    updateParticles(dt);
  }
};

class ParticleApp : public App {
public:
  void setup() override {
    setWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);

    for (int i = 0; i < NPARTICLES; i++) {
      float x = randFloat(0, WINDOW_WIDTH);
      float y = randFloat(0, WINDOW_HEIGHT);
      float vx = randFloat(MIN_VELOCITY, MAX_VELOCITY);
      float vy = randFloat(MIN_VELOCITY, MAX_VELOCITY);
      float radius = randFloat(MIN_RADIUS, MAX_RADIUS);
      Color color = Color(1.0f, 1.0f, 1.0f);
      Particle particle = Particle(vec2(x, y), vec2(vx, vy), radius, color);
      world.addParticle(particle);
    }
  }

  void mouseMove(MouseEvent event) override { mousePos = event.getPos(); }

  void keyDown(KeyEvent event) override {
    if (event.getChar() == 'q' || event.getChar() == 'Q') {
      quit();
    } else if (event.getChar() == 's' || event.getChar() == 'S') {
      writeImage("screenshot_" + std::to_string(frameNumber) + ".png",
                 copyWindowSurface());
      frameNumber++;
    }
  }

  void update() override { world.update(mousePos); }

  void draw() override {
    gl::clear(BG_COLOR);

    if (DRAW_GRID) {
      for (int i = 0; i < world.gridNumRows; ++i) {
        for (int j = 0; j < world.gridNumCols; ++j) {
          float angle = world.grid[i][j];
          float y = static_cast<float>(i) * static_cast<float>(GRID_RESOLUTION);
          float x = static_cast<float>(j) * static_cast<float>(GRID_RESOLUTION);
          float arrowLength = GRID_RESOLUTION / 2.0f;
          float circleRadius = 4.0f;

          vec2 start = vec2(x, y);
          vec2 end =
              vec2(x + arrowLength * cos(angle), y + arrowLength * sin(angle));
          gl::color(Color(1.0f, 1.0f, 1.0f));
          gl::drawLine(start, end);
          gl::drawSolidCircle(end, circleRadius);
        }
      }
    }

    if (DRAW_PARTICLES) {
      for (auto &particle : world.particles) {
        particle.draw();
      }
    }

    if (world.particles.size() > 2) {
      // Create a TriMesh
      TriMesh::Format format = TriMesh::Format().positions(3).colors(4);
      TriMesh mesh(format);

      // Extrude the path
      float extrusionDepth = 10.0f;
      for (const auto &particle : world.particles) {
        vec3 position(particle.position, 0);
        mesh.appendPosition(position);

        position.z += extrusionDepth;
        mesh.appendPosition(position);
      }

      // Add colors
      size_t colorIndex = 0;
      for (size_t i = 0; i < mesh.getNumVertices(); ++i) {
        mesh.appendColorRgba(
            PALETTE[colorIndex++ % (sizeof(PALETTE) / sizeof(PALETTE[0]))]);
      }

      // Add triangles
      for (size_t i = 0; i < mesh.getNumVertices() - 4; i += 4) {
        mesh.appendTriangle(i, i + 2, i + 4);
        mesh.appendTriangle(i + 1, i + 3, i + 5);
      }

      // Draw the mesh
      gl::enableAlphaBlending();
      gl::draw(mesh);
      gl::disableAlphaBlending();
    }
  }

private:
  World world;
  vec2 mousePos;
  int frameNumber = 0;
};
void prepareSettings(ParticleApp::Settings *settings) {
  settings->setResizable(false);
}

CINDER_APP(ParticleApp, RendererGl, prepareSettings)