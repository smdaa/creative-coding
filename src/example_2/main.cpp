#include "cinder/CinderImGui.h"
#include "cinder/Perlin.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <glm/fwd.hpp>
#include <vector>

using namespace std;
using namespace ci;
using namespace ci::app;

#define WINDOW_WIDTH 1000
#define WINDOW_HEIGHT 1000
#define DEFAULT_GRID_RESOLUTION 20
#define DEFAULT_DIFFUSION_FACTOR 0.000001f
#define DEFAULT_VISCOSITY_FACTOR 0.0001f
#define DEFAULT_TIMESTEP 0.01f

class FluidCell {

private:
  float density;
  vec2 velocity;

public:
  FluidCell() : density(0.0f), velocity(vec2(0.0f, 0.0f)) {}
  FluidCell(float _density, const vec2 &_velocity)
      : density(_density), velocity(_velocity) {}

  float getDensity() const { return density; }
  vec2 getVelocity() const { return velocity; }

  void setDensity(float _density) { density = _density; }
  void setVelocity(const vec2 _velocity) { velocity = _velocity; }
};

class FluidGrid {

private:
  int numRows;
  int numColumns;
  vector<vector<FluidCell>> grid;

public:
  FluidGrid(int _numRows, int _numColumns)
      : numRows(_numRows), numColumns(_numColumns) {
    grid.resize(numRows);
    for (int i = 0; i < numRows; ++i) {
      grid[i].resize(numColumns, FluidCell());
    }
  }

  int getNumRows() const { return numRows; }
  int getNumColumns() const { return numColumns; }
};

class SmokeApp : public App {
public:
  void setup() override {
    setWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    initDensity(WINDOW_WIDTH, WINDOW_HEIGHT, gridResolution);
    initVelocity(WINDOW_WIDTH, WINDOW_HEIGHT, gridResolution);

    ImGui::Initialize();

    getWindow()->getSignalMouseDown().connect(
        [this](MouseEvent event) { onMouseDown(event); });

    getWindow()->getSignalMouseDrag().connect(
        [this](MouseEvent event) { onMouseDrag(event); });

    getWindow()->getSignalMouseUp().connect(
        [this](MouseEvent event) { onMouseUp(event); });
  }

  void keyDown(KeyEvent event) override {
    if (event.getChar() == 'q' || event.getChar() == 'Q') {
      quit();
    } else if (event.getChar() == 'c' || event.getChar() == 'C') {
      clearCurrentGrids();
      clearOldGrids();
    } else if (event.getChar() == 'p' || event.getChar() == 'P') {
      simulationPaused = !simulationPaused;
    }
  }

  void update() override {
    ImGui::Begin("Parameters");
    if (ImGui::Button("Clear")) {
      clearCurrentGrids();
      clearOldGrids();
    }
    ImGui::Checkbox("Pause", &simulationPaused);
    ImGui::SliderFloat("Time Step", &timeStep, 0.0f, 10.0f);
    ImGui::SliderFloat("Diffusion Factor", &diffusionFactor, 0.0f, 10.0f);
    ImGui::End();

    if (!simulationPaused) {
      clearOldGrids();
      stepDensity(diffusionFactor, timeStep);
      stepVelocity(viscosityFactor, timeStep);
    }
  }

  void resize() override {
    int newWidth = getWindowWidth();
    int newHeight = getWindowHeight();
    initDensity(newWidth, newHeight, gridResolution);
    initVelocity(newWidth, newHeight, gridResolution);
  }

  void draw() override {
    gl::clear(Color(0, 0, 0));
    drawDensity();
  }

private:
  vector<vector<float>> densityGrid;
  vector<vector<float>> densityGridOld;
  vector<vector<float>> velocityGridX;
  vector<vector<float>> velocityGridXOld;
  vector<vector<float>> velocityGridY;
  vector<vector<float>> velocityGridYOld;

  int gridResolution = DEFAULT_GRID_RESOLUTION;
  float diffusionFactor = DEFAULT_DIFFUSION_FACTOR;
  float viscosityFactor = DEFAULT_VISCOSITY_FACTOR;
  float timeStep = DEFAULT_TIMESTEP;

  bool simulationPaused = false;
  vec2 lastMousePositon;

  void initDensity(int windowWidth, int windowHeight, int gridResolution) {
    int gridNumRows = windowHeight / gridResolution;
    int gridNumCols = windowWidth / gridResolution;
    densityGrid.clear();
    densityGridOld.clear();
    densityGrid.resize(gridNumRows, vector<float>(gridNumCols, 0.0f));
    densityGridOld.resize(gridNumRows, vector<float>(gridNumCols, 0.0f));
    float initialDensityValue = 0.0f;
    for (size_t i = 0; i < gridNumRows; ++i) {
      for (size_t j = 0; j < gridNumCols; ++j) {
        densityGrid[i][j] = initialDensityValue;
        densityGridOld[i][j] = initialDensityValue;
      }
    }
  }

  void initVelocity(int windowWidth, int windowHeight, int gridResolution) {
    int gridNumRows = windowHeight / gridResolution;
    int gridNumCols = windowWidth / gridResolution;
    velocityGridX.clear();
    velocityGridXOld.clear();
    velocityGridY.clear();
    velocityGridYOld.clear();
    velocityGridX.resize(gridNumRows, vector<float>(gridNumCols, 0.0f));
    velocityGridXOld.resize(gridNumRows, vector<float>(gridNumCols, 0.0f));
    velocityGridY.resize(gridNumRows, vector<float>(gridNumCols, 0.0f));
    velocityGridYOld.resize(gridNumRows, vector<float>(gridNumCols, 0.0f));
    float initialVelocityX = 0.0f;
    float initialVelocityY = 0.0f;
    for (size_t i = 0; i < gridNumRows; ++i) {
      for (size_t j = 0; j < gridNumCols; ++j) {
        velocityGridX[i][j] = initialVelocityX;
        velocityGridXOld[i][j] = initialVelocityX;
        velocityGridY[i][j] = initialVelocityY;
        velocityGridYOld[i][j] = initialVelocityY;
      }
    }
  }

  void clearGrid(vector<vector<float>> &targetGrid) {
    size_t n = targetGrid.size();
    size_t m = targetGrid[0].size();
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        targetGrid[i][j] = 0.0f;
      }
    }
  }

  void clearOldGrids() {
    clearGrid(densityGridOld);
    clearGrid(velocityGridXOld);
    clearGrid(velocityGridYOld);
  }

  void clearCurrentGrids() {
    clearGrid(densityGrid);
    clearGrid(velocityGridX);
    clearGrid(velocityGridY);
  }

  void addDensitySource(const vec2 &position) {
    int gridX = position.x / gridResolution;
    int gridY = position.y / gridResolution;
    if (gridX >= 0 && gridX < densityGrid[0].size() && gridY >= 0 &&
        gridY < densityGrid.size()) {
      densityGrid[gridY][gridX] += 50.0f;
    }
  }

  void addVelocitySource(const vec2 &position, const vec2 &direction) {
    int gridX = position.x / gridResolution;
    int gridY = position.y / gridResolution;
    if (gridX >= 0 && gridX < densityGrid[0].size() && gridY >= 0 &&
        gridY < densityGrid.size()) {
      velocityGridX[gridY][gridX] += direction.x;
      velocityGridY[gridY][gridX] += direction.y;
    }
  }

  void addSource(vector<vector<float>> &targetGrid,
                 const vector<vector<float>> &sourceGrid, float timeStep) {
    size_t n = targetGrid.size();
    size_t m = targetGrid[0].size();
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        targetGrid[i][j] += timeStep * sourceGrid[i][j];
      }
    }
  }

  void setBoundary(vector<vector<float>> &targetGrid, int b) {
    size_t n = targetGrid.size();
    size_t m = targetGrid[0].size();
    for (size_t i = 1; i < n - 1; ++i) {
      targetGrid[i][0] = (b == 2) ? -targetGrid[i][1] : targetGrid[i][1];
      targetGrid[i][m - 1] =
          (b == 2) ? -targetGrid[i][m - 2] : targetGrid[i][m - 2];
    }
    for (size_t j = 1; j < m - 1; ++j) {
      targetGrid[0][j] = (b == 1) ? -targetGrid[1][j] : targetGrid[1][j];
      targetGrid[n - 1][j] =
          (b == 1) ? -targetGrid[n - 2][j] : targetGrid[n - 2][j];
    }
    targetGrid[0][0] = 0.5 * (targetGrid[1][0] + targetGrid[0][1]);
    targetGrid[0][m - 1] = 0.5 * (targetGrid[1][m - 1] + targetGrid[0][m - 2]);
    targetGrid[n - 1][0] = 0.5 * (targetGrid[n - 1][1] + targetGrid[n - 2][0]);
    targetGrid[n - 1][m - 1] =
        0.5 * (targetGrid[n - 1][m - 2] + targetGrid[n - 2][m - 1]);
  }

  void diffuse(vector<vector<float>> &targetGrid,
               const vector<vector<float>> &targetGridOld,
               size_t gaussSeidelIterations, float diffusionFactor, int b,
               float dt) {
    size_t n = targetGrid.size();
    size_t m = targetGrid[0].size();
    float a = dt * diffusionFactor * n * m;
    for (size_t k = 0; k < gaussSeidelIterations; ++k) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
          float sum = 0.0f;
          sum += (i > 0) ? targetGridOld[i - 1][j] : 0.0f;     // Left
          sum += (i < n - 1) ? targetGridOld[i + 1][j] : 0.0f; // Right
          sum += (j > 0) ? targetGridOld[i][j - 1] : 0.0f;     // Up
          sum += (j < m - 1) ? targetGridOld[i][j + 1] : 0.0f; // Down

          targetGrid[i][j] = (targetGridOld[i][j] + a * sum) / (1 + 4 * a);
        }
      }
      setBoundary(targetGrid, b);
    }
  }

  void advect(vector<vector<float>> &densityGrid,
              const vector<vector<float>> &densityGridOld,
              const vector<vector<float>> &velocityGridX,
              const vector<vector<float>> &velocityGridY, int b, float dt) {
    float x, y;
    size_t n = densityGrid.size();
    size_t m = densityGrid[0].size();
    float dtRatio = dt * (max(n, m) - 1);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < m; ++j) {
        x = i - dtRatio * velocityGridX[i][j];
        y = j - dtRatio * velocityGridY[i][j];
        x = std::max(0.5f, std::min((float)n - 1.5f, x));
        y = std::max(0.5f, std::min((float)m - 1.5f, y));
        int x0 = (int)x;
        int y0 = (int)y;
        int x1 = std::min(x0 + 1, (int)n - 1);
        int y1 = std::min(y0 + 1, (int)m - 1);
        float sx1 = x - x0;
        float sx0 = 1.0f - sx1;
        float sy1 = y - y0;
        float sy0 = 1.0f - sy1;
        densityGrid[i][j] =
            sx0 *
                (sy0 * densityGridOld[x0][y0] + sy1 * densityGridOld[x0][y1]) +
            sx1 * (sy0 * densityGridOld[x1][y0] + sy1 * densityGridOld[x1][y1]);
      }
    }
    setBoundary(densityGrid, b);
  }

  void stepDensity(float diffusionFactor, float timeStep) {
    addSource(densityGrid, densityGridOld, timeStep);
    diffuse(densityGridOld, densityGrid, 20, diffusionFactor, 0, timeStep);
    advect(densityGrid, densityGridOld, velocityGridX, velocityGridY, 0,
           timeStep);
  }

  void project(vector<vector<float>> &velocityGridX,
               vector<vector<float>> &velocityGridY, vector<vector<float>> &p,
               vector<vector<float>> &div, size_t gaussSeidelIterations) {

    size_t n = velocityGridX.size();
    size_t m = velocityGridX[0].size();
    for (size_t i = 1; i < n - 1; ++i) {
      for (size_t j = 1; j < m - 1; ++j) {
        div[i][j] = -0.5 * (velocityGridX[i + 1][j] - velocityGridX[i - 1][j] +
                            velocityGridY[i][j + 1] - velocityGridY[i][j - 1]);
        p[i][j] = 0.0;
      }
    }
    setBoundary(div, 0);
    setBoundary(p, 0);
    for (size_t k = 0; k < gaussSeidelIterations; ++k) {
      for (size_t i = 1; i < n - 1; ++i) {
        for (size_t j = 1; j < m - 1; ++j) {
          p[i][j] = (div[i][j] + p[i - 1][j] + p[i + 1][j] + p[i][j - 1] +
                     p[i][j + 1]) /
                    4;
        }
      }
      setBoundary(p, 0);
    }
    for (size_t i = 1; i < n - 1; ++i) {
      for (size_t j = 1; j < m - 1; ++j) {
        velocityGridX[i][j] -= 0.5 * (p[i + 1][j] - p[i - 1][j]);
        velocityGridY[i][j] -= 0.5 * (p[i][j + 1] - p[i][j - 1]);
      }
    }
    setBoundary(velocityGridX, 1);
    setBoundary(velocityGridY, 2);
  }

  void stepVelocity(float viscosityFactor, float timeStep) {
    addSource(velocityGridX, velocityGridXOld, timeStep);
    addSource(velocityGridY, velocityGridYOld, timeStep);
    diffuse(velocityGridXOld, velocityGridX, 20, viscosityFactor, 1, timeStep);
    diffuse(velocityGridYOld, velocityGridY, 20, viscosityFactor, 2, timeStep);
    project(velocityGridXOld, velocityGridYOld, velocityGridX, velocityGridY,
            20);
    advect(velocityGridX, velocityGridXOld, velocityGridXOld, velocityGridYOld,
           1, timeStep);
    advect(velocityGridY, velocityGridYOld, velocityGridXOld, velocityGridYOld,
           2, timeStep);
    project(velocityGridX, velocityGridY, velocityGridXOld, velocityGridYOld,
            20);
  }

  void onMouseDown(MouseEvent event) { lastMousePositon = event.getPos(); }

  void onMouseDrag(MouseEvent event) {
    vec2 currentMousePosition = event.getPos();
    vec2 dragDirection = currentMousePosition - lastMousePositon;
    addDensitySource(currentMousePosition);
    addVelocitySource(currentMousePosition, dragDirection);
    lastMousePositon = currentMousePosition;
  }

  void onMouseUp(MouseEvent event) { lastMousePositon = vec2(0, 0); }

  void drawDensity() {
    float cellWidth = (float)getWindowWidth() / densityGrid[0].size();
    float cellHeight = (float)getWindowHeight() / densityGrid.size();

    for (size_t i = 0; i < densityGrid.size(); ++i) {
      for (size_t j = 0; j < densityGrid[i].size(); ++j) {
        float x = j * cellWidth;
        float y = i * cellHeight;

        float density = densityGrid[i][j];
        ColorA color(1.0f, 1.0f, 1.0f, density);

        gl::color(color);
        gl::drawSolidRect(Rectf(x, y, x + cellWidth, y + cellHeight));
      }
    }
  }
};

void prepareSettings(SmokeApp::Settings *settings) {
  settings->setResizable(true);
}

CINDER_APP(SmokeApp, RendererGl, prepareSettings)
