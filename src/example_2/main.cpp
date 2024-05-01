#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <glm/fwd.hpp>
#include <vector>

using namespace std;
using namespace ci;
using namespace ci::app;

#define DEFAULT_WINDOW_WIDTH 500
#define DEFAULT_WINDOW_HEIGHT 500
#define DEFAULT_GRID_RESOLUTION 10
#define DEFAULT_DIFFUSION_FACTOR 0.0001f
#define DEFAULT_VISCOSITY_FACTOR 0.0001f
#define DEFAULT_TIMESTEP 0.01f

class FluidGrid {

public:
  int numRows;
  int numColumns;
  int cellWidth;
  int cellHeight;
  vector<vector<float>> densityGrid;
  vector<vector<float>> velocityGridX;
  vector<vector<float>> velocityGridY;

  FluidGrid(int _numRows, int _numColumns, int _cellWidth, int _cellHeight)
      : numRows(_numRows), numColumns(_numColumns), cellWidth(_cellWidth),
        cellHeight(_cellHeight) {
    densityGrid.resize(numRows);
    velocityGridX.resize(numRows);
    velocityGridY.resize(numRows);
    for (int i = 0; i < numRows; ++i) {
      densityGrid[i].resize(numColumns);
      velocityGridX[i].resize(numColumns);
      velocityGridY[i].resize(numColumns);
      for (int j = 0; j < numColumns; ++j) {
        densityGrid[i][j] = 0.0f;
        velocityGridX[i][j] = 0.0f;
        velocityGridY[i][j] = 0.0f;
      }
    }
  }

  void reset() {
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numColumns; ++j) {
        densityGrid[i][j] = 0.0f;
        velocityGridX[i][j] = 0.0f;
        velocityGridY[i][j] = 0.0f;
      }
    }
  }

  void draw() const {
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numColumns; ++j) {
        float x = j * cellWidth;
        float y = i * cellHeight;
        float density = densityGrid[i][j];
        ColorA color(1.0f, 1.0f, 1.0f, density);
        gl::color(color);
        gl::drawSolidRect(Rectf(x, y, x + cellWidth, y + cellHeight));
      }
    }
  }

  void increaseCellDensity(int i, int j, float value) {
    if (i >= 0 && i < numRows && j >= 0 && j < numColumns) {
      densityGrid[i][j] += value;
    }
  }

  void increaseCellVelocity(int i, int j, float valueX, float valueY) {
    if (i >= 0 && i < numRows && j >= 0 && j < numColumns) {
      velocityGridX[i][j] += valueX;
      velocityGridY[i][j] += valueY;
    }
  }
};

void setBounds(int numRows, int numColumns, vector<vector<float>> &grid,
               int b) {
  for (int i = 1; i < numRows - 1; ++i) {
    grid[i][0] = (b == 2) ? -grid[i][1] : grid[i][1];
    grid[i][numColumns - 1] =
        (b == 2) ? -grid[i][numColumns - 2] : grid[i][numColumns - 2];
  }
  for (int j = 1; j < numColumns - 1; ++j) {
    grid[0][j] = (b == 1) ? -grid[1][j] : grid[1][j];
    grid[numRows - 1][j] =
        (b == 1) ? -grid[numRows - 2][j] : grid[numRows - 2][j];
  }
  grid[0][0] = 0.5 * (grid[1][0] + grid[0][1]);
  grid[0][numColumns - 1] =
      0.5 * (grid[1][numColumns - 1] + grid[0][numColumns - 2]);
  grid[numRows - 1][0] = 0.5 * (grid[numRows - 1][1] + grid[numRows - 2][0]);
  grid[numRows - 1][numColumns - 1] = 0.5 * (grid[numRows - 1][numColumns - 2] +
                                             grid[numRows - 2][numColumns - 1]);
}

void diffuse(int numRows, int numColumns, vector<vector<float>> &outGrid,
             const vector<vector<float>> &inGrid, int gaussSeidelIterations,
             float factor, int b, float timeStep) {
  float a = timeStep * factor * numRows * numColumns;
  for (int k = 0; k < gaussSeidelIterations; ++k) {
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numColumns; ++j) {
        float sum = 0.0f;
        sum += (i > 0) ? inGrid[i - 1][j] : 0.0f;              // Left
        sum += (i < numRows - 1) ? inGrid[i + 1][j] : 0.0f;    // Right
        sum += (j > 0) ? inGrid[i][j - 1] : 0.0f;              // Up
        sum += (j < numColumns - 1) ? inGrid[i][j + 1] : 0.0f; // Down
        outGrid[i][j] = (inGrid[i][j] + a * sum) / (1 + 4 * a);
      }
    }
    setBounds(numRows, numColumns, outGrid, b);
  }
}

void advect(int numRows, int numColumns, vector<vector<float>> &outGrid,
            const vector<vector<float>> &inGrid,
            const vector<vector<float>> &velocityGridX,
            const vector<vector<float>> &velocityGridY, int b, float timeStep) {
  float x, y;
  float dtRatio = timeStep * (max(numRows, numColumns) - 1);
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numColumns; ++j) {
      x = i - dtRatio * velocityGridX[i][j];
      y = j - dtRatio * velocityGridY[i][j];
      x = std::max(0.5f, std::min(static_cast<float>(numRows) - 1.5f, x));
      y = std::max(0.5f, std::min(static_cast<float>(numColumns) - 1.5f, y));
      int x0 = (int)x;
      int y0 = (int)y;
      int x1 = std::min(x0 + 1, (int)numRows - 1);
      int y1 = std::min(y0 + 1, (int)numColumns - 1);
      float sx1 = x - x0;
      float sx0 = 1.0f - sx1;
      float sy1 = y - y0;
      float sy0 = 1.0f - sy1;
      outGrid[i][j] = sx0 * (sy0 * inGrid[x0][y0] + sy1 * inGrid[x0][y1]) +
                      sx1 * (sy0 * inGrid[x1][y0] + sy1 * inGrid[x1][y1]);
    }
  }
  setBounds(numRows, numColumns, outGrid, b);
}

void project(int numRows, int numColumns, vector<vector<float>> &velocityGridX,
             vector<vector<float>> &velocityGridY, vector<vector<float>> &p,
             vector<vector<float>> &div, int gaussSeidelIterations) {

  for (int i = 1; i < numRows - 1; ++i) {
    for (int j = 1; j < numColumns - 1; ++j) {
      div[i][j] = -0.5 * (velocityGridX[i + 1][j] - velocityGridX[i - 1][j] +
                          velocityGridY[i][j + 1] - velocityGridY[i][j - 1]);
      p[i][j] = 0.0;
    }
  }
  setBounds(numRows, numColumns, div, 0);
  setBounds(numRows, numColumns, p, 0);
  for (int k = 0; k < gaussSeidelIterations; ++k) {
    for (int i = 1; i < numRows - 1; ++i) {
      for (int j = 1; j < numColumns - 1; ++j) {
        p[i][j] = (div[i][j] + p[i - 1][j] + p[i + 1][j] + p[i][j - 1] +
                   p[i][j + 1]) /
                  4;
      }
    }
    setBounds(numRows, numColumns, p, 0);
  }
  for (int i = 1; i < numRows - 1; ++i) {
    for (int j = 1; j < numColumns - 1; ++j) {
      velocityGridX[i][j] -= 0.5 * (p[i + 1][j] - p[i - 1][j]);
      velocityGridY[i][j] -= 0.5 * (p[i][j + 1] - p[i][j - 1]);
    }
  }
  setBounds(numRows, numColumns, velocityGridX, 1);
  setBounds(numRows, numColumns, velocityGridY, 2);
}

void stepDensity(int numRows, int numColumns,
                 vector<vector<float>> &densityGrid,
                 vector<vector<float>> &densityGridOld,
                 const vector<vector<float>> &velocityGridX,
                 const vector<vector<float>> &velocityGridY,
                 int diffusionFactor, int gaussSeidelIterations,
                 float timeStep) {
  diffuse(numRows, numColumns, densityGridOld, densityGrid,
          gaussSeidelIterations, diffusionFactor, 0, timeStep);
  advect(numRows, numColumns, densityGrid, densityGridOld, velocityGridX,
         velocityGridY, 0, timeStep);
}

void stepVelocity(int numRows, int numColumns,
                  vector<vector<float>> &velocityGridX,
                  vector<vector<float>> &velocityGridY,
                  vector<vector<float>> &velocityGridXOld,
                  vector<vector<float>> &velocityGridYOld, int viscosityFactor,
                  int gaussSeidelIterations, float timeStep) {
  diffuse(numRows, numColumns, velocityGridXOld, velocityGridX,
          gaussSeidelIterations, viscosityFactor, 1, timeStep);
  diffuse(numRows, numColumns, velocityGridYOld, velocityGridY,
          gaussSeidelIterations, viscosityFactor, 2, timeStep);
  project(numRows, numColumns, velocityGridXOld, velocityGridYOld,
          velocityGridX, velocityGridY, 20);
  advect(numRows, numColumns, velocityGridX, velocityGridXOld, velocityGridXOld,
         velocityGridYOld, 1, timeStep);
  advect(numRows, numColumns, velocityGridY, velocityGridYOld, velocityGridXOld,
         velocityGridYOld, 2, timeStep);
  project(numRows, numColumns, velocityGridX, velocityGridY, velocityGridXOld,
          velocityGridYOld, 20);
}

class ParticleApp : public App {

public:
  int numRows;
  int numColumns;
  int gridResolution;
  bool simulationPaused;
  vec2 lastMousePositon;
  float timeStep;
  float diffusionFactor;
  float viscosityFactor;
  FluidGrid fluidGrid;
  FluidGrid fluidGridOld;

  ParticleApp()
      : numRows(DEFAULT_WINDOW_HEIGHT / DEFAULT_GRID_RESOLUTION),
        numColumns(DEFAULT_WINDOW_WIDTH / DEFAULT_GRID_RESOLUTION),
        gridResolution(DEFAULT_GRID_RESOLUTION), simulationPaused(false),
        timeStep(DEFAULT_TIMESTEP), diffusionFactor(DEFAULT_DIFFUSION_FACTOR),
        viscosityFactor(DEFAULT_VISCOSITY_FACTOR),
        fluidGrid(numRows, numColumns, gridResolution, gridResolution),
        fluidGridOld(numRows, numColumns, gridResolution, gridResolution) {}

  void setup() override {
    setWindowSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
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
    } else if (event.getChar() == 'r' || event.getChar() == 'R') {
      fluidGrid.reset();
      fluidGridOld.reset();
    }
  }

  void onMouseDown(MouseEvent event) { lastMousePositon = event.getPos(); }

  void onMouseDrag(MouseEvent event) {
    vec2 currentMousePosition = event.getPos();
    vec2 dragDirection = currentMousePosition - lastMousePositon;
    fluidGrid.increaseCellDensity(currentMousePosition.y / gridResolution,
                                  currentMousePosition.x / gridResolution,
                                  5.0f);
    fluidGrid.increaseCellVelocity(currentMousePosition.y / gridResolution,
                                   currentMousePosition.x / gridResolution,
                                   dragDirection.x, dragDirection.y);
  }

  void onMouseUp(MouseEvent event) { lastMousePositon = vec2(0, 0); }

  void update() override {
    ImGui::Begin("Parameters");
    if (ImGui::Button("Clear")) {
      fluidGrid.reset();
      fluidGridOld.reset();
    }
    ImGui::Checkbox("Pause", &simulationPaused);
    ImGui::SliderFloat("Time Step", &timeStep, 0.0001f, 0.1f);
    ImGui::SliderFloat("Diffusion Factor", &diffusionFactor, 0.0f, 10.0f);
    ImGui::SliderFloat("Viscosity Factor", &viscosityFactor, 0.0f, 10.0f);
    ImGui::End();

    if (!simulationPaused) {
      fluidGridOld.reset();

      auto &densityGridOld = fluidGridOld.densityGrid;
      auto &densityGrid = fluidGrid.densityGrid;
      auto &velocityGridXOld = fluidGridOld.velocityGridX;
      auto &velocityGridX = fluidGrid.velocityGridX;
      auto &velocityGridYOld = fluidGridOld.velocityGridY;
      auto &velocityGridY = fluidGrid.velocityGridY;

      stepDensity(numRows, numColumns, densityGrid, densityGridOld,
                  velocityGridX, velocityGridY, diffusionFactor, 50, timeStep);

      stepVelocity(numRows, numColumns, velocityGridX, velocityGridY,
                   velocityGridXOld, velocityGridYOld, viscosityFactor, 50,
                   timeStep);
    }
  }

  void resize() override {
    numRows = getWindowHeight() / gridResolution;
    numColumns = getWindowWidth() / gridResolution;
    fluidGrid = FluidGrid(numRows, numColumns, gridResolution, gridResolution);
    fluidGridOld =
        FluidGrid(numRows, numColumns, gridResolution, gridResolution);
  }

  void draw() override {
    gl::clear(Color(0, 0, 0));
    fluidGrid.draw();
  }
};

void prepareSettings(ParticleApp::Settings *settings) {
  settings->setResizable(true);
}

CINDER_APP(ParticleApp, RendererGl, prepareSettings)