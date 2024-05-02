#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <omp.h>
#include <vector>

using namespace std;
using namespace ci;
using namespace ci::app;

#define DEFAULT_WINDOW_WIDTH 1280
#define DEFAULT_WINDOW_HEIGHT 720
#define DEFAULT_GRID_RESOLUTION 20
#define DEFAULT_DIFFUSION_FACTOR 0.0001f
#define DEFAULT_VISCOSITY_FACTOR 0.0001f
#define DEFAULT_TIMESTEP 0.1f

const char *vertexShader = R"(
            #version 150

            uniform mat4 ciModelViewProjection;

            in vec2 ciPosition;
            out vec2 TexCoord;

            void main() {
                TexCoord = ciPosition;
                gl_Position = ciModelViewProjection * vec4(ciPosition, 0.0, 1.0);
            }
        )";

const char *fragmentShader = R"(
            #version 150

            in vec2 TexCoord;
            out vec4 FragColor;

            uniform sampler2D FluidGrid;
            uniform vec2 GridSize;

            void main() {
                vec2 texCoord = TexCoord * GridSize;
                float density = texture(FluidGrid, texCoord).r;
                FragColor = vec4(1.0, 1.0, 1.0, density);
            }
        )";

static void addSource(int numRows, int numColumns, vector<vector<float>> &grid,
                      const vector<vector<float>> &sourceGrid, float timeStep) {
#pragma omp parallel for
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numColumns; ++j) {
      grid[i][j] += sourceGrid[i][j] * timeStep;
    }
  }
}

static void setBounds(int numRows, int numColumns, vector<vector<float>> &grid,
                      int b) {
#pragma omp parallel for
  for (int i = 1; i < numRows - 1; ++i) {
    grid[i][0] = (b == 2) ? -grid[i][1] : grid[i][1];
    grid[i][numColumns - 1] =
        (b == 2) ? -grid[i][numColumns - 2] : grid[i][numColumns - 2];
  }
#pragma omp parallel for
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

static void diffuse(int numRows, int numColumns, vector<vector<float>> &outGrid,
                    const vector<vector<float>> &inGrid,
                    int gaussSeidelIterations, float factor, int b,
                    float timeStep) {
  float a = timeStep * factor * numRows * numColumns;
  for (int k = 0; k < gaussSeidelIterations; ++k) {
#pragma omp parallel for
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

static void advect(int numRows, int numColumns, vector<vector<float>> &outGrid,
                   const vector<vector<float>> &inGrid,
                   const vector<vector<float>> &velocityGridX,
                   const vector<vector<float>> &velocityGridY, int b,
                   float timeStep) {
  float dtRatio = timeStep * (max(numRows, numColumns) - 1);
#pragma omp parallel for
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numColumns; ++j) {
      float x = i - dtRatio * velocityGridX[i][j];
      float y = j - dtRatio * velocityGridY[i][j];
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
      float newValue = sx0 * (sy0 * inGrid[x0][y0] + sy1 * inGrid[x0][y1]) +
                       sx1 * (sy0 * inGrid[x1][y0] + sy1 * inGrid[x1][y1]);
      outGrid[i][j] = newValue;
    }
  }
  setBounds(numRows, numColumns, outGrid, b);
}

static void project(int numRows, int numColumns,
                    vector<vector<float>> &velocityGridX,
                    vector<vector<float>> &velocityGridY,
                    vector<vector<float>> &p, vector<vector<float>> &div,
                    int gaussSeidelIterations) {
#pragma omp parallel for
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
#pragma omp parallel for
    for (int i = 1; i < numRows - 1; ++i) {
      for (int j = 1; j < numColumns - 1; ++j) {
        p[i][j] = (div[i][j] + p[i - 1][j] + p[i + 1][j] + p[i][j - 1] +
                   p[i][j + 1]) /
                  4;
      }
    }
    setBounds(numRows, numColumns, p, 0);
  }
#pragma omp parallel for
  for (int i = 1; i < numRows - 1; ++i) {
    for (int j = 1; j < numColumns - 1; ++j) {
      velocityGridX[i][j] -= 0.5 * (p[i + 1][j] - p[i - 1][j]);
      velocityGridY[i][j] -= 0.5 * (p[i][j + 1] - p[i][j - 1]);
    }
  }
  setBounds(numRows, numColumns, velocityGridX, 1);
  setBounds(numRows, numColumns, velocityGridY, 2);
}

class FluidGrid {

public:
  int numRows;
  int numColumns;
  vector<vector<float>> densityGrid;
  vector<vector<float>> densityGridOld;
  vector<vector<float>> velocityGridX;
  vector<vector<float>> velocityGridXOld;
  vector<vector<float>> velocityGridY;
  vector<vector<float>> velocityGridYOld;
  vector<vector<float>> densitySourceGrid;
  vector<vector<float>> velocitySourceGridX;
  vector<vector<float>> velocitySourceGridY;

  FluidGrid(int _numRows, int _numColumns)
      : numRows(_numRows), numColumns(_numColumns) {
    densityGrid.resize(numRows);
    densityGridOld.resize(numRows);
    velocityGridX.resize(numRows);
    velocityGridXOld.resize(numRows);
    velocityGridY.resize(numRows);
    velocityGridYOld.resize(numRows);
    densitySourceGrid.resize(numRows);
    velocitySourceGridX.resize(numRows);
    velocitySourceGridY.resize(numRows);
    for (int i = 0; i < numRows; ++i) {
      densityGrid[i].resize(numColumns);
      densityGridOld[i].resize(numColumns);
      velocityGridX[i].resize(numColumns);
      velocityGridXOld[i].resize(numColumns);
      velocityGridY[i].resize(numColumns);
      velocityGridYOld[i].resize(numColumns);
      densitySourceGrid[i].resize(numColumns);
      velocitySourceGridX[i].resize(numColumns);
      velocitySourceGridY[i].resize(numColumns);
      for (int j = 0; j < numColumns; ++j) {
        densityGrid[i][j] = 0.0f;
        densityGridOld[i][j] = 0.0f;
        velocityGridX[i][j] = 0.0f;
        velocityGridXOld[i][j] = 0.0f;
        velocityGridY[i][j] = 0.0f;
        velocityGridYOld[i][j] = 0.0f;
        densitySourceGrid[i][j] = 0.0f;
        velocitySourceGridX[i][j] = 0.0f;
        velocitySourceGridY[i][j] = 0.0f;
      }
    }
  }

  void resetGrids() {
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numColumns; ++j) {
        densityGrid[i][j] = 0.0f;
        densityGridOld[i][j] = 0.0f;
        velocityGridX[i][j] = 0.0f;
        velocityGridXOld[i][j] = 0.0f;
        velocityGridY[i][j] = 0.0f;
        velocityGridYOld[i][j] = 0.0f;
        densitySourceGrid[i][j] = 0.0f;
        velocitySourceGridX[i][j] = 0.0f;
        velocitySourceGridY[i][j] = 0.0f;
      }
    }
  }

  void stepDensity(int diffusionFactor, int gaussSeidelIterations,
                   float timeStep) {
    addSource(numRows, numColumns, densityGrid, densitySourceGrid, timeStep);
    diffuse(numRows, numColumns, densityGridOld, densityGrid,
            gaussSeidelIterations, diffusionFactor, 0, timeStep);
    advect(numRows, numColumns, densityGrid, densityGridOld, velocityGridX,
           velocityGridY, 0, timeStep);
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numColumns; ++j) {
        densitySourceGrid[i][j] = 0.0f;
        densityGridOld[i][j] = 0.0f;
      }
    }
  }

  void stepVelocity(int viscosityFactor, int gaussSeidelIterations,
                    float timeStep) {
    addSource(numRows, numColumns, velocityGridX, velocitySourceGridX,
              timeStep);
    addSource(numRows, numColumns, velocityGridY, velocitySourceGridY,
              timeStep);
    diffuse(numRows, numColumns, velocityGridXOld, velocityGridX,
            gaussSeidelIterations, viscosityFactor, 1, timeStep);
    diffuse(numRows, numColumns, velocityGridYOld, velocityGridY,
            gaussSeidelIterations, viscosityFactor, 2, timeStep);
    project(numRows, numColumns, velocityGridXOld, velocityGridYOld,
            velocityGridX, velocityGridY, 20);
    advect(numRows, numColumns, velocityGridX, velocityGridXOld,
           velocityGridXOld, velocityGridYOld, 1, timeStep);
    advect(numRows, numColumns, velocityGridY, velocityGridYOld,
           velocityGridXOld, velocityGridYOld, 2, timeStep);
    project(numRows, numColumns, velocityGridX, velocityGridY, velocityGridXOld,
            velocityGridYOld, 20);
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numColumns; ++j) {
        velocitySourceGridX[i][j] = 0.0f;
        velocitySourceGridY[i][j] = 0.0f;
        velocityGridXOld[i][j] = 0.0f;
        velocityGridYOld[i][j] = 0.0f;
      }
    }
  }
};

class FluidApp : public App {

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
  Font mFont;

  FluidApp()
      : numRows(DEFAULT_WINDOW_HEIGHT / DEFAULT_GRID_RESOLUTION),
        numColumns(DEFAULT_WINDOW_WIDTH / DEFAULT_GRID_RESOLUTION),
        gridResolution(DEFAULT_GRID_RESOLUTION), simulationPaused(false),
        timeStep(DEFAULT_TIMESTEP), diffusionFactor(DEFAULT_DIFFUSION_FACTOR),
        viscosityFactor(DEFAULT_VISCOSITY_FACTOR),
        fluidGrid(numRows, numColumns) {}

  void setup() override {
    setWindowSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
    getWindow()->getSignalMouseDown().connect(
        [this](MouseEvent event) { onMouseDown(event); });
    getWindow()->getSignalMouseDrag().connect(
        [this](MouseEvent event) { onMouseDrag(event); });
    getWindow()->getSignalMouseUp().connect(
        [this](MouseEvent event) { onMouseUp(event); });
    ImGui::Initialize();
    mFont = Font("Arial", 12);
  }

  void keyDown(KeyEvent event) override {
    if (event.getChar() == 'q' || event.getChar() == 'Q') {
      quit();
    } else if (event.getChar() == 'r' || event.getChar() == 'R') {
      fluidGrid.resetGrids();
    } else if (event.getChar() == 'p' || event.getChar() == 'P') {
      simulationPaused = !simulationPaused;
    }
  }

  void onMouseDown(MouseEvent event) { lastMousePositon = event.getPos(); }

  void onMouseDrag(MouseEvent event) {
    vec2 currentMousePosition = event.getPos();
    vec2 dragDirection = currentMousePosition - lastMousePositon;
    auto &densitySourceGrid = fluidGrid.densitySourceGrid;
    auto &velocitySourceGridX = fluidGrid.velocitySourceGridX;
    auto &velocitySourceGridY = fluidGrid.velocitySourceGridY;
    int i = currentMousePosition.y / gridResolution;
    int j = currentMousePosition.x / gridResolution;
    if (i >= 0 && i < numRows && j >= 0 && j < numColumns) {
      densitySourceGrid[i][j] += 50.0f * timeStep;
      velocitySourceGridX[i][j] += dragDirection.x * timeStep;
      velocitySourceGridY[i][j] += dragDirection.y * timeStep;
    }
  }

  void onMouseUp(MouseEvent event) { lastMousePositon = vec2(0, 0); }

  void update() override {
    ImGui::Begin("Parameters");
    if (ImGui::Button("Clear")) {
      fluidGrid.resetGrids();
    }
    ImGui::Checkbox("Pause", &simulationPaused);
    ImGui::SliderFloat("Time Step", &timeStep, 0.1f, 0.5f);
    ImGui::SliderFloat("Diffusion Factor", &diffusionFactor, 0.0f, 10.0f);
    ImGui::SliderFloat("Viscosity Factor", &viscosityFactor, 0.0f, 10.0f);
    ImGui::End();

    if (!simulationPaused) {
      fluidGrid.stepDensity(diffusionFactor, 50, timeStep);
      fluidGrid.stepVelocity(viscosityFactor, 50, timeStep);
    }
  }

  void resize() override {
    numRows = getWindowHeight() / gridResolution;
    numColumns = getWindowWidth() / gridResolution;
    fluidGrid = FluidGrid(numRows, numColumns);
  }

  void draw() override {
    gl::clear(Color(0, 0, 0));
    float cellWidth = getWindowWidth() / static_cast<float>(numColumns);
    float cellHeight = getWindowHeight() / static_cast<float>(numRows);

    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numColumns; ++j) {
        float density = fluidGrid.densityGrid[i][j];
        ColorA color(1.0f, 1.0f, 1.0f, density);
        Rectf rect(j * cellWidth, i * cellHeight, (j + 1) * cellWidth,
                   (i + 1) * cellHeight);
        gl::color(color);
        gl::drawSolidRect(rect);
      }
    }
    gl::color(Color::white());
    gl::drawString("FPS: " + to_string(getAverageFps()), vec2(10, 10),
                   Color::white(), mFont);
  }
};

void prepareSettings(FluidApp::Settings *settings) {
  settings->setResizable(true);
}

CINDER_APP(FluidApp, RendererGl, prepareSettings)
