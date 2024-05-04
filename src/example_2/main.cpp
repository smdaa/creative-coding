#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <omp.h>
#include <vector>

using namespace std;
using namespace ci;
using namespace ci::app;

#define GAUSS_SEIDEL_ITERATIONS 20
#define DEFAULT_WINDOW_WIDTH 1280
#define DEFAULT_WINDOW_HEIGHT 720
#define DEFAULT_GRID_RESOLUTION 10
#define DEFAULT_DIFFUSION_FACTOR 0.0001f
#define DEFAULT_VISCOSITY_FACTOR 0.0001f
#define DEFAULT_SOURCE_VALUE 50.0f
#define DEFAULT_TIMESTEP 0.1f

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
  float denominator = 1 + 4 * a;
  for (int k = 0; k < gaussSeidelIterations; ++k) {
#pragma omp parallel for
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numColumns; ++j) {
        float sum = 0.0f;
        sum += (i > 0) ? inGrid[i - 1][j] : 0.0f;              // Left
        sum += (i < numRows - 1) ? inGrid[i + 1][j] : 0.0f;    // Right
        sum += (j > 0) ? inGrid[i][j - 1] : 0.0f;              // Up
        sum += (j < numColumns - 1) ? inGrid[i][j + 1] : 0.0f; // Down
        outGrid[i][j] = (inGrid[i][j] + a * sum) / (denominator);
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
      outGrid[i][j] = sx0 * (sy0 * inGrid[x0][y0] + sy1 * inGrid[x0][y1]) +
                      sx1 * (sy0 * inGrid[x1][y0] + sy1 * inGrid[x1][y1]);
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
  int gridResolution;
  vector<vector<float>> densityGrid;
  vector<vector<float>> densityGridOld;
  vector<vector<float>> velocityGridX;
  vector<vector<float>> velocityGridXOld;
  vector<vector<float>> velocityGridY;
  vector<vector<float>> velocityGridYOld;
  vector<vector<float>> densitySourceGrid;
  vector<vector<float>> velocitySourceGridX;
  vector<vector<float>> velocitySourceGridY;
  gl::VboMeshRef mesh;

  FluidGrid(int _numRows, int _numColumns, int _gridResolution);

  void stepDensity(int diffusionFactor, int gaussSeidelIterations,
                   float timeStep);
  void stepVelocity(int viscosityFactor, int gaussSeidelIterations,
                    float timeStep);

  void updateMesh();
};

FluidGrid::FluidGrid(int _numRows, int _numColumns, int _gridResolution)
    : numRows(_numRows), numColumns(_numColumns),
      gridResolution(_gridResolution) {
  densityGrid = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  densityGridOld = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocityGridX = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocityGridXOld = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocityGridY = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocityGridYOld = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  densitySourceGrid = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocitySourceGridX = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocitySourceGridY = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));

  vector<vec2> positions;
  vector<ColorA> colors;
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numColumns; ++j) {
      float x0 = j * gridResolution;
      float y0 = i * gridResolution;
      float x1 = (j + 1) * gridResolution;
      float y1 = (i + 1) * gridResolution;

      positions.push_back(vec2(x0, y0));
      positions.push_back(vec2(x1, y0));
      positions.push_back(vec2(x0, y1));

      positions.push_back(vec2(x1, y0));
      positions.push_back(vec2(x1, y1));
      positions.push_back(vec2(x0, y1));

      for (int k = 0; k < 6; ++k) {
        colors.push_back(ColorA(1.0f, 1.0f, 1.0f, densityGrid[i][j]));
      }
    }
  }

  mesh = gl::VboMesh::create(
      positions.size(), GL_TRIANGLES,
      {gl::VboMesh::Layout().attrib(geom::POSITION, 2).attrib(geom::COLOR, 4)});
  mesh->bufferAttrib(geom::POSITION, positions);
  mesh->bufferAttrib(geom::COLOR, colors);
}

void FluidGrid::stepDensity(int diffusionFactor, int gaussSeidelIterations,
                            float timeStep) {
  addSource(numRows, numColumns, densityGrid, densitySourceGrid, timeStep);
  diffuse(numRows, numColumns, densityGridOld, densityGrid,
          gaussSeidelIterations, diffusionFactor, 0, timeStep);
  advect(numRows, numColumns, densityGrid, densityGridOld, velocityGridX,
         velocityGridY, 0, timeStep);
  densitySourceGrid = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
}

void FluidGrid::stepVelocity(int viscosityFactor, int gaussSeidelIterations,
                             float timeStep) {
  addSource(numRows, numColumns, velocityGridX, velocitySourceGridX, timeStep);
  addSource(numRows, numColumns, velocityGridY, velocitySourceGridY, timeStep);
  diffuse(numRows, numColumns, velocityGridXOld, velocityGridX,
          gaussSeidelIterations, viscosityFactor, 1, timeStep);
  diffuse(numRows, numColumns, velocityGridYOld, velocityGridY,
          gaussSeidelIterations, viscosityFactor, 2, timeStep);
  project(numRows, numColumns, velocityGridXOld, velocityGridYOld,
          velocityGridX, velocityGridY, gaussSeidelIterations);
  advect(numRows, numColumns, velocityGridX, velocityGridXOld, velocityGridXOld,
         velocityGridYOld, 1, timeStep);
  advect(numRows, numColumns, velocityGridY, velocityGridYOld, velocityGridXOld,
         velocityGridYOld, 2, timeStep);
  project(numRows, numColumns, velocityGridX, velocityGridY, velocityGridXOld,
          velocityGridYOld, gaussSeidelIterations);
  velocitySourceGridX = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocitySourceGridY = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
}

void FluidGrid::updateMesh() {
  vector<ColorA> colors;
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numColumns; ++j) {
      for (int k = 0; k < 6; ++k) {
        colors.push_back(ColorA(1.0f, 1.0f, 1.0f, densityGrid[i][j]));
      }
    }
  }
  mesh->bufferAttrib(geom::COLOR, colors);
}

struct FluidSource {
  int x;
  int y;
  float density;
  float velocity;
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
  float sourceValue;
  int constantSourceX;
  int constantSourceY;
  float constantSourceDensity;
  float constantSourceVelocity;
  std::vector<FluidSource> fluidSources;
  FluidGrid fluidGrid;

  FluidApp();

  void setup() override;
  void keyDown(KeyEvent event) override;
  void onMouseDown(MouseEvent event);
  void onMouseDrag(MouseEvent event);
  void onMouseUp(MouseEvent event);
  void update() override;
  void resize() override;
  void draw() override;
};

FluidApp::FluidApp()
    : numRows(DEFAULT_WINDOW_HEIGHT / DEFAULT_GRID_RESOLUTION),
      numColumns(DEFAULT_WINDOW_WIDTH / DEFAULT_GRID_RESOLUTION),
      gridResolution(DEFAULT_GRID_RESOLUTION), simulationPaused(false),
      timeStep(DEFAULT_TIMESTEP), diffusionFactor(DEFAULT_DIFFUSION_FACTOR),
      viscosityFactor(DEFAULT_VISCOSITY_FACTOR),
      sourceValue(DEFAULT_SOURCE_VALUE),
      constantSourceX(DEFAULT_WINDOW_WIDTH / 2),
      constantSourceY(DEFAULT_WINDOW_HEIGHT / 2), constantSourceDensity(10.0f),
      constantSourceVelocity(10.0f),
      fluidGrid(numRows, numColumns, gridResolution) {}

void FluidApp::keyDown(KeyEvent event) {
  if (event.getChar() == 'q' || event.getChar() == 'Q') {
    quit();
  } else if (event.getChar() == 'r' || event.getChar() == 'R') {
    fluidGrid = FluidGrid(numRows, numColumns, gridResolution);
  } else if (event.getChar() == 'p' || event.getChar() == 'P') {
    simulationPaused = !simulationPaused;
  }
}

void FluidApp::onMouseDown(MouseEvent event) {
  lastMousePositon = event.getPos();
}

void FluidApp::onMouseDrag(MouseEvent event) {
  vec2 currentMousePosition = event.getPos();
  vec2 dragDirection = currentMousePosition - lastMousePositon;
  int i = currentMousePosition.y / gridResolution;
  int j = currentMousePosition.x / gridResolution;
  if (i >= 0 && i < numRows && j >= 0 && j < numColumns) {
    if (event.isLeftDown()) {
      fluidGrid.densitySourceGrid[i][j] += sourceValue;
      fluidGrid.velocitySourceGridX[i][j] += dragDirection.x;
      fluidGrid.velocitySourceGridY[i][j] += dragDirection.y;
    } else if (event.isRightDown()) {
      fluidGrid.densitySourceGrid[i][j] -= sourceValue;
      fluidGrid.velocitySourceGridX[i][j] -= dragDirection.x;
      fluidGrid.velocitySourceGridY[i][j] -= dragDirection.y;
    }
  }
}

void FluidApp::onMouseUp(MouseEvent event) { lastMousePositon = vec2(0, 0); }

void FluidApp::setup() {
  getWindow()->getSignalMouseDown().connect(
      [this](MouseEvent event) { onMouseDown(event); });
  getWindow()->getSignalMouseDrag().connect(
      [this](MouseEvent event) { onMouseDrag(event); });
  getWindow()->getSignalMouseUp().connect(
      [this](MouseEvent event) { onMouseUp(event); });
  ImGui::Initialize();
}

void FluidApp::update() {
  ImGui::Begin("Parameters");
  if (ImGui::Button("Reset")) {
    fluidGrid = FluidGrid(numRows, numColumns, gridResolution);
  }
  ImGui::Checkbox("Pause", &simulationPaused);
  ImGui::SliderFloat("Timestep", &timeStep, 0.1f, 0.5f);
  ImGui::SliderFloat("Diffusion factor", &diffusionFactor, 0.0f, 10.0f);
  ImGui::SliderFloat("Viscosity factor", &viscosityFactor, 0.0f, 10.0f);
  ImGui::SliderFloat("Source value", &sourceValue, 10.0f, 100.0f);

  if (ImGui::BeginMenu("Add source")) {
    ImGui::InputInt("X", &constantSourceX);
    ImGui::InputInt("Y", &constantSourceY);
    ImGui::InputFloat("Density source Value", &constantSourceDensity);
    ImGui::InputFloat("Velocity source Value", &constantSourceVelocity);
    if (ImGui::Button("Add")) {
      FluidSource newSource;
      newSource.x = constantSourceX;
      newSource.y = constantSourceY;
      newSource.density = constantSourceDensity;
      newSource.velocity = constantSourceVelocity;
      fluidSources.push_back(newSource);
    }
    if (ImGui::Button("Pop")) {
      if (fluidSources.size() > 0) {
        fluidSources.pop_back();
      }
    }

    ImGui::EndMenu();
  }
  ImGui::End();

  if (!simulationPaused) {

    for (int k = 0; k < fluidSources.size(); ++k) {
      int i = min(fluidSources[k].y / gridResolution, numRows - 1);
      int j = min(fluidSources[k].x / gridResolution, numColumns - 1);

      fluidGrid.densitySourceGrid[i][j] =  fluidSources[k].density;
      fluidGrid.densitySourceGrid[max(0, i - 1)][j] = fluidSources[k].density;
      fluidGrid.densitySourceGrid[min(numRows - 1, i + 1)][j] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[i][max(0, j - 1)] = fluidSources[k].density;
      fluidGrid.densitySourceGrid[i][min(numColumns - 1, j + 1)] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[max(0, i - 1)][max(0, j - 1)] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[max(0, i - 1)][min(numColumns - 1, j + 1)] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[min(numRows - 1, i + 1)][max(0, j - 1)] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[min(numRows - 1, i + 1)]
                                 [min(numColumns - 1, j + 1)] =
          fluidSources[k].density;

      fluidGrid.velocitySourceGridX[i][j] -=  fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[max(0, i - 1)][j] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[min(numRows - 1, i + 1)][j] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[i][max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[i][min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[max(0, i - 1)][max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid
          .velocitySourceGridX[max(0, i - 1)][min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[min(numRows - 1, i + 1)][max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[min(numRows - 1, i + 1)]
                                   [min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;

      fluidGrid.velocitySourceGridY[i][j] -=  fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[max(0, i - 1)][j] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[min(numRows - 1, i + 1)][j] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[i][max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[i][min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[max(0, i - 1)][max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid
          .velocitySourceGridY[max(0, i - 1)][min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[min(numRows - 1, i + 1)][max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[min(numRows - 1, i + 1)]
                                   [min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;
    }
    fluidGrid.stepDensity(diffusionFactor, GAUSS_SEIDEL_ITERATIONS, timeStep);
    fluidGrid.stepVelocity(viscosityFactor, GAUSS_SEIDEL_ITERATIONS, timeStep);
    fluidGrid.updateMesh();
  }
}

void FluidApp::resize() {
  numRows = getWindowHeight() / gridResolution;
  numColumns = getWindowWidth() / gridResolution;
  fluidGrid = FluidGrid(numRows, numColumns, gridResolution);
}

void FluidApp::draw() {
  gl::clear(Color(0.0f, 0.0f, 0.0f));
  gl::draw(fluidGrid.mesh);
}
void prepareSettings(FluidApp::Settings *settings) {
  settings->setResizable(true);
  settings->setMultiTouchEnabled(false);
  settings->setWindowSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
}

CINDER_APP(FluidApp, RendererGl(RendererGl::Options().msaa(4)), prepareSettings)
