#include "FluidGrid.hpp"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <vector>

#define GAUSS_SEIDEL_ITERATIONS 20
#define DEFAULT_WINDOW_WIDTH 1280
#define DEFAULT_WINDOW_HEIGHT 720
#define DEFAULT_GRID_RESOLUTION 5
#define DEFAULT_DIFFUSION_FACTOR 0.0001f
#define DEFAULT_VISCOSITY_FACTOR 0.0001f
#define DEFAULT_SOURCE_VALUE 50.0f
#define DEFAULT_TIMESTEP 0.05f

struct FluidSource {
  int x;
  int y;
  float density;
  float velocity;
};

class FluidApp : public ci::app::App {

public:
  int numRows;
  int numColumns;
  int gridResolution;
  bool simulationPaused;
  ci::vec2 lastMousePositon;
  float timeStep;
  float diffusionFactor;
  float viscosityFactor;
  float sourceValue;
  int sourceX;
  int sourceY;
  float sourceDensity;
  float sourceVelocity;
  std::vector<FluidSource> fluidSources;
  FluidGrid fluidGrid;

  FluidApp();

  void setup() override;
  void keyDown(ci::app::KeyEvent event) override;
  void onMouseDown(ci::app::MouseEvent event);
  void onMouseDrag(ci::app::MouseEvent event);
  void onMouseUp(ci::app::MouseEvent event);
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
      sourceValue(DEFAULT_SOURCE_VALUE), sourceX(DEFAULT_WINDOW_WIDTH / 2),
      sourceY(DEFAULT_WINDOW_HEIGHT / 2), sourceDensity(10.0f),
      sourceVelocity(10.0f), fluidGrid(numRows, numColumns, gridResolution) {}

void FluidApp::keyDown(ci::app::KeyEvent event) {
  if (event.getChar() == 'q' || event.getChar() == 'Q') {
    quit();
  } else if (event.getChar() == 'r' || event.getChar() == 'R') {
    fluidGrid = FluidGrid(numRows, numColumns, gridResolution);
  } else if (event.getChar() == 'p' || event.getChar() == 'P') {
    simulationPaused = !simulationPaused;
  }
}

void FluidApp::onMouseDown(ci::app::MouseEvent event) {
  lastMousePositon = event.getPos();
}

void FluidApp::onMouseDrag(ci::app::MouseEvent event) {
  ci::vec2 currentMousePosition = event.getPos();
  ci::vec2 dragDirection = currentMousePosition - lastMousePositon;
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

void FluidApp::onMouseUp(ci::app::MouseEvent event) {
  lastMousePositon = ci::vec2(0, 0);
}

void FluidApp::setup() {
  getWindow()->getSignalMouseDown().connect(
      [this](ci::app::MouseEvent event) { onMouseDown(event); });
  getWindow()->getSignalMouseDrag().connect(
      [this](ci::app::MouseEvent event) { onMouseDrag(event); });
  getWindow()->getSignalMouseUp().connect(
      [this](ci::app::MouseEvent event) { onMouseUp(event); });
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
    ImGui::InputInt("X", &sourceX);
    ImGui::InputInt("Y", &sourceY);
    ImGui::InputFloat("Density source Value", &sourceDensity);
    ImGui::InputFloat("Velocity source Value", &sourceVelocity);
    if (ImGui::Button("Add source")) {
      FluidSource newSource;
      newSource.x = sourceX;
      newSource.y = sourceY;
      newSource.density = sourceDensity;
      newSource.velocity = sourceVelocity;
      fluidSources.push_back(newSource);
    }
    if (ImGui::Button("Pop source")) {
      if (fluidSources.size() > 0) {
        fluidSources.pop_back();
      }
    }

    ImGui::EndMenu();
  }
  ImGui::End();

  if (!simulationPaused) {

    for (int k = 0; k < fluidSources.size(); ++k) {
      int i = std::min(fluidSources[k].y / gridResolution, numRows - 1);
      int j = std::min(fluidSources[k].x / gridResolution, numColumns - 1);

      fluidGrid.densitySourceGrid[i][j] = fluidSources[k].density;
      fluidGrid.densitySourceGrid[std::max(0, i - 1)][j] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[std::min(numRows - 1, i + 1)][j] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[i][std::max(0, j - 1)] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[i][std::min(numColumns - 1, j + 1)] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[std::max(0, i - 1)][std::max(0, j - 1)] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[std::max(0, i - 1)]
                                 [std::min(numColumns - 1, j + 1)] =
          fluidSources[k].density;
      fluidGrid
          .densitySourceGrid[std::min(numRows - 1, i + 1)][std::max(0, j - 1)] =
          fluidSources[k].density;
      fluidGrid.densitySourceGrid[std::min(numRows - 1, i + 1)]
                                 [std::min(numColumns - 1, j + 1)] =
          fluidSources[k].density;

      fluidGrid.velocitySourceGridX[i][j] -= fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[std::max(0, i - 1)][j] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[std::min(numRows - 1, i + 1)][j] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[i][std::max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[i][std::min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[std::max(0, i - 1)][std::max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[std::max(0, i - 1)]
                                   [std::min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[std::min(numRows - 1, i + 1)]
                                   [std::max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridX[std::min(numRows - 1, i + 1)]
                                   [std::min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;

      fluidGrid.velocitySourceGridY[i][j] -= fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[std::max(0, i - 1)][j] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[std::min(numRows - 1, i + 1)][j] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[i][std::max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[i][std::min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[std::max(0, i - 1)][std::max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[std::max(0, i - 1)]
                                   [std::min(numColumns - 1, j + 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[std::min(numRows - 1, i + 1)]
                                   [std::max(0, j - 1)] +=
          fluidSources[k].velocity;
      fluidGrid.velocitySourceGridY[std::min(numRows - 1, i + 1)]
                                   [std::min(numColumns - 1, j + 1)] +=
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
  ci::gl::clear(ci::Color(0.0f, 0.0f, 0.0f));
  ci::gl::draw(fluidGrid.mesh);
}
void prepareSettings(FluidApp::Settings *settings) {
  settings->setResizable(true);
  settings->setMultiTouchEnabled(false);
  settings->setWindowSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
}

CINDER_APP(FluidApp, ci::app::RendererGl, prepareSettings)
