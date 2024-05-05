#include "FluidGrid.hpp"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <vector>

#define GAUSS_SEIDEL_ITERATIONS 30
#define DEFAULT_WINDOW_WIDTH 1280
#define DEFAULT_WINDOW_HEIGHT 720
#define DEFAULT_GRID_RESOLUTION 10
#define DEFAULT_DIFFUSION_FACTOR 0.0001f
#define DEFAULT_VISCOSITY_FACTOR 0.0001f
#define DEFAULT_SOURCE_VALUE 50.0f
#define DEFAULT_TIMESTEP 0.05f
#define BG_COLOR ci::Color(0.8f, 0.8f, 0.8f)
#define FLUID_COLOR ci::Color(0.0f, 0.0f, 0.0f)

struct FluidSource {
  int x;
  int y;
  float density;
  float velocity;
};

class FluidApp : public ci::app::App {
public:
  int numRows = DEFAULT_WINDOW_HEIGHT / DEFAULT_GRID_RESOLUTION;
  int numColumns = DEFAULT_WINDOW_WIDTH / DEFAULT_GRID_RESOLUTION;
  int gridResolution = DEFAULT_GRID_RESOLUTION;
  FluidGrid fluidGrid = FluidGrid(numRows, numColumns, gridResolution);
  ci::gl::VboMeshRef mesh;

  float timeStep = DEFAULT_TIMESTEP;
  float diffusionFactor = DEFAULT_DIFFUSION_FACTOR;
  float viscosityFactor = DEFAULT_VISCOSITY_FACTOR;

  bool simulationPaused = false;
  ci::vec2 lastMousePositon = ci::vec2(0.0f, 0.0f);
  float sourceValue = DEFAULT_SOURCE_VALUE;
  int sourceX = DEFAULT_WINDOW_WIDTH / 2;
  int sourceY = DEFAULT_WINDOW_HEIGHT / 2;
  float sourceDensity = DEFAULT_SOURCE_VALUE;
  float sourceVelocity = DEFAULT_SOURCE_VALUE;
  std::vector<FluidSource> fluidSources;

  void initMesh(int numRows, int numColumns, int gridResolution);
  void updateMesh();
  void setConstantFluidSource(const FluidSource &source);

  void setup() override;
  void keyDown(ci::app::KeyEvent event) override;
  void onMouseDown(ci::app::MouseEvent event);
  void onMouseDrag(ci::app::MouseEvent event);
  void onMouseUp(ci::app::MouseEvent event);
  void update() override;
  void resize() override;
  void draw() override;
};

void FluidApp::initMesh(int numRows, int numColumns, int gridResolution) {
  std::vector<ci::vec2> positions;
  std::vector<ci::ColorA> colors;
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numColumns; ++j) {
      float x0 = j * gridResolution;
      float y0 = i * gridResolution;
      float x1 = (j + 1) * gridResolution;
      float y1 = (i + 1) * gridResolution;

      positions.push_back(ci::vec2(x0, y0));
      positions.push_back(ci::vec2(x1, y0));
      positions.push_back(ci::vec2(x0, y1));

      positions.push_back(ci::vec2(x1, y0));
      positions.push_back(ci::vec2(x1, y1));
      positions.push_back(ci::vec2(x0, y1));

      for (int k = 0; k < 6; ++k) {
        colors.push_back(
            ci::ColorA(1.0f, 1.0f, 1.0f, fluidGrid.densityGrid[i][j]));
      }
    }
  }

  mesh = ci::gl::VboMesh::create(positions.size(), GL_TRIANGLES,
                                 {ci::gl::VboMesh::Layout()
                                      .attrib(ci::geom::POSITION, 2)
                                      .attrib(ci::geom::COLOR, 4)});
  mesh->bufferAttrib(ci::geom::POSITION, positions);
  mesh->bufferAttrib(ci::geom::COLOR, colors);
}

void FluidApp::updateMesh() {
  std::vector<ci::ColorA> colors;
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numColumns; ++j) {
      float density = fluidGrid.densityGrid[i][j];
      for (int k = 0; k < 6; ++k) {
        colors.push_back(ci::ColorA(FLUID_COLOR, density));
      }
    }
  }
  mesh->bufferAttrib(ci::geom::COLOR, colors);
}

void FluidApp::setConstantFluidSource(const FluidSource &source) {
  const std::vector<std::pair<int, int>> offsets = {{0, 0},  {-1, 0}, {1, 0},
                                                    {0, -1}, {0, 1},  {-1, -1},
                                                    {-1, 1}, {1, -1}, {1, 1}};
  int i = std::min(source.y / gridResolution, numRows - 1);
  int j = std::min(source.x / gridResolution, numColumns - 1);
  for (auto &offset : offsets) {
    int new_i = std::clamp(i + offset.first, 0, numRows - 1);
    int new_j = std::clamp(j + offset.second, 0, numColumns - 1);
    fluidGrid.densitySourceGrid[new_i][new_j] = source.density;
    fluidGrid.velocitySourceGridX[new_i][new_j] = source.velocity;
    fluidGrid.velocitySourceGridY[new_i][new_j] = source.velocity;
  }
}

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
    fluidGrid.densitySourceGrid[i][j] += sourceValue;
    fluidGrid.velocitySourceGridX[i][j] += dragDirection.x;
    fluidGrid.velocitySourceGridY[i][j] += dragDirection.y;
  }
}

void FluidApp::onMouseUp(ci::app::MouseEvent event) {
  lastMousePositon = ci::vec2(0, 0);
}

void FluidApp::setup() {
  initMesh(numRows, numColumns, gridResolution);
  ImGui::Initialize();
  getWindow()->getSignalMouseDown().connect(
      [this](ci::app::MouseEvent event) { onMouseDown(event); });
  getWindow()->getSignalMouseDrag().connect(
      [this](ci::app::MouseEvent event) { onMouseDrag(event); });
  getWindow()->getSignalMouseUp().connect(
      [this](ci::app::MouseEvent event) { onMouseUp(event); });
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
      fluidSources.push_back(
          FluidSource{sourceX, sourceY, sourceDensity, sourceVelocity});
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
      setConstantFluidSource(fluidSources[k]);
    }

    fluidGrid.stepDensity(diffusionFactor, GAUSS_SEIDEL_ITERATIONS, timeStep);
    fluidGrid.stepVelocity(viscosityFactor, GAUSS_SEIDEL_ITERATIONS, timeStep);
  }
}

void FluidApp::resize() {
  numRows = getWindowHeight() / gridResolution;
  numColumns = getWindowWidth() / gridResolution;
  fluidGrid = FluidGrid(numRows, numColumns, gridResolution);
  sourceX = getWindowWidth() / 2;
  sourceY = getWindowHeight() / 2;
  fluidSources.clear();
  initMesh(numRows, numColumns, gridResolution);
}

void FluidApp::draw() {
  updateMesh();
  ci::gl::clear(BG_COLOR);
  ci::gl::draw(mesh);
}

void prepareSettings(FluidApp::Settings *settings) {
  settings->setResizable(true);
  settings->setMultiTouchEnabled(false);
  settings->setWindowSize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
}

CINDER_APP(FluidApp, ci::app::RendererGl, prepareSettings)
