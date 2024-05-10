#include "FluidGrid.hpp"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include <vector>

#define GAUSS_SEIDEL_ITERATIONS 20
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080
#define GRID_RESOLUTION 10
#define DIFFUSION_FACTOR 0.001f
#define VISCOSITY_FACTOR 0.001f
#define SOURCE_VALUE 50.0f
#define TIMESTEP 0.01f
#define BG_COLOR ci::Color(0.8f, 0.8f, 0.8f)
#define FLUID_COLOR ci::Color(0.0f, 0.0f, 0.0f)

class FluidApp : public ci::app::App {
public:
  int numRows = WINDOW_HEIGHT / GRID_RESOLUTION;
  int numColumns = WINDOW_WIDTH / GRID_RESOLUTION;
  int gridResolution = GRID_RESOLUTION;
  FluidGrid fluidGrid = FluidGrid(numRows, numColumns, gridResolution);
  ci::gl::VboMeshRef fluidMesh;

  float timeStep = TIMESTEP;
  float diffusionFactor = DIFFUSION_FACTOR;
  float viscosityFactor = VISCOSITY_FACTOR;

  bool simulationPaused = false;
  ci::vec2 lastMousePositon = ci::vec2(0.0f, 0.0f);
  int sourceX = WINDOW_WIDTH / 2;
  int sourceY = WINDOW_HEIGHT / 2;
  std::vector<ci::vec2> sourcesXY;

  void initMesh();
  void updateMesh();
  void setConstantFluidSource(int sourceX, int sourceY);

  void setup() override;
  void keyDown(ci::app::KeyEvent event) override;
  void onMouseDown(ci::app::MouseEvent event);
  void onMouseDrag(ci::app::MouseEvent event);
  void onMouseUp(ci::app::MouseEvent event);
  void update() override;
  void draw() override;
};

void FluidApp::initMesh() {
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

  fluidMesh = ci::gl::VboMesh::create(positions.size(), GL_TRIANGLES,
                                      {ci::gl::VboMesh::Layout()
                                           .attrib(ci::geom::POSITION, 2)
                                           .attrib(ci::geom::COLOR, 4)});
  fluidMesh->bufferAttrib(ci::geom::POSITION, positions);
  fluidMesh->bufferAttrib(ci::geom::COLOR, colors);
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
  fluidMesh->bufferAttrib(ci::geom::COLOR, colors);
}

void FluidApp::setConstantFluidSource(int sourceX, int sourceY) {
  const std::vector<std::pair<int, int>> offsets = {{0, 0},  {-1, 0}, {1, 0},
                                                    {0, -1}, {0, 1},  {-1, -1},
                                                    {-1, 1}, {1, -1}, {1, 1}};
  int i = std::min(sourceY / gridResolution, numRows - 1);
  int j = std::min(sourceX / gridResolution, numColumns - 1);
  for (auto &offset : offsets) {
    int new_i = std::clamp(i + offset.first, 0, numRows - 1);
    int new_j = std::clamp(j + offset.second, 0, numColumns - 1);
    fluidGrid.densitySourceGrid[new_i][new_j] = SOURCE_VALUE;
    fluidGrid.velocitySourceGridX[new_i][new_j] = SOURCE_VALUE;
    fluidGrid.velocitySourceGridY[new_i][new_j] = SOURCE_VALUE;
  }
}

void FluidApp::setup() {
  initMesh();
  ImGui::Initialize();
  getWindow()->getSignalMouseDown().connect(
      [this](ci::app::MouseEvent event) { onMouseDown(event); });
  getWindow()->getSignalMouseDrag().connect(
      [this](ci::app::MouseEvent event) { onMouseDrag(event); });
  getWindow()->getSignalMouseUp().connect(
      [this](ci::app::MouseEvent event) { onMouseUp(event); });
}

void FluidApp::keyDown(ci::app::KeyEvent event) {
  if (event.getChar() == 'q' || event.getChar() == 'Q') {
    quit();
  } else if (event.getChar() == 'r' || event.getChar() == 'R') {
    fluidGrid.reset();
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
    fluidGrid.densitySourceGrid[i][j] += SOURCE_VALUE;
    fluidGrid.velocitySourceGridX[i][j] += dragDirection.x;
    fluidGrid.velocitySourceGridY[i][j] += dragDirection.y;
  }
}

void FluidApp::onMouseUp(ci::app::MouseEvent event) {
  lastMousePositon = ci::vec2(0, 0);
}

void FluidApp::update() {
  ImGui::Begin("Parameters");
  if (ImGui::Button("Reset")) {
    fluidGrid.reset();
  }
  ImGui::Checkbox("Pause", &simulationPaused);
  ImGui::SliderFloat("Timestep", &timeStep, 0.1f, 0.5f);
  ImGui::SliderFloat("Diffusion factor", &diffusionFactor, 0.0f, 10.0f);
  ImGui::SliderFloat("Viscosity factor", &viscosityFactor, 0.0f, 10.0f);
  if (ImGui::BeginMenu("Add source")) {
    ImGui::InputInt("X", &sourceX);
    ImGui::InputInt("Y", &sourceY);
    if (ImGui::Button("Add source")) {
      sourcesXY.push_back(ci::vec2(sourceX, sourceY));
    }
    if (ImGui::Button("Pop source")) {
      if (sourcesXY.size() > 0) {
        sourcesXY.pop_back();
      }
    }
    ImGui::EndMenu();
  }
  ImGui::End();

  if (!simulationPaused) {
    for (int k = 0; k < sourcesXY.size(); ++k) {
      setConstantFluidSource(sourcesXY[k].x, sourcesXY[k].y);
    }
    fluidGrid.stepDensity(diffusionFactor, GAUSS_SEIDEL_ITERATIONS, timeStep);
    fluidGrid.stepVelocity(viscosityFactor, GAUSS_SEIDEL_ITERATIONS, timeStep);
    updateMesh();
  }
}

void FluidApp::draw() {
  ci::gl::clear(BG_COLOR);
  ci::gl::draw(fluidMesh);
}

void prepareSettings(FluidApp::Settings *settings) {
  settings->setResizable(false);
  settings->setWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
}

CINDER_APP(FluidApp, ci::app::RendererGl, prepareSettings)
