#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

using namespace ci;
using namespace ci::app;

#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT 500
#define BG_COLOR ColorA(0.0f, 0.0f, 0.0f)
#define GRID_RESOLUTION 5

void addSource(std::vector<std::vector<float>> densityGrid, std::vector<std::vector<float>> sourceGrid)
{
  if (densityGrid.size() != sourceGrid.size())
  {
    std::cerr << "Error: Grid dimensions do not match." << std::endl;
    return;
  }

  for (size_t i = 0; i < densityGrid.size(); ++i)
  {
    if (densityGrid[i].size() != sourceGrid[i].size())
    {
      std::cerr << "Error: Row " << i << " dimensions do not match." << std::endl;
      return;
    }
  }

  for (size_t i = 0; i < densityGrid.size(); ++i)
  {
    for (size_t j = 0; j < densityGrid[i].size(); ++j)
    {
      densityGrid[i][j] += sourceGrid[i][j];
    }
  }
}

class SmokeApp : public App
{
public:
  void setup() override
  {
    setWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);

    gridNumRows = WINDOW_HEIGHT / GRID_RESOLUTION;
    gridNumCols = WINDOW_WIDTH / GRID_RESOLUTION;
    velocityGrid = std::vector<std::vector<vec2>>(
        gridNumRows, std::vector<vec2>(gridNumCols, vec2(0.0f, 0.0f)));
    velocityGridOld = std::vector<std::vector<vec2>>(
        gridNumRows, std::vector<vec2>(gridNumCols, vec2(0.0f, 0.0f)));
    densityGrid = std::vector<std::vector<float>>(gridNumRows, std::vector<float>(gridNumCols, 0.0f));
    densityGridOld = std::vector<std::vector<float>>(gridNumRows, std::vector<float>(gridNumCols, 0.0f));
  }

  void keyDown(KeyEvent event) override
  {
    if (event.getChar() == 'q' || event.getChar() == 'Q')
    {
      quit();
    }
  }

  void update() override {}

  void draw() override
  {
    gl::clear(BG_COLOR);
  }

private:
  int gridNumRows;
  int gridNumCols;
  std::vector<std::vector<vec2>> velocityGrid;
  std::vector<std::vector<vec2>> velocityGridOld;
  std::vector<std::vector<float>> densityGrid;
  std::vector<std::vector<float>> densityGridOld;
};

void prepareSettings(SmokeApp::Settings *settings)
{
  settings->setResizable(false);
}

CINDER_APP(SmokeApp, RendererGl, prepareSettings)
