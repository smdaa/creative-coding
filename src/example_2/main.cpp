#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

using namespace ci;
using namespace ci::app;

#define WINDOW_WIDTH 500
#define WINDOW_HEIGHT 500
#define BG_COLOR ColorA(0.0f, 0.0f, 0.0f)
#define GRID_RESOLUTION 5
#define DT 0.2f

class SmokeApp : public App
{
public:
  void setup() override
  {
    setWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    initVelocityGrid();
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

  void initVelocityGrid()
  {
    gridNumRows = WINDOW_HEIGHT / GRID_RESOLUTION;
    gridNumCols = WINDOW_WIDTH / GRID_RESOLUTION;
    velocityGrid = std::vector<std::vector<vec2>>(
        gridNumRows, std::vector<vec2>(gridNumCols, vec2(0.0f, 0.0f)));
  }
};
void prepareSettings(SmokeApp::Settings *settings)
{
  settings->setResizable(false);
}

CINDER_APP(SmokeApp, RendererGl, prepareSettings)
