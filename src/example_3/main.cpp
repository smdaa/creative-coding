#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <omp.h>
#include <type_traits>

using namespace ci;
using namespace ci::app;

using FloatType = double;

constexpr int initialWindowWidth = 1280;
constexpr int initialWindowHeight = 720;

constexpr FloatType escapeRadiusSquared = 4.0;
constexpr FloatType initialConstantX = -0.8;
constexpr FloatType initialConstantY = 0.156;
constexpr FloatType initialScale = 1.0;
constexpr FloatType initialOffsetX = 0.0;
constexpr FloatType initialOffsetY = 0.0;
constexpr int initialMaxIterations = 200;

FloatType computeJulia(FloatType positionX, FloatType positionY,
                       FloatType constantX, FloatType constantY,
                       FloatType escapeRadiusSquared, int maxIterations) {
  FloatType x = positionX;
  FloatType y = positionY;
  int iterations = 0;
  while (iterations < maxIterations && x * x + y * y < escapeRadiusSquared) {
    FloatType newX = x * x - y * y + constantX;
    y = 2.0 * x * y + constantY;
    x = newX;
    ++iterations;
  }
  return static_cast<FloatType>(iterations);
}

class FractalApp : public App {
public:
  void setup() override;
  void update() override;
  void draw() override;
  void keyDown(ci::app::KeyEvent event) override;
  void resize() override;

private:
  int mWindowWidth = initialWindowWidth;
  int mWindowHeight = initialWindowHeight;

  FloatType mEscapeRadiusSquared = escapeRadiusSquared;
  FloatType mConstantX = initialConstantX;
  FloatType mConstantY = initialConstantY;
  FloatType mScale = initialScale;
  FloatType mOffsetX = initialOffsetX;
  FloatType mOffsetY = initialOffsetY;
  int mMaxIterations = initialMaxIterations;

  std::vector<Color> mColors;
  std::vector<ci::vec2> mPositions;
  gl::VboMeshRef mFractalMesh;

  bool mNeedsUpdate = true;
  bool mIsZooming = false;

  void initFractalMesh();
  void updateFractalMesh();
};

void FractalApp::initFractalMesh() {
  mPositions.clear();
  mColors.clear();
  for (int y = 0; y < mWindowHeight; y++) {
    for (int x = 0; x < mWindowWidth; x++) {
      mPositions.push_back(ci::vec2(x, y));
      mColors.push_back(Color::black());
    }
  }
  mFractalMesh = gl::VboMesh::create(
      mWindowHeight * mWindowWidth, GL_POINTS,
      {gl::VboMesh::Layout().attrib(geom::POSITION, 2).attrib(geom::COLOR, 3)});

  mFractalMesh->bufferAttrib(geom::POSITION, mPositions);
  mFractalMesh->bufferAttrib(geom::COLOR, mColors);

  mNeedsUpdate = true;
};

void FractalApp::updateFractalMesh() {
#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < mWindowHeight; y++) {
    for (int x = 0; x < mWindowWidth; x++) {
      FloatType fractalX =
          (x - mWindowWidth / 2.0f) / (mWindowWidth / 2.0f) / mScale + mOffsetX;
      FloatType fractalY =
          (y - mWindowHeight / 2.0f) / (mWindowWidth / 2.0f) / mScale +
          mOffsetY;
      FloatType iterations =
          computeJulia(fractalX, fractalY, mConstantX, mConstantY,
                       mEscapeRadiusSquared, mMaxIterations);
      Color color = iterations == mMaxIterations
                        ? Color::black()
                        : Color(1.0f, 1.0f, 1.0f) *
                              (1.0f - iterations / static_cast<FloatType>(
                                                       mMaxIterations));
      mColors[y * mWindowWidth + x] = color;
    }
  }
  mFractalMesh->bufferAttrib(geom::COLOR, mColors);
}

void FractalApp::setup() {
  ImGui::Initialize();
  initFractalMesh();
}

void FractalApp::update() {
  ImGui::Begin("Parameters");

  int oldMaxIterations = mMaxIterations;
  FloatType oldConstantX = mConstantX;
  FloatType oldConstantY = mConstantY;
  FloatType oldScale = mScale;

  ImGui::InputInt("Max iterations", &mMaxIterations);
  ImGui::End();

  if (mMaxIterations != oldMaxIterations || mConstantX != oldConstantX ||
      mConstantY != oldConstantY || mScale != oldScale) {
    mNeedsUpdate = true;
  }

  if (mIsZooming) {
    mScale *= 1.01f;
    mNeedsUpdate = true;
  }
  if (mNeedsUpdate) {
    updateFractalMesh();
    mNeedsUpdate = false;
  }
}

void FractalApp::draw() {
  gl::clear(Color::gray(0.2f));
  gl::draw(mFractalMesh);

  ImGui::Render();
}

void FractalApp::keyDown(ci::app::KeyEvent event) {
  if (event.getChar() == 'q' || event.getChar() == 'Q') {
    quit();
  } else if (event.getChar() == '+') {
    mScale *= 1.1f;
    mNeedsUpdate = true;
  } else if (event.getChar() == '-') {
    mScale /= 1.1f;
    mNeedsUpdate = true;
  } else if (event.getCode() == KeyEvent::KEY_LEFT) {
    mOffsetX -= 0.1f / mScale;
    mNeedsUpdate = true;
  } else if (event.getCode() == KeyEvent::KEY_RIGHT) {
    mOffsetX += 0.1f / mScale;
    mNeedsUpdate = true;
  } else if (event.getCode() == KeyEvent::KEY_UP) {
    mOffsetY -= 0.1f / mScale;
    mNeedsUpdate = true;
  } else if (event.getCode() == KeyEvent::KEY_DOWN) {
    mOffsetY += 0.1f / mScale;
    mNeedsUpdate = true;
  } else if (event.getCode() == KeyEvent::KEY_SPACE) {
    mIsZooming = !mIsZooming;
  }
}

void FractalApp::resize() {
  mWindowWidth = getWindowWidth();
  mWindowHeight = getWindowHeight();

  initFractalMesh();
  mNeedsUpdate = true;
}

void prepareSettings(FractalApp::Settings *settings) {
  settings->setWindowSize(initialWindowWidth, initialWindowHeight);
  settings->setFrameRate(30);
}

CINDER_APP(FractalApp, RendererGl, prepareSettings)
