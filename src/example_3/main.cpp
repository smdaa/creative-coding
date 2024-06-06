#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <omp.h>

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
constexpr int initialMaxIterations = 100;

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

  enum ColorMap { GRAYSCALE, JET, HOT, COOL, RAINBOW };
  ColorMap mCurrentColorMap = GRAYSCALE;

  void initFractalMesh();
  void updateFractalMesh();
  Color getColorForIteration(int iterations) const;
  Color interpolateColor(const Color &color1, const Color &color2,
                         float factor) const;
  Color getGrayscaleColor(float t) const;
  Color getJetColor(float t) const;
  Color getHotColor(float t) const;
  Color getCoolColor(float t) const;
  Color getRainbowColor(float t) const;
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
}

Color FractalApp::interpolateColor(const Color &color1, const Color &color2,
                                   float factor) const {
  return color1 * (1.0f - factor) + color2 * factor;
}

Color FractalApp::getGrayscaleColor(float t) const { return Color(t, t, t); }

Color FractalApp::getJetColor(float t) const {
  if (t < 0.25f)
    return interpolateColor(Color(0, 0, 0.5), Color(0, 0, 1), t * 4);
  else if (t < 0.5f)
    return interpolateColor(Color(0, 0, 1), Color(0, 1, 1), (t - 0.25f) * 4);
  else if (t < 0.75f)
    return interpolateColor(Color(0, 1, 1), Color(1, 1, 0), (t - 0.5f) * 4);
  else
    return interpolateColor(Color(1, 1, 0), Color(1, 0, 0), (t - 0.75f) * 4);
}

Color FractalApp::getHotColor(float t) const {
  if (t < 0.33f)
    return interpolateColor(Color(0, 0, 0), Color(1, 0, 0), t * 3);
  else if (t < 0.66f)
    return interpolateColor(Color(1, 0, 0), Color(1, 1, 0), (t - 0.33f) * 3);
  else
    return interpolateColor(Color(1, 1, 0), Color(1, 1, 1), (t - 0.66f) * 3);
}

Color FractalApp::getCoolColor(float t) const {
  return interpolateColor(Color(0, 1, 1), Color(1, 0, 1), t);
}

Color FractalApp::getRainbowColor(float t) const {
  if (t < 0.2f)
    return interpolateColor(Color(1, 0, 0), Color(1, 0.5, 0), t * 5);
  else if (t < 0.4f)
    return interpolateColor(Color(1, 0.5, 0), Color(1, 1, 0), (t - 0.2f) * 5);
  else if (t < 0.6f)
    return interpolateColor(Color(1, 1, 0), Color(0, 1, 0), (t - 0.4f) * 5);
  else if (t < 0.8f)
    return interpolateColor(Color(0, 1, 0), Color(0, 0, 1), (t - 0.6f) * 5);
  else
    return interpolateColor(Color(0, 0, 1), Color(0.5, 0, 0.5), (t - 0.8f) * 5);
}

Color FractalApp::getColorForIteration(int iterations) const {
  if (iterations == mMaxIterations) {
    return Color::black();
  }

  float t = static_cast<float>(iterations) / mMaxIterations;

  switch (mCurrentColorMap) {
  case GRAYSCALE:
    return getGrayscaleColor(t);
  case JET:
    return getJetColor(t);
  case HOT:
    return getHotColor(t);
  case COOL:
    return getCoolColor(t);
  case RAINBOW:
    return getRainbowColor(t);
  default:
    return getGrayscaleColor(t);
  }
}

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
      mColors[y * mWindowWidth + x] =
          getColorForIteration(static_cast<int>(iterations));
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
  ColorMap oldColorMap = mCurrentColorMap;

  ImGui::InputInt("Max iterations", &mMaxIterations);
  const char *items[] = {"Grayscale", "Jet", "Hot", "Cool", "Rainbow"};
  int currentColorMapIndex = static_cast<int>(mCurrentColorMap);
  ImGui::Combo("Color map", &currentColorMapIndex, items, IM_ARRAYSIZE(items));
  mCurrentColorMap = static_cast<ColorMap>(currentColorMapIndex);

  ImGui::End();

  if (mMaxIterations != oldMaxIterations || mConstantX != oldConstantX ||
      mConstantY != oldConstantY || mScale != oldScale ||
      mCurrentColorMap != oldColorMap) {
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
