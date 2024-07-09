#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <immintrin.h>
#include <omp.h>

using namespace ci;
using namespace ci::app;

constexpr int initialWindowWidth = 512;
constexpr int initialWindowHeight = 512;

constexpr double escapeRadiusSquared = 4.0;
constexpr double initialConstantX = -0.8;
constexpr double initialConstantY = 0.156;
constexpr double initialOffsetX = 0.0;
constexpr double initialOffsetY = 0.0;
constexpr double initialScale = 1.0;
constexpr int initialMaxIterations = 500;

__m256i computeJulia(__m256d positionX, __m256d positionY,
                     __m256d constantX, __m256d constantY,
                     __m256d escapeRadiusSquared, int maxIterations)
{
  __m256d x = positionX;
  __m256d y = positionY;
  __m256i iterations = _mm256_setzero_si256();
  __m256i ones = _mm256_set1_epi64x(1);
  __m256i maxIter = _mm256_set1_epi64x(maxIterations);
  for (int i = 0; i < maxIterations; ++i)
  {
    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d y2 = _mm256_mul_pd(y, y);
    __m256d xy = _mm256_mul_pd(x, y);
    __m256d mag2 = _mm256_add_pd(x2, y2);
    __m256d mask = _mm256_cmp_pd(mag2, escapeRadiusSquared, _CMP_LT_OQ);
    if (_mm256_movemask_pd(mask) == 0)
      break;
    __m256d newX = _mm256_add_pd(_mm256_sub_pd(x2, y2), constantX);
    y = _mm256_add_pd(_mm256_add_pd(xy, xy), constantY);
    x = newX;
    iterations = _mm256_add_epi64(iterations, _mm256_and_si256(ones, _mm256_castpd_si256(mask)));
  }
  return iterations;
}

class FractalApp : public App
{
public:
  void setup() override;
  void update() override;
  void draw() override;
  void keyDown(ci::app::KeyEvent event) override;
  void resize() override;

private:
  int mWindowWidth = initialWindowWidth;
  int mWindowHeight = initialWindowHeight;

  double mEscapeRadiusSquared = escapeRadiusSquared;
  double mConstantX = initialConstantX;
  double mConstantY = initialConstantY;
  double mScale = initialScale;
  double mOffsetX = initialOffsetX;
  double mOffsetY = initialOffsetY;
  int mMaxIterations = initialMaxIterations;

  Surface32f mSurface = Surface32f(mWindowWidth, mWindowHeight, false);

  bool mNeedsUpdate = true;
  bool mIsZooming = false;
};

void FractalApp::setup()
{
  ImGui::Initialize();
}

void FractalApp::update()
{
  ImGui::Begin("Parameters");
  int oldMaxIterations = mMaxIterations;
  double oldScale = mScale;
  ImGui::InputInt("Max iterations", &mMaxIterations, 50);
  ImGui::End();

  if (mIsZooming)
  {
    mScale *= 1.01f;
    mNeedsUpdate = true;
  }

  if (oldMaxIterations != mMaxIterations)
  {
    mNeedsUpdate = true;
  }
}

void FractalApp::draw()
{
  gl::clear(Color::black());

  if (mNeedsUpdate)
  {
#pragma omp parallel for
    for (int y = 0; y < mWindowHeight; ++y)
    {
      for (int x = 0; x < mWindowWidth; x += 4)
      {
        __m256d posX = _mm256_set_pd(
            (x + 3 - mWindowWidth / 2.0) / (mWindowWidth / 2.0f) / mScale + mOffsetX,
            (x + 2 - mWindowWidth / 2.0) / (mWindowWidth / 2.0f) / mScale + mOffsetX,
            (x + 1 - mWindowWidth / 2.0) / (mWindowWidth / 2.0f) / mScale + mOffsetX,
            (x - mWindowWidth / 2.0) / (mWindowWidth / 2.0f) / mScale + mOffsetX);

        __m256d posY = _mm256_set1_pd(
            (y - mWindowHeight / 2.0) / (mWindowWidth / 2.0f) / mScale + mOffsetY);

        __m256d constX = _mm256_set1_pd(mConstantX);
        __m256d constY = _mm256_set1_pd(mConstantY);
        __m256d escRadSq = _mm256_set1_pd(mEscapeRadiusSquared);

        __m256i iterations = computeJulia(posX, posY, constX, constY, escRadSq, mMaxIterations);

        int64_t iterCounts[4];
        _mm256_storeu_si256((__m256i *)iterCounts, iterations);

        for (int i = 0; i < 4; ++i)
        {
          if (x + i < mWindowWidth)
          {

            float normalizedValue = static_cast<float>(iterCounts[i]) / static_cast<float>(mMaxIterations);
            float r = normalizedValue < 0.5f ? 0 : 2 * (normalizedValue - 0.5f);
            float g = 0;
            float b = normalizedValue < 0.5f ? 2 * normalizedValue : 1 - 2 * (normalizedValue - 0.5f);
            Color color = Color(r, g, b);

            mSurface.setPixel(ivec2(x + i, y), color);
          }
        }
      }
    }
    mNeedsUpdate = false;
  }
  gl::Texture2dRef texture = gl::Texture2d::create(mSurface);
  gl::draw(texture);

  ImGui::Render();
}

void FractalApp::keyDown(ci::app::KeyEvent event)
{
  if (event.getChar() == 'q' || event.getChar() == 'Q')
  {
    quit();
  }
  else if (event.getChar() == 'r' || event.getChar() == 'R')
  {
    mScale = initialScale;
    mOffsetX = initialOffsetX;
    mOffsetY = initialOffsetY;
    mIsZooming = false;
    mNeedsUpdate = true;
  }
  else if (event.getChar() == '+')
  {
    mScale *= 1.1f;
    mNeedsUpdate = true;
  }
  else if (event.getChar() == '-')
  {
    mScale /= 1.1f;
    mNeedsUpdate = true;
  }
  else if (event.getCode() == KeyEvent::KEY_LEFT)
  {
    mOffsetX -= 0.1f / mScale;
    mNeedsUpdate = true;
  }
  else if (event.getCode() == KeyEvent::KEY_RIGHT)
  {
    mOffsetX += 0.1f / mScale;
    mNeedsUpdate = true;
  }
  else if (event.getCode() == KeyEvent::KEY_UP)
  {
    mOffsetY -= 0.1f / mScale;
    mNeedsUpdate = true;
  }
  else if (event.getCode() == KeyEvent::KEY_DOWN)
  {
    mOffsetY += 0.1f / mScale;
    mNeedsUpdate = true;
  }
  else if (event.getCode() == KeyEvent::KEY_SPACE)
  {
    mIsZooming = !mIsZooming;
  }
}

void FractalApp::resize()
{
  mWindowWidth = getWindowWidth();
  mWindowHeight = getWindowHeight();
  mSurface = Surface32f(mWindowWidth, mWindowHeight, false);
  mNeedsUpdate = true;
}

void prepareSettings(FractalApp::Settings *settings)
{
  settings->setWindowSize(initialWindowWidth, initialWindowHeight);
  settings->setFrameRate(60);
}

CINDER_APP(FractalApp, RendererGl, prepareSettings)
