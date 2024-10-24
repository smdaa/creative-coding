#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <omp.h>

constexpr int screenWidth = 500;
constexpr int screenHeight = 500;
constexpr double constantX = -0.391;
constexpr double constantY = -0.587;
constexpr double scale = 1.0;
constexpr double offsetX = 0.0;
constexpr double offsetY = 0.0;
constexpr int maxIterations = 100;
constexpr int escapeRadiusSquared = 4;

__m256i computeJulia(__m256d positionX, __m256d positionY, __m256d constantX,
                     __m256d constantY, __m256d escapeRadiusSquared,
                     int maxIterations) {
  __m256d x = positionX;
  __m256d y = positionY;
  __m256i iterations = _mm256_setzero_si256();
  __m256i ones = _mm256_set1_epi64x(1);
  __m256i maxIter = _mm256_set1_epi64x(maxIterations);
  for (int i = 0; i < maxIterations; ++i) {
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
    iterations = _mm256_add_epi64(
        iterations, _mm256_and_si256(ones, _mm256_castpd_si256(mask)));
  }
  return iterations;
}

void renderJulia(ci::Surface32f *surface, int screenWidth, int screenHeight,
                 double constantX, double constantY, double offsetX,
                 double offsetY, double scale, int maxIterations,
                 double escapeRadiusSquared) {
#pragma omp parallel for
  for (int y = 0; y < screenHeight; ++y) {
    for (int x = 0; x < screenWidth; x += 4) {
      double screenWidthHalf = screenWidth / 2.0;
      double scaleInv = 1.0 / scale;
      __m256d screenWidthHalfVec = _mm256_set1_pd(screenWidthHalf);
      __m256d scaleInvVec = _mm256_set1_pd(scaleInv);
      __m256d xVec = _mm256_set_pd(x + 3, x + 2, x + 1, x);
      __m256d xShifted = _mm256_sub_pd(xVec, screenWidthHalfVec);
      __m256d posX =
          _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(xShifted, scaleInvVec),
                                      _mm256_set1_pd(1.0 / screenWidthHalf)),
                        _mm256_set1_pd(offsetX));
      __m256d posY = _mm256_set1_pd(
          (y - screenHeight / 2.0) / screenWidthHalf * scaleInv + offsetY);

      __m256d constX = _mm256_set1_pd(constantX);
      __m256d constY = _mm256_set1_pd(constantY);
      __m256d escRadSq = _mm256_set1_pd(escapeRadiusSquared);

      __m256i iterations =
          computeJulia(posX, posY, constX, constY, escRadSq, maxIterations);

      int64_t iterCounts[4];
      _mm256_storeu_si256((__m256i *)iterCounts, iterations);

      for (int i = 0; i < 4; ++i) {
        if (x + i < screenWidth) {
          float brightness = static_cast<float>(iterCounts[i]) /
                             static_cast<float>(maxIterations);
          ci::ColorA color =
              ci::ColorA(brightness, brightness, brightness, brightness);
          surface->setPixel(ci::ivec2(x + i, y), color);
        }
      }
    }
  }
}

class FractalApp : public ci::app::App {
public:
  FractalApp() {
    targetScale = 1.0;
    activeScale = 1.0;
    backScale = 1.0;
    fadeTime = 0.0f;

    activeSurface = ci::Surface32f(screenWidth, screenHeight, true);
    backSurface = ci::Surface32f(screenWidth, screenHeight, true);

    needsNewRender = false;
    workerRunning = true;
    worker = std::thread(&FractalApp::workerLoop, this);
  }

  ~FractalApp() {
    {
      std::lock_guard<std::mutex> lock(mtx);
      workerRunning = false;
    }
    cv.notify_one();

    if (worker.joinable()) {
      worker.join();
    }
  }

  void setup() override;
  void update() override;
  void draw() override;

private:
  double targetScale;
  double activeScale;
  double backScale;
  float fadeTime;

  ci::Surface32f activeSurface;
  ci::Surface32f backSurface;
  ci::gl::Texture2dRef activeTexture;
  ci::gl::Texture2dRef backTexture;

  bool needsNewRender;
  bool workerRunning;
  std::thread worker;

  std::mutex mtx;
  std::condition_variable cv;

  void workerLoop() {
    while (true) {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [this] { return needsNewRender || !workerRunning; });

      if (!workerRunning)
        break;

      std::cout << "workerLoop : targetScale " << targetScale << std::endl;
      renderJulia(&backSurface, screenWidth, screenHeight, constantX, constantY,
                  offsetX, offsetY, targetScale, maxIterations,
                  escapeRadiusSquared);

      needsNewRender = false;
    }
  }

  void swapBuffers() {
    std::swap(activeSurface, backSurface);
    std::swap(activeTexture, backTexture);
    activeScale = backScale;
    backScale = targetScale;
  }
};

void FractalApp::setup() {
  renderJulia(&activeSurface, screenWidth, screenHeight, constantX, constantY,
              offsetX, offsetY, activeScale, maxIterations,
              escapeRadiusSquared);
  renderJulia(&backSurface, screenWidth, screenHeight, constantX, constantY,
              offsetX, offsetY, backScale, maxIterations, escapeRadiusSquared);

  activeTexture = ci::gl::Texture2d::create(activeSurface);
  backTexture = ci::gl::Texture2d::create(backSurface);
}

void FractalApp::update() {

  if (!needsNewRender && fadeTime >= 1.0) {
    std::lock_guard<std::mutex> lock(mtx);
    std::swap(activeSurface, backSurface);
    std::swap(activeTexture, backTexture);

    backTexture = ci::gl::Texture2d::create(backSurface);

    targetScale = targetScale * 1.01;
    needsNewRender = true;
    cv.notify_one();

    fadeTime = 0.0f;
  }

  fadeTime += 0.1;
  std::cout << "FractalApp::update : fadeAmount " << fadeTime << std::endl;
}

void FractalApp::draw() {
  ci::gl::clear(ci::Color(0, 0, 0));

  ci::gl::color(1, 1, 1, 1.0f);
  ci::gl::draw(activeTexture);

  ci::gl::color(1, 1, 1, 1.0f - fadeTime / 1.0);
  ci::gl::draw(backTexture);
}

void prepareSettings(FractalApp::Settings *settings) {
  settings->setFrameRate(60);
  settings->setWindowSize(screenWidth, screenHeight);
  settings->setResizable(false);
}

CINDER_APP(FractalApp, ci::app::RendererGl, prepareSettings)
