#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <immintrin.h>
#include <iostream>
#include <mutex>
#include <omp.h>

constexpr int screenWidth = 1920;
constexpr int screenHeight = 1080;
constexpr double constantX = -0.391;
constexpr double constantY = -0.587;
constexpr double scale = 1.0;
constexpr double offsetX = 0.0;
constexpr double offsetY = 0.0;
constexpr int maxIterations = 1000;
constexpr int escapeRadiusSquared = 4;

constexpr float zoom_speed = 0.5f;
constexpr float zoom_factor = 1.5f;

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
    baseScale = 1.0;
    targetScale = 1.0;
    currentScale = 1.0;

    zoomTime = 0.0;
    lastFrameTime = 0.0;

    surface = ci::Surface32f(screenWidth, screenHeight, true);

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
  double baseScale;
  double targetScale;
  double currentScale;

  double zoomTime;
  double lastFrameTime;

  ci::Surface32f surface;
  ci::gl::Texture2dRef texture;

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

      renderJulia(&surface, screenWidth, screenHeight, constantX, constantY,
                  offsetX, offsetY, targetScale, maxIterations,
                  escapeRadiusSquared);

      std::cout << "workerLoop : DONE targetScale " << targetScale << std::endl;
      needsNewRender = false;
    }
  }
};

void FractalApp::setup() { lastFrameTime = getElapsedSeconds(); }

void FractalApp::update() {
  double currentTime = getElapsedSeconds();
  double deltaTime = currentTime - lastFrameTime;
  lastFrameTime = currentTime;

  zoomTime += deltaTime * zoom_speed;
  currentScale = baseScale * exp(zoomTime * log(zoom_factor));

  double scaleRatio = currentScale / baseScale;
  if (scaleRatio >= zoom_factor && !needsNewRender) {
    std::lock_guard<std::mutex> lock(mtx);

    texture = ci::gl::Texture2d::create(surface);

    baseScale = currentScale;
    targetScale = baseScale * zoom_factor;
    zoomTime = 0.0;

    needsNewRender = true;
    cv.notify_one();
  }
}

void FractalApp::draw() {
  ci::gl::clear(ci::Color(0, 0, 0));

  ci::vec2 center(getWindowWidth() * 0.5f, getWindowHeight() * 0.5f);
  float zoom = currentScale / baseScale;

  ci::gl::pushModelMatrix();
  ci::gl::translate(center);
  ci::gl::scale(ci::vec2(zoom));
  ci::gl::translate(-center);
  ci::gl::color(1, 1, 1, 1.0f);
  ci::gl::draw(texture);
  ci::gl::popModelMatrix();
}

void prepareSettings(FractalApp::Settings *settings) {
  settings->setFrameRate(60);
  settings->setWindowSize(screenWidth, screenHeight);
  settings->setResizable(false);
}

CINDER_APP(FractalApp, ci::app::RendererGl, prepareSettings)
