#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "cinder/ip/Resize.h"

#include <immintrin.h>
#include <omp.h>
#include <queue>

constexpr int screenWidth = 1920;
constexpr int screenHeight = 1080;
constexpr double constantX = -0.391;
constexpr double constantY = -0.587;
constexpr double scale = 1.0;
constexpr double offsetX = 0.0;
constexpr double offsetY = 0.0;
constexpr int maxIterations = 1000;
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
          ci::Color color = ci::Color(brightness, brightness, brightness);
          surface->setPixel(ci::ivec2(x + i, y), color);
        }
      }
    }
  }
}

class Renderer {
public:
  Renderer() : isRunning(true), scale(1.0), zoomSpeed(0.1) {
    surface = ci::Surface32f(screenWidth, screenHeight, false);
    worker = std::thread(&Renderer::workerLoop, this);
  }
  ~Renderer() {
    isRunning = false;
    if (worker.joinable()) {
      worker.join();
    }
  }

  void getFrame(ci::Surface32f &outSurface, double requested_scale) {
    std::lock_guard<std::mutex> lock(queueMutex);

    if (queue.empty()) {
      return;
    }

    while (queue.size() > 1) {
      queue.pop();
    }

    double frame_scale = queue.front().first;
    const ci::Surface32f &frame = queue.front().second;
    double zoomFactor = requested_scale / frame_scale;
    int originalWidth = frame.getWidth();
    int originalHeight = frame.getHeight();
    float aspectRatio =
        static_cast<float>(originalWidth) / static_cast<float>(originalHeight);
    int cropWidth = static_cast<int>(originalWidth / zoomFactor);
    int cropHeight = static_cast<int>(cropWidth / aspectRatio);
    int cropX = (originalWidth - cropWidth) / 2;
    int cropY = (originalHeight - cropHeight) / 2;
    ci::Area cropArea(cropX, cropY, cropX + cropWidth, cropY + cropHeight);
    ci::Surface32f croppedSurface = frame.clone(cropArea);
    outSurface = ci::Surface32f(screenWidth, screenHeight, frame.hasAlpha(),
                                frame.getChannelOrder());
    ci::ip::resize(croppedSurface, &outSurface);
  }

private:
  bool isRunning;
  double scale;
  double zoomSpeed;
  std::queue<std::pair<double, ci::Surface32f>> queue;
  std::mutex queueMutex;
  std::thread worker;
  ci::Surface32f surface;

  void workerLoop() {
    while (isRunning) {
      renderJulia(&surface, screenWidth, screenHeight, constantX, constantY,
                  offsetX, offsetY, scale, maxIterations, escapeRadiusSquared);
      {
        std::lock_guard<std::mutex> lock(queueMutex);
        queue.push(std::make_pair(scale, surface));

        std::cout << "Rendered at scale: " << scale
                  << ". Frame added to queue. Queue size: " << queue.size()
                  << std::endl;
      }
      scale = scale + zoomSpeed;
    }
  }
};

class FractalApp : public ci::app::App {
public:
  FractalApp() : scale(1.0), zoomSpeed(0.01), renderer() {}

  void setup() override;
  void update() override;
  void draw() override;

private:
  double scale;
  double zoomSpeed;
  Renderer renderer;
  ci::gl::Texture2dRef texture;
};

void FractalApp::setup() {}

void FractalApp::update() {
  ci::Surface32f latestSurface;
  renderer.getFrame(latestSurface, scale);
  texture = ci::gl::Texture::create(latestSurface);
  scale = scale + zoomSpeed;
}

void FractalApp::draw() {
  ci::gl::clear(ci::Color(0, 0, 0));
  ci::gl::draw(texture);
}

void prepareSettings(FractalApp::Settings *settings) {
  settings->setFrameRate(60);
  settings->setWindowSize(screenWidth, screenHeight);
  settings->setResizable(false);
}

CINDER_APP(FractalApp, ci::app::RendererGl, prepareSettings)
