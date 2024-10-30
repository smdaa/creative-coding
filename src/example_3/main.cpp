#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <atomic>
#include <condition_variable>
#include <immintrin.h>
#include <mutex>
#include <queue>
#include <thread>

constexpr int defaultScreenWidth = 500;
constexpr int defaultScreenHeight = 500;

constexpr double defaultConstantX = -0.7269;
constexpr double defaultConstantY = 0.1889;
constexpr double defaultOffsetX = 0.0;
constexpr double defaultOffsetY = 0.0;
constexpr double defaultScale = 1.0;
constexpr int defaultMaxIterations = 200;
constexpr int defaultEscapeRadiusSquared = 4;
constexpr int defaultSamplingCount = 2;

constexpr double defaultZoomFactor = 1.1;

struct RenderParameters {
  double constantX;
  double constantY;
  double offsetX;
  double offsetY;
  double scale;
  int maxIterations;
  int escapeRadiusSquared;
  int samplingCount;

  bool operator==(const RenderParameters &other) const {
    return constantX == other.constantX && constantY == other.constantY &&
           offsetX == other.offsetX && offsetY == other.offsetY &&
           scale == other.scale && maxIterations == other.maxIterations &&
           escapeRadiusSquared == other.escapeRadiusSquared &&
           samplingCount == other.samplingCount;
  }
};

__m256i computeJulia(__m256d positionX, __m256d positionY, __m256d constantX,
                     __m256d constantY, __m256d escapeRadiusSquared,
                     int maxIterations) {
  __m256d x = positionX;
  __m256d y = positionY;
  __m256i iterations = _mm256_setzero_si256();
  __m256i ones = _mm256_set1_epi64x(1);
  for (int i = 0; i < maxIterations; ++i) {
    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d y2 = _mm256_mul_pd(y, y);
    __m256d mag2 = _mm256_add_pd(x2, y2);

    __m256d mask = _mm256_cmp_pd(mag2, escapeRadiusSquared, _CMP_LT_OQ);
    if (_mm256_movemask_pd(mask) == 0)
      break;

    __m256d xy = _mm256_mul_pd(x, y);
    __m256d newX = _mm256_add_pd(_mm256_sub_pd(x2, y2), constantX);
    y = _mm256_add_pd(_mm256_add_pd(xy, xy), constantY);
    x = newX;

    iterations = _mm256_add_epi64(
        iterations, _mm256_and_si256(ones, _mm256_castpd_si256(mask)));
  }
  return iterations;
}

void renderJulia(ci::Surface32f *surface, const RenderParameters params) {

  int surfaceWidth = surface->getWidth();
  int screenHeight = surface->getHeight();

  double surfaceWidthHalf = surfaceWidth / 2.0;
  double surfaceHeightHalf = screenHeight / 2.0;
  double scaleInv = 1.0 / params.scale;

#pragma omp parallel for
  for (int y = 0; y < screenHeight; y += 1) {
    for (int x = 0; x < surfaceWidth; x += 4) {
      __m256d surfaceWidthHalfVec = _mm256_set1_pd(surfaceWidthHalf);
      __m256d scaleInvVec = _mm256_set1_pd(scaleInv);
      __m256d xVec = _mm256_set_pd(x + 3, x + 2, x + 1, x);
      __m256d xShifted = _mm256_sub_pd(xVec, surfaceWidthHalfVec);

      double accumulatedR[4] = {0}, accumulatedG[4] = {0};
      double accumulatedB[4] = {0}, accumulatedA[4] = {0};

      for (int sy = 0; sy < params.samplingCount; sy++) {
        for (int sx = 0; sx < params.samplingCount; sx++) {
          double subX = sx / static_cast<double>(params.samplingCount);
          double subY = sy / static_cast<double>(params.samplingCount);

          __m256d posX = _mm256_add_pd(
              _mm256_mul_pd(
                  _mm256_mul_pd(_mm256_add_pd(xShifted, _mm256_set1_pd(subX)),
                                scaleInvVec),
                  _mm256_set1_pd(1.0 / surfaceWidthHalf)),
              _mm256_set1_pd(params.offsetX));

          __m256d posY = _mm256_set1_pd((y + subY - surfaceHeightHalf) /
                                            surfaceWidthHalf * scaleInv +
                                        params.offsetY);

          __m256d constX = _mm256_set1_pd(params.constantX);
          __m256d constY = _mm256_set1_pd(params.constantY);
          __m256d escRadSq = _mm256_set1_pd(params.escapeRadiusSquared);

          __m256i iterations = computeJulia(posX, posY, constX, constY,
                                            escRadSq, params.maxIterations);

          int64_t iterCounts[4];
          _mm256_storeu_si256((__m256i *)iterCounts, iterations);

          for (int i = 0; i < 4; ++i) {
            if (x + i < surfaceWidth) {
              /*
                int idx = std::min(iterCounts[i],
                                   static_cast<int64_t>(params.maxIterations));
                ci::ColorA color = palette[idx];
                accumulatedR[i] += color.r;
                accumulatedG[i] += color.g;
                accumulatedB[i] += color.b;
                accumulatedA[i] += color.a;
              */
              double brightness = static_cast<double>(iterCounts[i]) /
                                  static_cast<double>(params.maxIterations);
              accumulatedR[i] += brightness;
              accumulatedG[i] += brightness;
              accumulatedB[i] += brightness;
              accumulatedA[i] += brightness;
            }
          }
        }
      }

      double totalSamples = params.samplingCount * params.samplingCount;
      for (int i = 0; i < 4; ++i) {
        if (x + i < surfaceWidth) {
          ci::ColorA finalColor(
              accumulatedR[i] / totalSamples, accumulatedG[i] / totalSamples,
              accumulatedB[i] / totalSamples, accumulatedA[i] / totalSamples);

          surface->setPixel(ci::ivec2(x + i, y), finalColor);
        }
      }
    }
  }
}

struct RenderRequest {
  ci::Surface32f *surface;
  RenderParameters params;
  std::function<void()> callback;
};

class Renderer {
public:
  Renderer() : running(true) {
    renderThread = std::thread(&Renderer::run, this);
  }

  ~Renderer() {
    running = false;
    cv.notify_one();
    if (renderThread.joinable()) {
      renderThread.join();
    };
  }

  void requestRender(ci::Surface32f *surface, const RenderParameters &params,
                     std::function<void()> callback) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      requests.push({surface, params, callback});
    }
    cv.notify_one();
  }

private:
  std::thread renderThread;
  std::atomic<bool> running;
  std::queue<RenderRequest> requests;
  std::mutex mtx;
  std::condition_variable cv;

  void run() {
    while (running) {
      RenderRequest request;
      {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [this] { return !requests.empty() || !running; });
        if (!running && requests.empty()) {
          return;
        }
        request = std::move(requests.front());
        requests.pop();
      }
      renderJulia(request.surface, request.params);
      if (request.callback) {
        request.callback();
      }
    }
  }
};

class FractalApp : public ci::app::App {
public:
  FractalApp() {
    params = {defaultConstantX,
              defaultConstantY,
              defaultOffsetX,
              defaultOffsetY,
              defaultScale,
              defaultMaxIterations,
              defaultEscapeRadiusSquared,
              defaultSamplingCount};

    currentSurface =
        ci::Surface32f(defaultScreenWidth, defaultScreenHeight, false);
    backSurface =
        ci::Surface32f(defaultScreenWidth, defaultScreenHeight, false);
    texture =
        ci::gl::Texture2d::create(defaultScreenWidth, defaultScreenHeight);
    renderer = std::make_unique<Renderer>();
    renderInProgress = false;
    firstStaticRenderFlag = true;

    isZooming = false;
    zoomFactor = 1.1;
    firstZoomRenderStart = true;
    firstZoomRenderDone = false;
    currentDisplayedScale = params.scale;
  }

  void setup() override { ImGui::Initialize(); }

  void update() override {

    bool parametersChanged = false;
    ImGui::Begin("Parameter Control");
    parametersChanged |=
        ImGui::InputDouble("Constant X", &params.constantX, 0.01, 0.1, "%.6f");
    parametersChanged |=
        ImGui::InputDouble("Constant Y", &params.constantY, 0.01, 0.1, "%.6f");
    parametersChanged |=
        ImGui::InputDouble("Offset X", &params.offsetX, 0.1, 1.0, "%.3f");
    parametersChanged |=
        ImGui::InputDouble("Offset Y", &params.offsetY, 0.1, 1.0, "%.3f");
    parametersChanged |=
        ImGui::InputDouble("Scale", &params.scale, 0.01, 0.1, "%.4f");
    parametersChanged |=
        ImGui::InputInt("Max Iterations", &params.maxIterations, 10, 100);
    parametersChanged |= ImGui::InputInt("Escape Radius Squared",
                                         &params.escapeRadiusSquared, 1);
    parametersChanged |=
        ImGui::InputInt("Sampling Count", &params.samplingCount, 1);
    ImGui::End();

    if (isZooming) {
      // Initiate the first render for zooming if this is the start
      if (firstZoomRenderStart) {
        clearSurface(currentSurface, ci::Color(0, 0, 0));
        clearSurface(backSurface, ci::Color(0, 0, 0));

        renderInProgress = true;
        firstZoomRenderStart = false;

        renderer->requestRender(&backSurface, params, [this]() {
          renderInProgress = false;
          firstZoomRenderDone = true;
          std::swap(backSurface, currentSurface);
          zoomTimer.start();
        });
      }

      // Adjust scale and request new renders after the first frame is ready
      if (firstZoomRenderDone) {
        std::cout << "OK firstZoomRenderDone" << std::endl;
        std::cout << currentDisplayedScale << std::endl;

        currentDisplayedScale = params.scale * std::exp(zoomTimer.getSeconds() *
                                                        std::log(zoomFactor));

        // Only start a new render if the current displayed scale exceeds the
        // zoom factor
        if ((currentDisplayedScale / params.scale) >= zoomFactor &&
            !renderInProgress) {
          params.scale = currentDisplayedScale;

          std::cout << params.scale << std::endl;

          renderInProgress = true;

          renderer->requestRender(&backSurface, params, [this]() {
            renderInProgress = false;
            std::swap(backSurface,
                      currentSurface); // Swap the surfaces after render
          });

          zoomTimer.start();
        }
      }

    } else {
      if ((parametersChanged || firstStaticRenderFlag) && !renderInProgress) {
        renderInProgress = true;
        firstStaticRenderFlag = false;
        renderer->requestRender(&backSurface, params, [this]() {
          renderInProgress = false;
          std::swap(backSurface, currentSurface);
        });
      }
    }

    texture->update(currentSurface);
  }

  void draw() override {
    ci::gl::clear(ci::Color(0, 0, 0));

    if (isZooming && firstZoomRenderDone) {

      ci::vec2 center(defaultScreenWidth * 0.5f, defaultScreenHeight * 0.5f);
      float zoomLevel = currentDisplayedScale / params.scale;

      ci::gl::pushModelMatrix();
      ci::gl::translate(center);
      ci::gl::scale(ci::vec2(zoomLevel));
      ci::gl::translate(-center);
      ci::gl::color(1, 1, 1, 1.0f);
      ci::gl::draw(texture);
      ci::gl::popModelMatrix();

    } else {
      ci::gl::draw(texture);
    }
  }

  void keyDown(ci::app::KeyEvent event) override {
    char key = event.getChar();
    if (key == 'q' || key == 'Q') {
      quit();
    } else if (event.getCode() == ci::app::KeyEvent::KEY_SPACE) {
      isZooming = !isZooming;
      firstZoomRenderStart = true;
    }
  }

private:
  RenderParameters params;
  ci::Surface32f currentSurface;
  ci::Surface32f backSurface;
  ci::gl::Texture2dRef texture;
  std::unique_ptr<Renderer> renderer;
  bool renderInProgress;
  bool firstStaticRenderFlag;

  bool isZooming;
  double zoomFactor;
  bool firstZoomRenderStart;
  bool firstZoomRenderDone;
  double currentDisplayedScale;
  ci::Timer zoomTimer;

  void clearSurface(ci::Surface32f &surface, const ci::Color &color) {
    int width = surface.getWidth();
    int height = surface.getHeight();

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        surface.setPixel(ci::ivec2(x, y), color);
      }
    }
  }
};

void prepareSettings(FractalApp::Settings *settings) {
  settings->setFrameRate(60);
  settings->setWindowSize(defaultScreenWidth, defaultScreenHeight);
  settings->setResizable(false);
}

CINDER_APP(FractalApp, ci::app::RendererGl, prepareSettings)
