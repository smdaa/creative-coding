#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <immintrin.h>
#include <queue>

constexpr int defaultScreenWidth = 1920;
constexpr int defaultScreenHeight = 1080;

constexpr double defaultConstantX = -0.7269;
constexpr double defaultConstantY = 0.1889;
constexpr double defaultOffsetX = 0.0;
constexpr double defaultOffsetY = 0.0;
constexpr double defaultScale = 1.0;
constexpr int defaultMaxIterations = 1000;
constexpr int defaultEscapeRadiusSquared = 4;
constexpr int defaultSamplingCount = 4;

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
      std::lock_guard<std::mutex> lock(m);
      requests.push({surface, params, callback});
    }
    cv.notify_one();
  }

private:
  struct RenderRequest {
    ci::Surface32f *surface;
    RenderParameters params;
    std::function<void()> callback;
  };

  std::thread renderThread;
  std::atomic<bool> running;
  std::queue<RenderRequest> requests;
  std::mutex m;
  std::condition_variable cv;

  void run() {
    while (running) {
      RenderRequest request;
      {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [this] { return !requests.empty() || !running; });
        if (!running && requests.empty()) {
          return;
        }
        request = requests.front();
        requests.pop();
      }
      renderJulia(request.surface, request.params);
      if (request.callback) {
        request.callback();
      }
    }
  }
};

class ViewManager {
public:
  ci::Surface32f currentSurface;

  ViewManager()
      : screenWidth(defaultScreenWidth), screenHeight(defaultScreenHeight),
        renderInProgress(false), renderer() {
    currentSurface = ci::Surface32f(screenWidth, screenHeight, false);
    backSurface = ci::Surface32f(screenWidth, screenHeight, false);
    currentCrop = ci::Area(0, 0, screenWidth, screenHeight);
  }

  void requestStaticRender(const RenderParameters &params) {
    // If we have current params and they're the same as requested, no need to
    // re-render
    if (hasCurrentParams && currentParams == params) {
      return;
    }
    // Ignore new requests while rendering
    if (renderInProgress) {
      return;
    }
    renderInProgress = true;
    currentParams = params;
    hasCurrentParams = true;
    auto callback = [this]() {
      std::lock_guard<std::mutex> lock(staticRenderMutex);
      staticRenderQueue.push(true);
    };
    renderer.requestRender(&backSurface, params, callback);
  }

  void update() { processStaticRenderQueue(); }

private:
  int screenWidth;
  int screenHeight;
  bool renderInProgress;
  bool hasCurrentParams;
  Renderer renderer;

  ci::Area currentCrop;
  ci::Surface32f backSurface;
  RenderParameters currentParams;

  std::mutex staticRenderMutex;
  std::queue<bool> staticRenderQueue;

  void processStaticRenderQueue() {
    std::lock_guard<std::mutex> lock(staticRenderMutex);
    if (!staticRenderQueue.empty()) {
      staticRenderQueue.pop();
      std::swap(currentSurface, backSurface);
      renderInProgress = false;
    }
  }
};

class FractalApp : public ci::app::App {
public:
  FractalApp() : viewManager() {
    texture =
        ci::gl::Texture2d::create(defaultScreenWidth, defaultScreenHeight);
  }

  void setup() override {
    params = {defaultConstantX,
              defaultConstantY,
              defaultOffsetX,
              defaultOffsetY,
              defaultScale,
              defaultMaxIterations,
              defaultEscapeRadiusSquared,
              defaultSamplingCount};
    ImGui::Initialize();
  }

  void update() override {
    ImGui::Begin("Parameter Control");
    ImGui::InputDouble("Constant X", &params.constantX, 0.01, 0.1, "%.6f");
    ImGui::InputDouble("Constant Y", &params.constantY, 0.01, 0.1, "%.6f");
    ImGui::InputDouble("Offset X", &params.offsetX, 0.1, 1.0, "%.3f");
    ImGui::InputDouble("Offset Y", &params.offsetY, 0.1, 1.0, "%.3f");
    ImGui::InputDouble("Scale", &params.scale, 0.01, 0.1, "%.4f");
    ImGui::InputInt("Max Iterations", &params.maxIterations, 10, 100);
    ImGui::InputInt("Escape Radius Squared", &params.escapeRadiusSquared, 1);
    ImGui::InputInt("Sampling Count", &params.samplingCount, 1);
    ImGui::End();

    viewManager.requestStaticRender(params);
    viewManager.update();
    texture->update(viewManager.currentSurface);
  }

  void draw() override {
    ci::gl::clear(ci::Color(0, 0, 0));
    ci::gl::draw(texture);
  }

private:
  ViewManager viewManager;
  RenderParameters params;
  ci::gl::Texture2dRef texture;
};

void prepareSettings(FractalApp::Settings *settings) {
  settings->setFrameRate(60);
  settings->setWindowSize(defaultScreenWidth, defaultScreenHeight);
  settings->setResizable(false);
}

CINDER_APP(FractalApp, ci::app::RendererGl, prepareSettings)