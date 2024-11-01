#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <immintrin.h>
#include <omp.h>
#include <optional>
#include <queue>

constexpr int defaultScreenWidth = 1000;
constexpr int defaultScreenHeight = 1000;
constexpr double defaultConstantX = -0.7269;
constexpr double defaultConstantY = 0.1889;
constexpr double defaultOffsetX = 0.0;
constexpr double defaultOffsetY = 0.0;
constexpr double defaultScale = 1.0;
constexpr int defaultMaxIterations = 200;
constexpr int defaultEscapeRadiusSquared = 4;
constexpr int defaultSamplingCount = 2;
constexpr double defaultZoomFactor = 1.5;

const std::vector<ci::Color> defaultBaseColors = {
    ci::Color(0.7f, 0.7f, 0.8f), // Gray with slight purple tint
    ci::Color(0.6f, 0.4f, 0.8f), // Light Purple
    ci::Color(0.4f, 0.2f, 0.8f), // Purple-Blue
    ci::Color(0.2f, 0.3f, 0.9f)  // Blue
};

struct RenderParameters {
  double constantX;
  double constantY;
  double offsetX;
  double offsetY;
  double scale;
  int maxIterations;
  int escapeRadiusSquared;
  int samplingCount;
};

const RenderParameters defaultRenderParameters =
    RenderParameters{defaultConstantX,
                     defaultConstantY,
                     defaultOffsetX,
                     defaultOffsetY,
                     defaultScale,
                     defaultMaxIterations,
                     defaultEscapeRadiusSquared,
                     defaultSamplingCount};

void clampRenderParameters(RenderParameters &params) {
  params.constantX = std::clamp(params.constantX, -1.0, 1.0);
  params.constantY = std::clamp(params.constantY, -1.0, 1.0);
  params.offsetX = std::clamp(params.offsetX, -1.0, 1.0);
  params.offsetY = std::clamp(params.offsetY, -1.0, 1.0);
  params.samplingCount = std::max(params.samplingCount, 1);
  params.maxIterations = std::max(params.maxIterations, 1);
  params.escapeRadiusSquared = std::max(params.escapeRadiusSquared, 1);
  params.scale = std::max(params.scale, 0.01);
}

std::vector<ci::Color> generatePalette(const std::vector<ci::Color> &baseColors,
                                       int maxIterations) {
  std::vector<ci::Color> palette(maxIterations + 1);
  palette[maxIterations] = ci::Color(0.1f, 0.1f, 0.3f);
  for (int i = 0; i < maxIterations; i++) {
    float t = (float)i / maxIterations;
    t = t * (baseColors.size() - 1);
    int idx = static_cast<int>(t);
    float fract = t - idx;
    const ci::Color &c1 = baseColors[idx];
    const ci::Color &c2 =
        baseColors[std::min(idx + 1, (int)baseColors.size() - 1)];
    palette[i] =
        ci::Color(ci::lerp(c1.r, c2.r, fract), ci::lerp(c1.g, c2.g, fract),
                  ci::lerp(c1.b, c2.b, fract));
  }

  return palette;
}

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

void renderJulia(ci::Surface32f &surface, const RenderParameters &params,
                 const std::vector<ci::Color> &palette) {
  int surfaceWidth = surface.getWidth();
  int screenHeight = surface.getHeight();
  double surfaceWidthHalf = surfaceWidth / 2.0;
  double surfaceHeightHalf = screenHeight / 2.0;
  double scaleInv = 1.0 / params.scale;
  double totalSamples = params.samplingCount * params.samplingCount;
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
              int idx = std::min(iterCounts[i],
                                 static_cast<int64_t>(params.maxIterations));
              ci::ColorA color = palette[idx];
              accumulatedR[i] += color.r;
              accumulatedG[i] += color.g;
              accumulatedB[i] += color.b;
              accumulatedA[i] += color.a;
            }
          }
        }
      }
      for (int i = 0; i < 4; ++i) {
        if (x + i < surfaceWidth) {
          ci::ColorA finalColor(
              accumulatedR[i] / totalSamples, accumulatedG[i] / totalSamples,
              accumulatedB[i] / totalSamples, accumulatedA[i] / totalSamples);
          surface.setPixel(ci::ivec2(x + i, y), finalColor);
        }
      }
    }
  }
}

struct RenderRequest {
  ci::Surface32f &surface;
  const RenderParameters &params;
  const std::vector<ci::Color> &palette;
  const std::function<void()> callback;

  RenderRequest(ci::Surface32f &surf, const RenderParameters &p,
                const std::vector<ci::Color> &pal,
                const std::function<void()> cb)
      : surface(surf), params(p), palette(pal), callback(cb) {}
};

class Renderer {
public:
  Renderer() : running(true) {
    renderThread = std::thread(&Renderer::run, this);
  }

  ~Renderer() {
    {
      std::lock_guard<std::mutex> lock(mtx);
      running = false;
    }
    cv.notify_one();
    if (renderThread.joinable()) {
      renderThread.join();
    }
  }

  void requestRender(ci::Surface32f &surface, const RenderParameters &params,
                     const std::vector<ci::Color> &palette,
                     const std::function<void()> callback) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      requests.emplace(surface, params, palette, callback);
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
    while (true) {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [this] { return !requests.empty() || !running; });
      if (!running && requests.empty()) {
        return;
      }
      if (!requests.empty()) {
        RenderRequest request = std::move(requests.front());
        requests.pop();
        lock.unlock();
        renderJulia(request.surface, request.params, request.palette);
        if (request.callback) {
          request.callback();
        }
      }
    }
  }
};

class FractalApp : public ci::app::App {
public:
  FractalApp() : renderer() {
    currentRenderParams = defaultRenderParameters;
    backRenderParams = defaultRenderParameters;
    currentSurface =
        ci::Surface32f(defaultScreenWidth, defaultScreenHeight, false);
    backSurface =
        ci::Surface32f(defaultScreenWidth, defaultScreenHeight, false);
    texture =
        ci::gl::Texture2d::create(defaultScreenWidth, defaultScreenHeight);
    renderInProgress = false;
    firstStaticRenderFlag = true;
    isZooming = false;
    zoomFactor = defaultZoomFactor;
    firstZoomRenderFlag = true;
    firstZoomRenderDone = false;
    currentDisplayedScale = currentRenderParams.scale;
    palette =
        generatePalette(defaultBaseColors, currentRenderParams.maxIterations);
    isDragging = false;
    needsTextureUpdate = false;
  }

  void setup() override { ImGui::Initialize(); }

  void update() override {
    if (isZooming) {
      if (firstZoomRenderFlag && !renderInProgress) {
        // Initiate the Zoom, we will prepare the two surfaces.
        clearSurface(currentSurface, ci::Color(0, 0, 0));
        clearSurface(backSurface, ci::Color(0, 0, 0));
        texture->update(currentSurface);
        renderInProgress = true;
        firstZoomRenderFlag = false;
        renderer.requestRender(
            currentSurface, currentRenderParams, palette, [this]() {
              backRenderParams = currentRenderParams;
              backRenderParams.scale = currentRenderParams.scale * zoomFactor;
              renderer.requestRender(backSurface, backRenderParams, palette,
                                     [this]() {
                                       renderInProgress = false;
                                       firstZoomRenderDone = true;
                                       zoomTimer.start();
                                     });
            });
      }
      if (firstZoomRenderDone) {
        // Adjust scale and request new renders after the first frame is ready
        currentDisplayedScale =
            currentRenderParams.scale *
            std::exp(zoomTimer.getSeconds() * std::log(zoomFactor));
        if ((currentDisplayedScale / currentRenderParams.scale) >= zoomFactor &&
            !renderInProgress) {
          std::swap(backSurface, currentSurface);
          std::swap(backRenderParams, currentRenderParams);
          backRenderParams.scale = currentRenderParams.scale * zoomFactor;
          renderInProgress = true;
          zoomTimer.start();
          renderer.requestRender(backSurface, backRenderParams, palette,
                                 [this]() { renderInProgress = false; });
        }
        texture->update(currentSurface);
      }
    } else {
      // Parameter changes allowed in static mode.
      bool parametersChanged = false;
      bool maxIterationsChanged = false;
      ImGui::Begin("Parameter Control");
      parametersChanged |= ImGui::InputDouble(
          "Constant X", &currentRenderParams.constantX, 0.01, 0.1, "%.6f");
      parametersChanged |= ImGui::InputDouble(
          "Constant Y", &currentRenderParams.constantY, 0.01, 0.1, "%.6f");
      parametersChanged |= ImGui::InputDouble(
          "Offset X", &currentRenderParams.offsetX, 0.1, 1.0, "%.3f");
      parametersChanged |= ImGui::InputDouble(
          "Offset Y", &currentRenderParams.offsetY, 0.1, 1.0, "%.3f");
      parametersChanged |= ImGui::InputDouble(
          "Scale", &currentRenderParams.scale, 0.01, 0.1, "%.4f");
      int previousMaxIterations = currentRenderParams.maxIterations;
      parametersChanged |= ImGui::InputInt(
          "Max Iterations", &currentRenderParams.maxIterations, 10, 100);
      if (currentRenderParams.maxIterations != previousMaxIterations) {
        maxIterationsChanged = true;
      }
      parametersChanged |= ImGui::InputInt(
          "Escape Radius Squared", &currentRenderParams.escapeRadiusSquared, 1);
      parametersChanged |= ImGui::InputInt(
          "Sampling Count", &currentRenderParams.samplingCount, 1);
      ImGui::End();
      if ((parametersChanged || firstStaticRenderFlag) && !renderInProgress) {
        if (parametersChanged) {
          clampRenderParameters(currentRenderParams);
          if (maxIterationsChanged) {
            palette = generatePalette(defaultBaseColors,
                                      currentRenderParams.maxIterations);
          }
        }
        if (firstStaticRenderFlag) {
          clearSurface(currentSurface, ci::Color(0, 0, 0));
          clearSurface(backSurface, ci::Color(0, 0, 0));
          texture->update(currentSurface);
          firstStaticRenderFlag = false;
        }
        renderInProgress = true;
        renderer.requestRender(backSurface, currentRenderParams, palette,
                               [this]() {
                                 std::lock_guard<std::mutex> lock(surfaceMutex);
                                 std::swap(backSurface, currentSurface);
                                 needsTextureUpdate = true;
                                 renderInProgress = false;
                               });
      }
      if (needsTextureUpdate) {
        std::lock_guard<std::mutex> lock(surfaceMutex);
        texture->update(currentSurface);
        needsTextureUpdate = false;
      }
    }
  }

  void draw() override {
    ci::gl::clear(ci::Color(0, 0, 0));
    if (isZooming) {
      if (firstZoomRenderDone) {
        ci::vec2 center(defaultScreenWidth * 0.5f, defaultScreenHeight * 0.5f);
        float zoomLevel = currentDisplayedScale / currentRenderParams.scale;
        ci::gl::pushModelMatrix();
        ci::gl::translate(center);
        ci::gl::scale(ci::vec2(zoomLevel));
        ci::gl::translate(-center);
        ci::gl::color(1, 1, 1, 1.0f);
        ci::gl::draw(texture);
        ci::gl::popModelMatrix();
      }
    } else {
      ci::gl::draw(texture);
    }
    ImGui::Render();
  }

  void keyDown(ci::app::KeyEvent event) override {
    bool needsUpdate = false;
    char key = event.getChar();
    if (key == 'q' || key == 'Q') {
      quit();
    } else if (key == 'r' || key == 'R') {
      currentRenderParams = defaultRenderParameters;
      backRenderParams = defaultRenderParameters;
      palette = generatePalette(defaultBaseColors,
                                defaultRenderParameters.maxIterations);
      clearSurface(currentSurface, ci::Color(0, 0, 0));
      clearSurface(backSurface, ci::Color(0, 0, 0));
      renderInProgress = false;
      firstStaticRenderFlag = true;
      isZooming = false;
      firstZoomRenderFlag = true;
      firstZoomRenderDone = false;
      currentDisplayedScale = defaultRenderParameters.scale;
      texture->update(currentSurface);

    } else if (event.getChar() == '+' && !isZooming && !renderInProgress) {
      currentRenderParams.scale *= zoomFactor;
      needsUpdate = true;
    } else if (event.getChar() == '-' && !isZooming && !renderInProgress) {
      currentRenderParams.scale /= zoomFactor;
      needsUpdate = true;
    } else if (event.getCode() == ci::app::KeyEvent::KEY_LEFT && !isZooming &&
               !renderInProgress) {
      currentRenderParams.offsetX -= 0.1f / currentRenderParams.scale;
      needsUpdate = true;
    } else if (event.getCode() == ci::app::KeyEvent::KEY_RIGHT && !isZooming &&
               !renderInProgress) {
      currentRenderParams.offsetX += 0.1f / currentRenderParams.scale;
      needsUpdate = true;
    } else if (event.getCode() == ci::app::KeyEvent::KEY_UP && !isZooming &&
               !renderInProgress) {
      currentRenderParams.offsetY -= 0.1f / currentRenderParams.scale;
      needsUpdate = true;
    } else if (event.getCode() == ci::app::KeyEvent::KEY_DOWN && !isZooming &&
               !renderInProgress) {
      currentRenderParams.offsetY += 0.1f / currentRenderParams.scale;
      needsUpdate = true;
    } else if (event.getCode() == ci::app::KeyEvent::KEY_SPACE) {
      if (isZooming) {
        firstStaticRenderFlag = true;
        currentRenderParams.scale = currentDisplayedScale;
      } else {
        firstZoomRenderFlag = true;
        firstZoomRenderDone = false;
      }
      isZooming = !isZooming;
    }
    if (needsUpdate) {
      renderInProgress = true;
      clampRenderParameters(currentRenderParams);
      renderer.requestRender(backSurface, currentRenderParams, palette,
                             [this]() {
                               std::lock_guard<std::mutex> lock(surfaceMutex);
                               std::swap(backSurface, currentSurface);
                               needsTextureUpdate = true;
                               renderInProgress = false;
                             });
    }
  }

  void mouseDown(ci::app::MouseEvent event) override {
    if (!isZooming) {
      isDragging = true;
      prevMousePos = event.getPos();
    }
  }

  void mouseDrag(ci::app::MouseEvent event) override {
    if (!isZooming && isDragging && !renderInProgress) {
      ci::vec2 currentMousePos = event.getPos();
      ci::vec2 delta = currentMousePos - prevMousePos;
      currentRenderParams.offsetX -=
          delta.x / (defaultScreenWidth * currentRenderParams.scale);
      currentRenderParams.offsetY -=
          delta.y / (defaultScreenHeight * currentRenderParams.scale);
      prevMousePos = currentMousePos;
      renderInProgress = true;
      clampRenderParameters(currentRenderParams);
      renderer.requestRender(backSurface, currentRenderParams, palette,
                             [this]() {
                               std::lock_guard<std::mutex> lock(surfaceMutex);
                               std::swap(backSurface, currentSurface);
                               needsTextureUpdate = true;
                               renderInProgress = false;
                             });
    }
  }

  void mouseUp(ci::app::MouseEvent event) override { isDragging = false; }

  void mouseWheel(ci::app::MouseEvent event) override {
    if (!isZooming && !renderInProgress) {
      currentRenderParams.scale *= (1.0f + event.getWheelIncrement() * 0.1f);
      renderInProgress = true;
      clampRenderParameters(currentRenderParams);
      renderer.requestRender(backSurface, currentRenderParams, palette,
                             [this]() {
                               std::lock_guard<std::mutex> lock(surfaceMutex);
                               std::swap(backSurface, currentSurface);
                               needsTextureUpdate = true;
                               renderInProgress = false;
                             });
    }
  }

private:
  RenderParameters currentRenderParams;
  RenderParameters backRenderParams;
  ci::Surface32f currentSurface;
  ci::Surface32f backSurface;
  ci::gl::Texture2dRef texture;
  Renderer renderer;
  bool renderInProgress;
  bool firstStaticRenderFlag;
  bool isZooming;
  double zoomFactor;
  bool firstZoomRenderFlag;
  bool firstZoomRenderDone;
  double currentDisplayedScale;
  ci::Timer zoomTimer;
  std::vector<ci::Color> palette;
  bool isDragging;
  ci::vec2 prevMousePos;
  std::mutex surfaceMutex;
  bool needsTextureUpdate;

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
  settings->setFrameRate(30);
  settings->setWindowSize(defaultScreenWidth, defaultScreenHeight);
  settings->setResizable(false);
}

CINDER_APP(FractalApp, ci::app::RendererGl, prepareSettings)
