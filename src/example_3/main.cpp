#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <immintrin.h>
#include <mutex>
#include <omp.h>

constexpr int screenWidth = 1000;
constexpr int screenHeight = 1000;
constexpr double initialConstantX = -0.7269;
constexpr double initialConstantY = 0.1889;
constexpr double offsetX = 0.0;
constexpr double offsetY = 0.0;
constexpr int maxIterations = 1000;
constexpr int escapeRadiusSquared = 4;
constexpr int samplingCount = 4;
constexpr float zoom_factor = 1.5f;

std::vector<ci::ColorA> generatePalette(int maxIterations) {
  std::vector<ci::ColorA> palette(maxIterations + 1);

  // Define key colors for gradient (in reverse order)
  const std::vector<ci::ColorA> baseColors = {
      ci::ColorA(0.7f, 0.7f, 0.8f, 1.0f), // Gray with slight purple tint
      ci::ColorA(0.6f, 0.4f, 0.8f, 1.0f), // Light Purple
      ci::ColorA(0.4f, 0.2f, 0.8f, 1.0f), // Purple-Blue
      ci::ColorA(0.2f, 0.3f, 0.9f, 1.0f)  // Blue
  };

  // Points inside set are dark blue
  palette[maxIterations] = ci::ColorA(0.1f, 0.1f, 0.3f, 1.0f);

  for (int i = 0; i < maxIterations; i++) {
    float t = (float)i / maxIterations;
    t = t * (baseColors.size() - 1);

    int idx = static_cast<int>(t);
    float fract = t - idx;

    const ci::ColorA &c1 = baseColors[idx];
    const ci::ColorA &c2 =
        baseColors[std::min(idx + 1, (int)baseColors.size() - 1)];

    palette[i] =
        ci::ColorA(ci::lerp(c1.r, c2.r, fract), ci::lerp(c1.g, c2.g, fract),
                   ci::lerp(c1.b, c2.b, fract), 1.0f);
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

void renderJulia(ci::Surface32f *surface, int screenWidth, int screenHeight,
                 double constantX, double constantY, double offsetX,
                 double offsetY, double scale, int maxIterations,
                 double escapeRadiusSquared, int samplingCount,
                 const std::vector<ci::ColorA> &palette) {

  double screenWidthHalf = screenWidth / 2.0;
  double screenHeightHalf = screenHeight / 2.0;
  double scaleInv = 1.0 / scale;

#pragma omp parallel for
  for (int y = 0; y < screenHeight; y += 1) {
    for (int x = 0; x < screenWidth; x += 4) {
      __m256d screenWidthHalfVec = _mm256_set1_pd(screenWidthHalf);
      __m256d scaleInvVec = _mm256_set1_pd(scaleInv);
      __m256d xVec = _mm256_set_pd(x + 3, x + 2, x + 1, x);
      __m256d xShifted = _mm256_sub_pd(xVec, screenWidthHalfVec);

      double accumulatedR[4] = {0}, accumulatedG[4] = {0};
      double accumulatedB[4] = {0}, accumulatedA[4] = {0};

      for (int sy = 0; sy < samplingCount; sy++) {
        for (int sx = 0; sx < samplingCount; sx++) {
          double subX = sx / static_cast<double>(samplingCount);
          double subY = sy / static_cast<double>(samplingCount);

          __m256d posX = _mm256_add_pd(
              _mm256_mul_pd(
                  _mm256_mul_pd(_mm256_add_pd(xShifted, _mm256_set1_pd(subX)),
                                scaleInvVec),
                  _mm256_set1_pd(1.0 / screenWidthHalf)),
              _mm256_set1_pd(offsetX));

          __m256d posY = _mm256_set1_pd((y + subY - screenHeightHalf) /
                                            screenWidthHalf * scaleInv +
                                        offsetY);

          __m256d constX = _mm256_set1_pd(constantX);
          __m256d constY = _mm256_set1_pd(constantY);
          __m256d escRadSq = _mm256_set1_pd(escapeRadiusSquared);

          __m256i iterations =
              computeJulia(posX, posY, constX, constY, escRadSq, maxIterations);

          int64_t iterCounts[4];
          _mm256_storeu_si256((__m256i *)iterCounts, iterations);

          for (int i = 0; i < 4; ++i) {
            if (x + i < screenWidth) {
              int idx =
                  std::min(iterCounts[i], static_cast<int64_t>(maxIterations));
              ci::ColorA color = palette[idx];
              accumulatedR[i] += color.r;
              accumulatedG[i] += color.g;
              accumulatedB[i] += color.b;
              accumulatedA[i] += color.a;
            }
          }
        }
      }

      double totalSamples = samplingCount * samplingCount;
      for (int i = 0; i < 4; ++i) {
        if (x + i < screenWidth) {
          ci::ColorA finalColor(
              accumulatedR[i] / totalSamples, accumulatedG[i] / totalSamples,
              accumulatedB[i] / totalSamples, accumulatedA[i] / totalSamples);
          surface->setPixel(ci::ivec2(x + i, y), finalColor);
        }
      }
    }
  }
}

class FractalApp : public ci::app::App {
public:
  FractalApp() {
    constantX = initialConstantX;
    constantY = initialConstantY;
    renderNeeded = true;
    isZooming = false;
    lastFrameTime = getElapsedSeconds();
    zoomTime = 0.0;
    baseScale = 1.0;
    targetScale = baseScale * zoom_factor;
    currentScale = baseScale;
    needsNewRender = true;
    workerRunning = true;

    surface = ci::Surface32f(screenWidth, screenHeight, false);
    zSurface = ci::Surface32f(screenWidth, screenHeight, false);
    texture = ci::gl::Texture2d::create(surface);
    zTexture = ci::gl::Texture2d::create(zSurface);
    colorPalette = generatePalette(maxIterations);

    worker = std::thread(&FractalApp::workerLoop, this);
  }

  void setup() override;
  void update() override;
  void draw() override;

  void keyDown(ci::app::KeyEvent event) override;

private:
  // State variables
  double constantX;
  double constantY;
  bool renderNeeded;
  bool isZooming;
  double lastFrameTime;
  double zoomTime;
  double baseScale;
  double targetScale;
  double currentScale;
  bool needsNewRender;
  bool workerRunning;

  // Surfaces and textures
  ci::Surface32f surface;
  ci::Surface32f zSurface;
  ci::gl::Texture2dRef texture;
  ci::gl::Texture2dRef zTexture;
  std::vector<ci::ColorA> colorPalette;

  // Threading
  std::mutex mtx;
  std::condition_variable cv;
  std::thread worker;

  // Helper Functions
  void resetZoomState();
  void updateZoom(double deltaTime);
  void startNewZoomRender();
  void renderSurface();

  // Worker Loop
  void workerLoop();
};

// Reset zoom variables to initial state
void FractalApp::resetZoomState() {
  isZooming = false;
  lastFrameTime = getElapsedSeconds();
  zoomTime = 0.0;
  baseScale = 1.0;
  targetScale = baseScale * zoom_factor;
  currentScale = baseScale;
}

void FractalApp::updateZoom(double deltaTime) {
  zoomTime += deltaTime;
  currentScale = baseScale * std::exp(zoomTime * std::log(zoom_factor));

  if ((currentScale / baseScale) >= zoom_factor && !needsNewRender) {
    startNewZoomRender();
  }
}

void FractalApp::startNewZoomRender() {
  std::lock_guard<std::mutex> lock(mtx);
  zTexture->update(zSurface);
  baseScale = targetScale;
  targetScale = baseScale * zoom_factor;
  zoomTime = 0.0;
  needsNewRender = true;
  renderNeeded = true;
  cv.notify_one();
}

void FractalApp::renderSurface() {
  renderJulia(&surface, screenWidth, screenHeight, constantX, constantY,
              offsetX, offsetY, currentScale, maxIterations,
              escapeRadiusSquared, samplingCount, colorPalette);
}

void FractalApp::workerLoop() {
  while (true) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [this] { return needsNewRender || !workerRunning; });

    if (!workerRunning)
      break;

    renderJulia(&zSurface, screenWidth, screenHeight, constantX, constantY,
                offsetX, offsetY, targetScale, maxIterations,
                escapeRadiusSquared, samplingCount, colorPalette);

    needsNewRender = false;
    renderNeeded = true;
  }
}

void FractalApp::setup() {
  ImGui::Initialize();
  renderSurface();
  texture->update(surface);
  zTexture->update(surface);
  renderNeeded = false;
};

void FractalApp::update() {
  std::cout << renderNeeded << std::endl;

  bool constantsChanged = false;

  ImGui::Begin("Parameters");
  double newConstantX = constantX;
  double newConstantY = constantY;
  if (ImGui::InputDouble("Constant X", &newConstantX, 0.01) ||
      ImGui::InputDouble("Constant Y", &newConstantY, 0.01)) {
    constantX = newConstantX;
    constantY = newConstantY;
    renderNeeded = true;
  }
  ImGui::End();

  if (constantsChanged) {
    std::lock_guard<std::mutex> lock(mtx);
    needsNewRender = true; // Signal the worker to render a new zoomed surface
    cv.notify_one();
  }

  double currentTime = getElapsedSeconds();
  double deltaTime = currentTime - lastFrameTime;
  lastFrameTime = currentTime;

  if (isZooming) {
    updateZoom(deltaTime);
  } else if (renderNeeded) {
    renderSurface();
    texture->update(surface);
    renderNeeded = false;
  }
};

void FractalApp::draw() {
  ci::gl::clear(ci::Color(0, 0, 0));
  if (isZooming) {
    ci::vec2 center(screenWidth * 0.5f, screenHeight * 0.5f);
    float zoomLevel = currentScale / baseScale;

    ci::gl::pushModelMatrix();
    ci::gl::translate(center);
    ci::gl::scale(ci::vec2(zoomLevel));
    ci::gl::translate(-center);
    ci::gl::color(1, 1, 1, 1.0f);
    ci::gl::draw(zTexture);
    ci::gl::popModelMatrix();
  } else {
    ci::gl::draw(texture);
  }
  ImGui::Render();
};

void FractalApp::keyDown(ci::app::KeyEvent event) {
  char key = event.getChar();
  if (key == 'q' || key == 'Q') {
    quit();
  } else if (key == 'r' || key == 'R') {
    resetZoomState();
    renderNeeded = true;
    std::lock_guard<std::mutex> lock(mtx);
    needsNewRender = true;
    cv.notify_one();
    renderSurface();
    texture->update(surface);
    zTexture->update(surface);
  } else if (key == 's' || key == 'S') {
    try {
      auto now = std::chrono::system_clock::now();
      auto timestamp = std::chrono::system_clock::to_time_t(now);
      std::stringstream ss;
      ss << "julia_"
         << std::put_time(std::localtime(&timestamp), "%Y%m%d_%H%M%S")
         << ".png";
      ci::writeImage(ss.str(), surface);
      std::cout << "Saved to: " << ss.str() << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error saving image: " << e.what() << std::endl;
    }
  } else if (event.getCode() == ci::app::KeyEvent::KEY_SPACE) {
    isZooming = !isZooming;
    renderNeeded = true;
  }
}

void prepareSettings(FractalApp::Settings *settings) {
  settings->setFrameRate(60);
  settings->setWindowSize(screenWidth, screenHeight);
  settings->setResizable(false);
}

CINDER_APP(FractalApp, ci::app::RendererGl, prepareSettings)
