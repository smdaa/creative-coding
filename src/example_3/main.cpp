#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include <immintrin.h>
#include <omp.h>

using namespace ci;
using namespace ci::app;

struct fractalConfiguration
{
    std::string constantName;
    double constantX;
    double constantY;
    double offsetX;
    double offsetY;
    double scale;
    int maxIterations;
    double escapeRadiusSquared;
};

struct screenConfiguration
{
    int width;
    int height;
};

const std::map<std::string, std::pair<double, double>> fractalConstants = {
    {"San Marco", {-0.75, 0.0}},
    {"Douady's Rabbit", {-0.123, 0.745}},
    {"Siegel Disk", {-0.391, -0.587}},
    {"Feigenbaum Point", {-1.401155, 0.0}},
    {"Misiurewicz Point", {0.0, 1.0}},
};

const fractalConfiguration defaultFractalConfiguration = {"Dendrite", fractalConstants.at("Douady's Rabbit").first, fractalConstants.at("Douady's Rabbit").second, 0.0, 0.0, 1.0, 100, 4.0};

const screenConfiguration defaultScreenConfiguration = {1920, 1080};

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

Surface32f renderFractal(const fractalConfiguration &fractalConfig, const screenConfiguration &screenConfig)
{
    Surface32f surface(screenConfig.width, screenConfig.height, false);
#pragma omp parallel for
    for (int y = 0; y < screenConfig.height; ++y)
    {
        for (int x = 0; x < screenConfig.width; x += 4)
        {
            __m256d posX = _mm256_set_pd(
                (x + 3 - screenConfig.width / 2.0) / (screenConfig.width / 2.0f) / fractalConfig.scale + fractalConfig.offsetX,
                (x + 2 - screenConfig.width / 2.0) / (screenConfig.width / 2.0f) / fractalConfig.scale + fractalConfig.offsetX,
                (x + 1 - screenConfig.width / 2.0) / (screenConfig.width / 2.0f) / fractalConfig.scale + fractalConfig.offsetX,
                (x - screenConfig.width / 2.0) / (screenConfig.width / 2.0f) / fractalConfig.scale + fractalConfig.offsetX);

            __m256d posY = _mm256_set1_pd(
                (y - screenConfig.height / 2.0) / (screenConfig.width / 2.0f) / fractalConfig.scale + fractalConfig.offsetY);

            __m256d constX = _mm256_set1_pd(fractalConfig.constantX);
            __m256d constY = _mm256_set1_pd(fractalConfig.constantY);
            __m256d escRadSq = _mm256_set1_pd(fractalConfig.escapeRadiusSquared);

            __m256i iterations = computeJulia(posX, posY, constX, constY, escRadSq, fractalConfig.maxIterations);

            int64_t iterCounts[4];
            _mm256_storeu_si256((__m256i *)iterCounts, iterations);

            for (int i = 0; i < 4; ++i)
            {
                if (x + i < screenConfig.width)
                {
                    Color color = Color(iterCounts[i] / (float)fractalConfig.maxIterations, 0.0f, 0.0f);
                    surface.setPixel(ivec2(x + i, y), color);
                }
            }
        }
    }

    return surface;
}

class FractalApp : public App
{
public:
    void setup() override;
    void update() override;
    void draw() override;
    void keyDown(ci::app::KeyEvent event) override;
    void resize() override;

    void mouseDown(ci::app::MouseEvent event) override;
    void mouseDrag(ci::app::MouseEvent event) override;
    void mouseWheel(ci::app::MouseEvent event) override;
    void mouseUp(ci::app::MouseEvent event) override;

private:
    screenConfiguration mScreenConfiguration = defaultScreenConfiguration;
    fractalConfiguration mFractalConfiguration = defaultFractalConfiguration;
    bool mNeedsUpdate = true;
    gl::Texture2dRef mFractalTexture;
    bool mIsDragging = false;
    vec2 mPrevMousePos;
    bool mIsZooming = false;
    int mFps = 0;
    double mLastUpdateTime = 0.0;
};

void FractalApp::setup()
{
    setWindowSize(mScreenConfiguration.width, mScreenConfiguration.height);
    ImGui::Initialize();
}

void FractalApp::update()
{
    std::string constantName = mFractalConfiguration.constantName;
    double constantX = mFractalConfiguration.constantX;
    double constantY = mFractalConfiguration.constantY;
    double offsetX = mFractalConfiguration.offsetX;
    double offsetY = mFractalConfiguration.offsetY;
    double scale = mFractalConfiguration.scale;
    int maxIterations = mFractalConfiguration.maxIterations;
    double escapeRadiusSquared = mFractalConfiguration.escapeRadiusSquared;

    ImGui::Begin("Parameters");
    ImGui::Text("Fractal configuration");
    ImGui::InputDouble("Constant X", &constantX);
    ImGui::InputDouble("Constant Y", &constantY);
    ImGui::InputDouble("Offset X", &offsetX);
    ImGui::InputDouble("Offset Y", &offsetY);
    ImGui::InputDouble("Scale", &scale);
    ImGui::InputInt("Max iterations", &maxIterations, 50);
    ImGui::InputDouble("Escape radius squared", &escapeRadiusSquared);
    if (ImGui::BeginCombo("Fractal Constants", constantName.c_str()))
    {
        for (const auto &[name, constant] : fractalConstants)
        {
            bool isSelected = (constantName == name);
            if (ImGui::Selectable(name.c_str(), isSelected))
            {
                constantName = name;
            }
            if (isSelected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }

    ImGui::End();

    if (constantX != mFractalConfiguration.constantX || constantY != mFractalConfiguration.constantY)
    {
        bool isInSet = false;
        for (const auto &[name, constant] : fractalConstants)
        {
            if (constant.first == constantX && constant.second == constantY)
            {
                constantName = name;
                isInSet = true;
                break;
            }
        }
        if (!isInSet)
        {
            constantName = "Custom";
        }
    }

    if (constantName != mFractalConfiguration.constantName && constantName != "Custom")
    {
        constantX = fractalConstants.at(constantName).first;
        constantY = fractalConstants.at(constantName).second;
    }

    if (constantX != mFractalConfiguration.constantX || constantY != mFractalConfiguration.constantY || offsetX != mFractalConfiguration.offsetX || offsetY != mFractalConfiguration.offsetY || scale != mFractalConfiguration.scale || maxIterations != mFractalConfiguration.maxIterations || escapeRadiusSquared != mFractalConfiguration.escapeRadiusSquared)
    {
        mFractalConfiguration.constantName = constantName;
        mFractalConfiguration.constantX = constantX;
        mFractalConfiguration.constantY = constantY;
        mFractalConfiguration.offsetX = offsetX;
        mFractalConfiguration.offsetY = offsetY;
        mFractalConfiguration.scale = scale;
        mFractalConfiguration.maxIterations = maxIterations;
        mFractalConfiguration.escapeRadiusSquared = escapeRadiusSquared;
        mNeedsUpdate = true;
    }

    if (mIsZooming)
    {
        mFractalConfiguration.scale *= 1.01f;
        mNeedsUpdate = true;
    }

    // Calculate FPS
    double currentTime = getElapsedSeconds();
    double deltaTime = currentTime - mLastUpdateTime;
    mLastUpdateTime = currentTime;
    if (deltaTime > 0.0)
    {
        mFps = (int)(1.0f / deltaTime);
    }
}

void FractalApp::draw()
{
    gl::clear(Color(0, 0, 0));
    if (mNeedsUpdate)
    {
        Surface32f fractalSurface = renderFractal(mFractalConfiguration, mScreenConfiguration);
        mFractalTexture = gl::Texture2d::create(fractalSurface);
        mNeedsUpdate = false;
    }
    gl::draw(mFractalTexture);
    ImGui::Begin("Performance");
    ImGui::Text("FPS: %d", mFps);
    ImGui::End();
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
        mFractalConfiguration = defaultFractalConfiguration;
        mNeedsUpdate = true;
        mIsZooming = false;
    }
    else if (event.getChar() == '+')
    {
        mFractalConfiguration.scale *= 1.1f;
        mNeedsUpdate = true;
    }
    else if (event.getChar() == '-')
    {
        mFractalConfiguration.scale /= 1.1f;
        mNeedsUpdate = true;
    }
    else if (event.getCode() == KeyEvent::KEY_LEFT)
    {
        mFractalConfiguration.offsetX -= 0.1f / mFractalConfiguration.scale;
        mNeedsUpdate = true;
    }
    else if (event.getCode() == KeyEvent::KEY_RIGHT)
    {
        mFractalConfiguration.offsetX += 0.1f / mFractalConfiguration.scale;
        mNeedsUpdate = true;
    }
    else if (event.getCode() == KeyEvent::KEY_UP)
    {
        mFractalConfiguration.offsetY -= 0.1f / mFractalConfiguration.scale;
        mNeedsUpdate = true;
    }
    else if (event.getCode() == KeyEvent::KEY_DOWN)
    {
        mFractalConfiguration.offsetY += 0.1f / mFractalConfiguration.scale;
        mNeedsUpdate = true;
    }
    else if (event.getCode() == KeyEvent::KEY_SPACE)
    {
        mIsZooming = !mIsZooming;
        mNeedsUpdate = true;
    }
}

void FractalApp::resize()
{
    mScreenConfiguration.width = getWindowWidth();
    mScreenConfiguration.height = getWindowHeight();
    mNeedsUpdate = true;
}

void FractalApp::mouseDown(MouseEvent event)
{
    mIsDragging = true;
    mPrevMousePos = event.getPos();
}

void FractalApp::mouseDrag(MouseEvent event)
{
    if (mIsDragging)
    {
        vec2 currentMousePos = event.getPos();
        vec2 delta = currentMousePos - mPrevMousePos;
        mFractalConfiguration.offsetX -= delta.x / (mScreenConfiguration.width * mFractalConfiguration.scale);
        mFractalConfiguration.offsetY -= delta.y / (mScreenConfiguration.height * mFractalConfiguration.scale);
        mPrevMousePos = currentMousePos;
        mNeedsUpdate = true;
    }
}

void FractalApp::mouseWheel(MouseEvent event)
{
    float zoomFactor = event.getWheelIncrement();
    mFractalConfiguration.scale *= (1.0f + zoomFactor * 0.1f);
    mNeedsUpdate = true;
}

void FractalApp::mouseUp(MouseEvent event)
{
    mIsDragging = false;
}

void prepareSettings(FractalApp::Settings *settings)
{
    settings->setFrameRate(60);
}

CINDER_APP(FractalApp, RendererGl, prepareSettings)
