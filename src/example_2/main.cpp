#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

using namespace ci;
using namespace ci::app;

#define SWAP_GRIDS(grid, gridOld)                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        auto temp = grid;                                                                                              \
        grid = gridOld;                                                                                                \
        gridOld = temp;                                                                                                \
    } while (0)
#define WINDOW_WIDTH 1000
#define WINDOW_HEIGHT 1000
#define BG_COLOR ColorA(0.0f, 0.0f, 0.0f)
#define GRID_RESOLUTION 10
#define DT 0.1

class SmokeApp : public App
{
  public:
    void setup() override
    {
        setWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);

        gridNumRows = WINDOW_HEIGHT / GRID_RESOLUTION;
        gridNumCols = WINDOW_WIDTH / GRID_RESOLUTION;

        densityGrid = std::vector<std::vector<float>>(gridNumRows, std::vector<float>(gridNumCols, 0.0f));
        densityGridOld = std::vector<std::vector<float>>(gridNumRows, std::vector<float>(gridNumCols, 0.0f));
        for (size_t i = 0; i < gridNumRows; i++)
        {
            for (size_t j = 0; j < gridNumCols; j++)
            {
                densityGridOld[i][j] = rand() / (float)RAND_MAX;
            }
        }

        velocityGridX = std::vector<std::vector<float>>(gridNumRows, std::vector<float>(gridNumCols, 0.0f));
        velocityGridY = std::vector<std::vector<float>>(gridNumRows, std::vector<float>(gridNumCols, 0.0f));

        auto velocityFields = generateVelocityField(gridNumRows, gridNumCols, 0.5f);
        velocityGridXOld = velocityFields.first;
        velocityGridYOld = velocityFields.second;
    }

    void keyDown(KeyEvent event) override
    {
        if (event.getChar() == 'q' || event.getChar() == 'Q')
        {
            quit();
        }
    }

    void update() override
    {
        densityStep(densityGrid, densityGridOld, velocityGridX, velocityGridY, 1.0, DT);
        // velocityStep(velocityGrid, velocityGridOld, 0.1, DT);
    }

    void draw() override
    {
        gl::clear(BG_COLOR);
        drawDensity();
    }

  private:
    int gridNumRows;
    int gridNumCols;
    std::vector<std::vector<float>> densityGrid;
    std::vector<std::vector<float>> densityGridOld;
    std::vector<std::vector<float>> velocityGridX;
    std::vector<std::vector<float>> velocityGridXOld;
    std::vector<std::vector<float>> velocityGridY;
    std::vector<std::vector<float>> velocityGridYOld;

    void addSource(std::vector<std::vector<float>> &targetGrid, const std::vector<std::vector<float>> &sourceGrid,
                   float dt)
    {
        size_t n = targetGrid.size();
        size_t m = targetGrid[0].size();

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < m; ++j)
            {
                targetGrid[i][j] += dt * sourceGrid[i][j];
            }
        }
    }

    void diffuse(std::vector<std::vector<float>> &targetGrid, const std::vector<std::vector<float>> &targetGridOld,
                 size_t gaussSeidelIterations, float diffusionFactor, int b, float dt)
    {
        size_t n = targetGrid.size();
        size_t m = targetGrid[0].size();

        for (size_t k = 0; k < gaussSeidelIterations; ++k)
        {
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                    float sum = 0.0f;
                    sum += (i > 0) ? targetGridOld[i - 1][j] : 0.0f;     // Left
                    sum += (i < n - 1) ? targetGridOld[i + 1][j] : 0.0f; // Right
                    sum += (j > 0) ? targetGridOld[i][j - 1] : 0.0f;     // Up
                    sum += (j < m - 1) ? targetGridOld[i][j + 1] : 0.0f; // Down

                    targetGrid[i][j] =
                        (targetGridOld[i][j] + dt * diffusionFactor * sum) / (1 + 4 * diffusionFactor * dt);
                }
            }
            setBoundary(b, targetGrid);
        }
    }

    void advect(std::vector<std::vector<float>> &densityGrid, const std::vector<std::vector<float>> &densityGridOld,
                const std::vector<std::vector<float>> &velocityGridX,
                const std::vector<std::vector<float>> &velocityGridY, int b, float dt)
    {
        float x, y;

        size_t n = densityGrid.size();
        size_t m = densityGrid[0].size();

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                x = i - dt * velocityGridX[i][j];
                y = j - dt * velocityGridY[i][j];
                x = std::max(0.0f, std::min((float)n - 1.0f, x));
                y = std::max(0.0f, std::min((float)m - 1.0f, y));

                int x0 = (int)x;
                int y0 = (int)y;
                int x1 = std::min(x0 + 1, (int)n - 1);
                int y1 = std::min(y0 + 1, (int)m - 1);

                float sx1 = x - x0;
                float sx0 = 1.0f - sx1;
                float sy1 = y - y0;
                float sy0 = 1.0f - sy1;

                densityGrid[i][j] = sx0 * (sy0 * densityGridOld[x0][y0] + sy1 * densityGridOld[x0][y1]) +
                                    sx1 * (sy0 * densityGridOld[x1][y0] + sy1 * densityGridOld[x1][y1]);
            }
        }
        setBoundary(b, densityGrid);
    }

    void setBoundary(int b, std::vector<std::vector<float>> &targetGrid)
    {
        size_t n = densityGrid.size();
        size_t m = densityGrid[0].size();
        for (size_t i = 1; i < n - 1; i++)
        {
            targetGrid[i][0] = (b == 2) ? -targetGrid[i][1] : targetGrid[i][1];
            targetGrid[i][m - 1] = (b == 2) ? -targetGrid[i][m - 2] : targetGrid[i][m - 2];
        }

        for (size_t j = 1; j < m - 1; j++)
        {
            targetGrid[0][j] = (b == 1) ? -targetGrid[1][j] : targetGrid[1][j];
            targetGrid[n - 1][j] = (b == 1) ? -targetGrid[n - 2][j] : targetGrid[n - 2][j];
        }

        targetGrid[0][0] = 0.5 * (targetGrid[1][0] + targetGrid[0][1]);
        targetGrid[0][m - 1] = 0.5 * (targetGrid[1][m - 1] + targetGrid[0][m - 2]);
        targetGrid[n - 1][0] = 0.5 * (targetGrid[n - 1][1] + targetGrid[n - 2][0]);
        targetGrid[n - 1][m - 1] = 0.5 * (targetGrid[n - 1][m - 2] + targetGrid[n - 2][m - 1]);
    }

    void densityStep(std::vector<std::vector<float>> &densityGrid, std::vector<std::vector<float>> &densityGridOld,
                     const std::vector<std::vector<float>> &velocityGridX,
                     const std::vector<std::vector<float>> &velocityGridY, float diffusion, float dt)
    {
        addSource(densityGrid, densityGridOld, dt);
        SWAP_GRIDS(densityGridOld, densityGrid);
        diffuse(densityGrid, densityGridOld, 20, diffusion, 0, dt);
        SWAP_GRIDS(densityGridOld, densityGrid);
        advect(densityGrid, densityGridOld, velocityGridX, velocityGridY, 0, dt);
    }

    void velocityStep(std::vector<std::vector<float>> velocityGridX, std::vector<std::vector<float>> velocityGridY,
                      std::vector<std::vector<float>> velocityGridXOld,
                      std::vector<std::vector<float>> velocityGridYOld, float viscosityFactor, float dt)
    {
        addSource(velocityGridX, velocityGridXOld, dt);
        addSource(velocityGridY, velocityGridYOld, dt);
        SWAP_GRIDS(velocityGridXOld, velocityGridX);
        diffuse(velocityGridX, velocityGridXOld, 20, viscosityFactor, 1, dt);
        SWAP_GRIDS(velocityGridYOld, velocityGridY);
        diffuse(velocityGridY, velocityGridYOld, 20, viscosityFactor, 2, dt);
    }

    /*
    ***************************
    * Temp functions          *
    ***************************
    */

    std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> generateVelocityField(
        int numRows, int numCols, float angularVelocity)
    {
        std::vector<std::vector<float>> velocityFieldX(numRows, std::vector<float>(numCols, 0.0f));
        std::vector<std::vector<float>> velocityFieldY(numRows, std::vector<float>(numCols, 0.0f));

        float centerX = numRows / 2.0f;
        float centerY = numCols / 2.0f;

        for (int i = 0; i < numRows; ++i)
        {
            for (int j = 0; j < numCols; ++j)
            {
                float dx = i - centerX;
                float dy = j - centerY;

                float angle = atan2(dy, dx);
                float radius = sqrt(dx * dx + dy * dy);

                // Rotate the angle
                float newAngle = angle + angularVelocity;

                // Calculate the velocity components
                float velocityX = cos(newAngle) * radius;
                float velocityY = sin(newAngle) * radius;

                velocityFieldX[i][j] = velocityX;
                velocityFieldY[i][j] = velocityY;
            }
        }

        return {velocityFieldX, velocityFieldY};
    }

    void drawDensity()
    {
        // Set up drawing parameters
        gl::color(ColorA(1.0f, 1.0f, 1.0f)); // White color for density
        gl::enableAlphaBlending();

        // Calculate cell size
        float cellWidth = (float)getWindowWidth() / gridNumCols;
        float cellHeight = (float)getWindowHeight() / gridNumRows;

        // Draw each cell with its density value
        for (int i = 0; i < gridNumRows; ++i)
        {
            for (int j = 0; j < gridNumCols; ++j)
            {
                // Calculate cell position
                float x = j * cellWidth;
                float y = i * cellHeight;

                // Calculate density value as grayscale color
                float density = densityGrid[i][j];
                ColorA densityColor(density, density, density, density);

                // Draw cell
                gl::color(densityColor);
                gl::drawSolidRect(Rectf(x, y, x + cellWidth, y + cellHeight));
            }
        }

        gl::disableAlphaBlending();
    }
};

void prepareSettings(SmokeApp::Settings *settings)
{
    settings->setResizable(false);
}

CINDER_APP(SmokeApp, RendererGl, prepareSettings)
