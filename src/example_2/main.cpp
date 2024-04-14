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
        mouseSourceGrid = std::vector<std::vector<float>>(gridNumRows, std::vector<float>(gridNumCols, 0.0f));

        float angularVelocity = 0.5f;
        velocityGrid = generateVelocityField(gridNumRows, gridNumCols, angularVelocity);
    }

    void mouseDown(MouseEvent event) override
    {
        ivec2 mousePos = event.getPos();

        int x = mousePos.x / GRID_RESOLUTION;
        int y = mousePos.y / GRID_RESOLUTION;

        if (x >= 0 && x < mouseSourceGrid.size() && y >= 0 && y < mouseSourceGrid[0].size())
        {
            mouseSourceGrid[x][y] += 1.0f;
        }
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
        densityStep(densityGrid, densityGridOld, mouseSourceGrid, velocityGrid, DT);
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
    std::vector<std::vector<float>> mouseSourceGrid;
    std::vector<std::vector<glm::vec2>> velocityGrid;

    template <typename T>
    void addSource(std::vector<std::vector<T>> &targetGrid, const std::vector<std::vector<T>> &sourceGrid, float dt)
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

    template <typename T>
    void diffuse(std::vector<std::vector<T>> &targetGrid, const std::vector<std::vector<T>> &targetGridOld,
                 size_t max_iter, float diff, float dt)
    {
        size_t n = targetGrid.size();
        size_t m = targetGrid[0].size();

        for (size_t k = 0; k < max_iter; k++)
        {
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t j = 0; j < m; ++j)
                {
                    T sum = 0;
                    sum += (i > 0) ? targetGridOld[i - 1][j] : 0;     // Left
                    sum += (i < n - 1) ? targetGridOld[i + 1][j] : 0; // Right
                    sum += (j > 0) ? targetGridOld[i][j - 1] : 0;     // Up
                    sum += (j < m - 1) ? targetGridOld[i][j + 1] : 0; // Down

                    targetGrid[i][j] = (targetGridOld[i][j] + dt * diff * sum) / (1 + 4 * diff * dt);
                }
            }
        }
    }

    void advect(std::vector<std::vector<float>> &densityGrid, const std::vector<std::vector<float>> &densityGridOld,
                const std::vector<std::vector<vec2>> &velocityGrid, float dt)
    {
        float x, y;

        size_t n = densityGrid.size();
        size_t m = densityGrid[0].size();

        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < m; j++)
            {
                x = i - dt * velocityGrid[i][j].x;
                y = j - dt * velocityGrid[i][j].y;
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
    }

    void densityStep(std::vector<std::vector<float>> &densityGrid, std::vector<std::vector<float>> &densityGridOld,
                     const std::vector<std::vector<float>> mouseSourceGrid,
                     const std::vector<std::vector<vec2>> &velocityGrid, float dt)
    {
        addSource(densityGrid, mouseSourceGrid, dt);
        SWAP_GRIDS(densityGridOld, densityGrid);
        diffuse(densityGrid, densityGridOld, 20, 1.0f, dt);
        SWAP_GRIDS(densityGridOld, densityGrid);
        advect(densityGrid, densityGridOld, velocityGrid, dt);
    }

    /*
    ***************************
    * Temp functions          *
    ***************************
    */

    std::vector<std::vector<vec2>> generateVelocityField(int numRows, int numCols, float angularVelocity)
    {
        std::vector<std::vector<vec2>> velocityField(numRows, std::vector<vec2>(numCols, vec2(0.0f)));

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

                // Rotate the velocity vector
                float newAngle = angle + angularVelocity;
                vec2 newVelocity(cos(newAngle) * radius, sin(newAngle) * radius);

                velocityField[i][j] = newVelocity;
            }
        }

        return velocityField;
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
