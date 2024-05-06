#include "FluidGrid.hpp"

namespace FluidGridUtils {

void addSource(int numRows, int numColumns,
               std::vector<std::vector<float>> &grid,
               const std::vector<std::vector<float>> &sourceGrid,
               float timeStep) {
//#pragma omp parallel for
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numColumns; ++j) {
      grid[i][j] += sourceGrid[i][j] * timeStep;
    }
  }
}

void setBounds(int numRows, int numColumns,
               std::vector<std::vector<float>> &grid, int b) {
//#pragma omp parallel for
  for (int i = 1; i < numRows - 1; ++i) {
    grid[i][0] = (b == 2) ? -grid[i][1] : grid[i][1];
    grid[i][numColumns - 1] =
        (b == 2) ? -grid[i][numColumns - 2] : grid[i][numColumns - 2];
  }
//#pragma omp parallel for
  for (int j = 1; j < numColumns - 1; ++j) {
    grid[0][j] = (b == 1) ? -grid[1][j] : grid[1][j];
    grid[numRows - 1][j] =
        (b == 1) ? -grid[numRows - 2][j] : grid[numRows - 2][j];
  }
  grid[0][0] = 0.5 * (grid[1][0] + grid[0][1]);
  grid[0][numColumns - 1] =
      0.5 * (grid[1][numColumns - 1] + grid[0][numColumns - 2]);
  grid[numRows - 1][0] = 0.5 * (grid[numRows - 1][1] + grid[numRows - 2][0]);
  grid[numRows - 1][numColumns - 1] = 0.5 * (grid[numRows - 1][numColumns - 2] +
                                             grid[numRows - 2][numColumns - 1]);
}

void diffuse(int numRows, int numColumns,
             std::vector<std::vector<float>> &outGrid,
             const std::vector<std::vector<float>> &inGrid,
             int gaussSeidelIterations, float factor, int b, float timeStep) {
  float a = timeStep * factor * numRows * numColumns;
  float denominator = 1 + 4 * a;
  for (int k = 0; k < gaussSeidelIterations; ++k) {
//#pragma omp parallel for
    for (int i = 0; i < numRows; ++i) {
      for (int j = 0; j < numColumns; ++j) {
        float sum = 0.0f;
        sum += (i > 0) ? inGrid[i - 1][j] : 0.0f;              // Left
        sum += (i < numRows - 1) ? inGrid[i + 1][j] : 0.0f;    // Right
        sum += (j > 0) ? inGrid[i][j - 1] : 0.0f;              // Up
        sum += (j < numColumns - 1) ? inGrid[i][j + 1] : 0.0f; // Down
        outGrid[i][j] = (inGrid[i][j] + a * sum) / (denominator);
      }
    }
    setBounds(numRows, numColumns, outGrid, b);
  }
}

void advect(int numRows, int numColumns,
            std::vector<std::vector<float>> &outGrid,
            const std::vector<std::vector<float>> &inGrid,
            const std::vector<std::vector<float>> &velocityGridX,
            const std::vector<std::vector<float>> &velocityGridY, int b,
            float timeStep) {
  float dtRatio = timeStep * (std::max(numRows, numColumns) - 1);
//#pragma omp parallel for
  for (int i = 0; i < numRows; ++i) {
    for (int j = 0; j < numColumns; ++j) {
      float x = i - dtRatio * velocityGridX[i][j];
      float y = j - dtRatio * velocityGridY[i][j];
      x = std::max(0.5f, std::min(static_cast<float>(numRows) - 1.5f, x));
      y = std::max(0.5f, std::min(static_cast<float>(numColumns) - 1.5f, y));
      int x0 = (int)x;
      int y0 = (int)y;
      int x1 = std::min(x0 + 1, (int)numRows - 1);
      int y1 = std::min(y0 + 1, (int)numColumns - 1);
      float sx1 = x - x0;
      float sx0 = 1.0f - sx1;
      float sy1 = y - y0;
      float sy0 = 1.0f - sy1;
      outGrid[i][j] = sx0 * (sy0 * inGrid[x0][y0] + sy1 * inGrid[x0][y1]) +
                      sx1 * (sy0 * inGrid[x1][y0] + sy1 * inGrid[x1][y1]);
    }
  }
  setBounds(numRows, numColumns, outGrid, b);
}

void project(int numRows, int numColumns,
             std::vector<std::vector<float>> &velocityGridX,
             std::vector<std::vector<float>> &velocityGridY,
             std::vector<std::vector<float>> &p,
             std::vector<std::vector<float>> &div, int gaussSeidelIterations) {
//#pragma omp parallel for
  for (int i = 1; i < numRows - 1; ++i) {
    for (int j = 1; j < numColumns - 1; ++j) {
      div[i][j] = -0.5 * (velocityGridX[i + 1][j] - velocityGridX[i - 1][j] +
                          velocityGridY[i][j + 1] - velocityGridY[i][j - 1]);
      p[i][j] = 0.0;
    }
  }
  setBounds(numRows, numColumns, div, 0);
  setBounds(numRows, numColumns, p, 0);
  for (int k = 0; k < gaussSeidelIterations; ++k) {
//#pragma omp parallel for
    for (int i = 1; i < numRows - 1; ++i) {
      for (int j = 1; j < numColumns - 1; ++j) {
        p[i][j] = (div[i][j] + p[i - 1][j] + p[i + 1][j] + p[i][j - 1] +
                   p[i][j + 1]) /
                  4;
      }
    }
    setBounds(numRows, numColumns, p, 0);
  }
//#pragma omp parallel for
  for (int i = 1; i < numRows - 1; ++i) {
    for (int j = 1; j < numColumns - 1; ++j) {
      velocityGridX[i][j] -= 0.5 * (p[i + 1][j] - p[i - 1][j]);
      velocityGridY[i][j] -= 0.5 * (p[i][j + 1] - p[i][j - 1]);
    }
  }
  setBounds(numRows, numColumns, velocityGridX, 1);
  setBounds(numRows, numColumns, velocityGridY, 2);
}
} // namespace FluidGridUtils

FluidGrid::FluidGrid(int _numRows, int _numColumns, int _gridResolution)
    : numRows(_numRows), numColumns(_numColumns),
      gridResolution(_gridResolution) {
  densityGrid = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  densityGridOld = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocityGridX = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocityGridXOld = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocityGridY = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocityGridYOld = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  densitySourceGrid = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocitySourceGridX = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocitySourceGridY = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
}

void FluidGrid::stepDensity(int diffusionFactor, int gaussSeidelIterations,
                            float timeStep) {
  FluidGridUtils::addSource(numRows, numColumns, densityGrid, densitySourceGrid,
                            timeStep);
  FluidGridUtils::diffuse(numRows, numColumns, densityGridOld, densityGrid,
                          gaussSeidelIterations, diffusionFactor, 0, timeStep);
  FluidGridUtils::advect(numRows, numColumns, densityGrid, densityGridOld,
                         velocityGridX, velocityGridY, 0, timeStep);
  densitySourceGrid = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
}

void FluidGrid::stepVelocity(int viscosityFactor, int gaussSeidelIterations,
                             float timeStep) {
  FluidGridUtils::addSource(numRows, numColumns, velocityGridX,
                            velocitySourceGridX, timeStep);
  FluidGridUtils::addSource(numRows, numColumns, velocityGridY,
                            velocitySourceGridY, timeStep);
  FluidGridUtils::diffuse(numRows, numColumns, velocityGridXOld, velocityGridX,
                          gaussSeidelIterations, viscosityFactor, 1, timeStep);
  FluidGridUtils::diffuse(numRows, numColumns, velocityGridYOld, velocityGridY,
                          gaussSeidelIterations, viscosityFactor, 2, timeStep);
  FluidGridUtils::project(numRows, numColumns, velocityGridXOld,
                          velocityGridYOld, velocityGridX, velocityGridY,
                          gaussSeidelIterations);
  FluidGridUtils::advect(numRows, numColumns, velocityGridX, velocityGridXOld,
                         velocityGridXOld, velocityGridYOld, 1, timeStep);
  FluidGridUtils::advect(numRows, numColumns, velocityGridY, velocityGridYOld,
                         velocityGridXOld, velocityGridYOld, 2, timeStep);
  FluidGridUtils::project(numRows, numColumns, velocityGridX, velocityGridY,
                          velocityGridXOld, velocityGridYOld,
                          gaussSeidelIterations);
  velocitySourceGridX = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
  velocitySourceGridY = std::vector<std::vector<float>>(
      numRows, std::vector<float>(numColumns, 0.0f));
}
