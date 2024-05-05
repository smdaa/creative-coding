
#ifndef FLUIDGRID_HPP
#define FLUIDGRID_HPP

#include <omp.h>
#include <vector>

class FluidGrid {

public:
  int numRows;
  int numColumns;
  int gridResolution;
  std::vector<std::vector<float>> densityGrid;
  std::vector<std::vector<float>> densityGridOld;
  std::vector<std::vector<float>> velocityGridX;
  std::vector<std::vector<float>> velocityGridXOld;
  std::vector<std::vector<float>> velocityGridY;
  std::vector<std::vector<float>> velocityGridYOld;
  std::vector<std::vector<float>> densitySourceGrid;
  std::vector<std::vector<float>> velocitySourceGridX;
  std::vector<std::vector<float>> velocitySourceGridY;

  FluidGrid(int _numRows, int _numColumns, int _gridResolution);

  void stepDensity(int diffusionFactor, int gaussSeidelIterations,
                   float timeStep);
  void stepVelocity(int viscosityFactor, int gaussSeidelIterations,
                    float timeStep);
};

namespace FluidGridUtils {
void addSource(int numRows, int numColumns,
               std::vector<std::vector<float>> &grid,
               const std::vector<std::vector<float>> &sourceGrid,
               float timeStep);
void setBounds(int numRows, int numColumns,
               std::vector<std::vector<float>> &grid, int b);
void diffuse(int numRows, int numColumns,
             std::vector<std::vector<float>> &outGrid,
             const std::vector<std::vector<float>> &inGrid,
             int gaussSeidelIterations, float factor, int b, float timeStep);
void advect(int numRows, int numColumns,
            std::vector<std::vector<float>> &outGrid,
            const std::vector<std::vector<float>> &inGrid,
            const std::vector<std::vector<float>> &velocityGridX,
            const std::vector<std::vector<float>> &velocityGridY, int b,
            float timeStep);
void project(int numRows, int numColumns,
             std::vector<std::vector<float>> &velocityGridX,
             std::vector<std::vector<float>> &velocityGridY,
             std::vector<std::vector<float>> &p,
             std::vector<std::vector<float>> &div, int gaussSeidelIterations);
} // namespace FluidGridUtils
#endif // FLUIDGRID_HPP
