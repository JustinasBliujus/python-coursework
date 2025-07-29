## Task Overview

**Goal:** Analyze the functionality of an artificial neuron and apply it to classify 2D data points into two binary classes.

### Implemented Features

1. **Data Generation**
   - 20 2D points (10 per class), linearly separable.
   - Visualized on an XY plot with different colors.

2. **Neuron Model**
   - Inputs: 2D feature points.
   - Activation functions:
     - Threshold (step)
     - Sigmoid (smooth, outputs between 0 and 1)
   - Output: 0 or 1, representing class.

3. **Manual Weight Search (No Learning Algorithm)**
   - Random generation of weights `w1`, `w2`, and bias `b`.
   - Search for 3 valid sets that correctly classify all data points.

4. **Visualization**
   - Plots with:
     - Data points,
     - 3 decision boundaries (from threshold activation),
     - Corresponding weight vectors (perpendicular to lines).
   - All elements color-coded and labeled.

## Output

- Matplotlib visualizations of:
  - Data distribution,
  - Class separation lines,
  - Weight vectors.
