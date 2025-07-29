
## Task Summary

### Goal

- Train CNNs to classify:
  - Images using **2D convolutions**
  - Keystroke time series using **1D convolutions**
- Perform experiments by modifying network architecture and hyperparameters
- Analyze performance via accuracy, loss, and confusion matrices

---
### Files listed and remarks

- `img.py` training and evaluating of a model to distinguish between Muffin vs Chihuahua → [Kaggle](https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification).
- A local file containing images should be created for `img.py` to work.
- `text.py` training and evaluating of a model for the data in `SPD.csv`
- `data_processing.py` functions for loading and processing data
- `error_analysis` functions for computing error
- `graphs.py` contains plotting functions
- `predict_and_evaluate.py` contains functions for prediction and evaluation

- For this task we were required to find an existing solution and adjust it to our needs.
- This solutions is based on → [Youtube](https://www.youtube.com/watch?v=jztwpsIzEGc) by Nicholas Renotte

- **CNN Details:**
  - Input: Resized color images
  - Output: Class probabilities via softmax
  - Architecture: Multiple 2D convolution + pooling + dropout layers

---

## Time Series Classification

- **Dataset:** CMU Keystroke Dynamics → [CMU dataset](https://www.cs.cmu.edu/~keystroke/)
- **Task:** Identify which of 30 users typed the password “tie5Roanl”
- **Data Format:** Each time series has 31 numerical features per record
- **CNN Details:**
  - Input: 1D feature sequences
  - Output: 30-class classification

---

## Features

For both models:
- Adjustable architecture (via code)
- Customizable hyperparameters:
  - Activation functions
  - Number of epochs
  - Learning rate
  - Batch size
  - Kernel/filter size
  - Dropout rate
  - Optimizer
  - Loss function

- Evaluation metrics:
  - Accuracy
  - Loss
  - Confusion matrix

---

## Experiments Conducted

1. **Architectural Variations**  
   - Tested at least 3 different CNN architectures per dataset
   - Goal: Find most accurate configuration

2. **Hyperparameter Tuning**
   - Dropout rate
   - Batch normalization
   - Activation functions
   - Optimizer types (e.g. Adam, SGD)
   - Learning rate influence on results

3. **Performance Analysis**
   - Plots:
     - Accuracy vs Epochs
     - Loss vs Epochs
   - Comparisons across:
     - Model complexity
     - Regularization methods
     - Training time

4. **Final Model Evaluation**
   - Best performing model tested on held-out test set
   - Confusion matrix created
   - 30 sample predictions shown (true vs predicted class)

---

## Result Files

- Accuracy/Loss plots
- Confusion matrix images
- Test result breakdowns
- Per-class accuracy metrics

---
