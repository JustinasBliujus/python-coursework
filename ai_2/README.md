## Task Summary

**Goal:** Train and evaluate a single neuron using sigmoid activation to classify tumors as benign (0) or malignant (1).

### Files listed

- `AI2_stochastinis.py` training and evaluating using stochastic descent
- `AI2_paketinis.py` training and evaluating using batch gradient descent
- `data_processing.py` functions for loading and processing data
- `error_analysis` functions for computing error
- `graphs.py` contains plotting functions
- `predict_and_evaluate.py` contains functions for prediction and evaluation
  
### Dataset

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)
- **Used file:** `breast-cancer-wisconsin.data`
- **Preprocessing:**
  - Removed missing entries (`?`)
  - Removed ID column
  - Converted labels: `2 → 0`, `4 → 1`
  - Shuffled data
  - Split into training (80%), validation (10%), and testing (10%) sets

### Neuron Model

- **Activation function:** Sigmoid
- **Output range:** (0, 1), rounded to nearest integer (0 or 1) for classification
- **Input features:** 9
- **Output:** Binary class (benign or malignant)

### Learning Methods

1. **Batch Gradient Descent (BGD)**
2. **Stochastic Gradient Descent (SGD)**

### Outputs and Results

- Final weights and bias after training
- Error per epoch:
  - Training
  - Validation
- Final test error
- Accuracy per epoch (training & validation)
- Final test accuracy
- Training time
- Predictions for test set with actual vs predicted classes

### Experimental Analysis

- Plotted:
  - Error vs Epochs (line graph)
  - Accuracy vs Epochs (line graph)
- Tested effect of:
  - Different learning rates (min. 3 values)
  - BGD vs SGD on performance and speed
- Compared:
  - Accuracy
  - Error
  - Training time for BGD vs SGD

