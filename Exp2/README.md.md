# Experiment 2: Multi-Layer Perceptron (MLP) for Multi-Class Image Classification

## Objective
Implement a Multi-Layer Perceptron (MLP) architecture using TensorFlow/Keras to perform multi-class image classification on the Fashion-MNIST dataset. This module details extensive structural design, automated model optimization via hyperparameter sweeps, evaluation comparisons, and performance metrics tracking.

## Technical Tasks Executed
1. **Dataset Intake:** Loaded Fashion-MNIST split partitions, performed dimensional validation, and generated visual verification samples.
2. **Feature Optimization:** Flattened matrices from $28 \times 28 \to 784$ shapes, scaled pixel values to $[0, 1]$, and converted classification targets to one-hot vector representations.
3. **Baseline Architecture Construction:** Built a sequential configuration matching `784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`.
4. **Automated Hyperparameter Sweep:** Applied an cross-validated `RandomizedSearchCV` pattern integrated via `SciKeras` wrappers across network depth, neuron counts, learning rates, optimization algorithms, activation structures, and batch configurations.
5. **Comparative Evaluation:** Re-fit parameters on the optimal structural combination and benchmarked performance metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Training Footprint) directly against the baseline layout.

## How to Run

```bash
# 1. Activate your Python environment and verify path context
source venv/bin/activate

# 2. Synchronize precise version requirements
pip install -r requirements.txt

# 3. Execute the full integrated classification pipeline
python src/main.py