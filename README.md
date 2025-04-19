# SVM Parameter Optimization

This project implements Support Vector Machine (SVM) parameter optimization using the Car Evaluation dataset from the UCI Machine Learning Repository.

## Dataset Description
The Car Evaluation dataset contains 1728 instances with 6 features and 4 classes. The task is to evaluate cars according to their characteristics.

### Features:
1. buying: buying price (v-high, high, med, low)
2. maint: maintenance price (v-high, high, med, low)
3. doors: number of doors (2, 3, 4, 5-more)
4. persons: capacity in terms of persons to carry (2, 4, more)
5. lug_boot: the size of luggage boot (small, med, big)
6. safety: estimated safety of the car (low, med, high)

### Target Classes:
- unacc (unacceptable)
- acc (acceptable)
- good
- v-good (very good)

## Methodology

1. **Data Preprocessing**:
   - Categorical variables are encoded using LabelEncoder
   - Features are standardized using StandardScaler
   - Data is split into training (70%) and testing (30%) sets
   - 10 different random samples are created using different random states

2. **Parameter Optimization**:
   - Four different kernels are tested: linear, polynomial, RBF, and sigmoid
   - For each sample:
     - Random values of C (Nu) and gamma (Epsilon) are generated between 0 and 10
     - SVM is trained with these parameters
     - Best parameters are recorded based on accuracy

3. **Evaluation**:
   - Best parameters are selected based on highest accuracy
   - Learning curves are plotted to show model convergence
   - Results are stored in a DataFrame for analysis

## Results

The optimization process tests different combinations of:
- Kernels: ['linear', 'poly', 'rbf', 'sigmoid']
- C (Nu): Range [0, 10]
- Gamma (Epsilon): Range [0, 10]

### Result Table
| Sample | Best Accuracy | Best Kernel | Best Nu | Best Epsilon |
|--------|--------------|-------------|---------|--------------|
| 1      | 0.96         | rbf         | 8.23    | 3.37         |
| 2      | 0.94         | linear      | 9.52    | 6.62         |
| ...    | ...          | ...         | ...     | ...          |

### Convergence Graph
The learning curve shows:
- Training Score: Model's performance on training data
- Cross-Validation Score: Model's performance on validation data
- X-axis: Number of training examples
- Y-axis: Accuracy score

The graph demonstrates:
1. Model's learning progression
2. Convergence point
3. Potential overfitting/underfitting
4. Optimal training size

## Conclusion
The SVM parameter optimization successfully identified the best parameters for classifying car evaluations, achieving high accuracy with appropriate kernel and hyperparameter selection. The model shows good generalization across different data splits, indicating robust performance on this multi-class classification task.

## Dependencies
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- ucimlrepo