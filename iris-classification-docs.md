# Iris Classification Model Documentation

## Table of Contents
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Data Loading and Preparation](#data-loading-and-preparation)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Model Implementation](#model-implementation)
6. [Results and Evaluation](#results-and-evaluation)
7. [Usage Guide](#usage-guide)

## Overview
This Python script implements a machine learning solution for classifying iris flowers using the classic Iris dataset. The implementation compares two different classification algorithms: Naive Bayes and Support Vector Machine (SVM). The script includes data visualization, preprocessing, model training, and evaluation components.

## Dependencies
The following Python libraries are required to run this code:
```python
numpy==1.21.0
pandas==1.3.0
matplotlib==3.4.2
scikit-learn==0.24.2
```

## Data Loading and Preparation
### Dataset
The script uses the built-in Iris dataset from scikit-learn, which contains:
- 150 samples
- 4 features (sepal length, sepal width, petal length, petal width)
- 3 target classes (different iris species)

### Data Loading
```python
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                   columns=iris['feature_names'] + ['target'])
```

### Feature and Target Separation
```python
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable
```

## Exploratory Data Analysis
The script includes two visualization components:

1. **Scatter Plot**
   - Visualizes the relationship between sepal length and petal length
   - Color-coded by class
   - Includes a colorbar for class identification

2. **Feature Distributions**
   - Histogram plots for all features
   - Helps understand the distribution of each measurement
   - Uses 15 bins for detailed distribution visualization

## Model Implementation

### Data Preprocessing
1. **Label Encoding**
   - Converts class labels to numerical format
   ```python
   le = LabelEncoder()
   y = le.fit_transform(y)
   ```

2. **Train-Test Split**
   - 80% training data, 20% test data
   - Random state set to 42 for reproducibility
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

### Classification Models

1. **Naive Bayes Classifier**
   ```python
   nb_model = GaussianNB()
   nb_model.fit(X_train, y_train)
   ```

2. **Support Vector Machine (SVM)**
   ```python
   svm_model = SVC(kernel='linear', random_state=42)
   svm_model.fit(X_train, y_train)
   ```

## Results and Evaluation
Both models are evaluated using:
- Accuracy score
- Classification report (precision, recall, f1-score)

The evaluation metrics are printed for each model separately, allowing for easy comparison of performance.

## Usage Guide

1. **Setup**
   ```python
   # Clone the repository or copy the script
   # Install required dependencies
   pip install numpy pandas matplotlib scikit-learn
   ```

2. **Running the Script**
   ```python
   # Execute the script
   python Mithu.py
   ```

3. **Expected Output**
   - Visualization plots will be displayed
   - Model performance metrics will be printed to console
   - Classification reports will show detailed performance metrics

### Notes
- The random state is set to 42 for reproducibility
- The SVM model uses a linear kernel
- The visualization settings can be modified by adjusting the figure size and plot parameters

## Contributing
To contribute to this project:
1. Fork the repository
2. Create a new branch for your feature
3. Submit a pull request with a detailed description of your changes

## License
This project is available under the MIT License.
