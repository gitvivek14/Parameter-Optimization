# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import learning_curve
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

# Cell 2: Load Dataset
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 

# Convert categorical variables to numeric
le = LabelEncoder()
for column in X.columns:
    X[column] = le.fit_transform(X[column])
y = le.fit_transform(y.values.ravel())

# Display first few rows
print("First few rows of the dataset:")
display(X.head())

# Plot class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=y)
plt.title('Distribution of Car Evaluation Classes')
plt.show()

# Cell 3: Data Preparation
# Scale the features
ss = StandardScaler()
X_scaled = ss.fit_transform(X)

# Create samples
samples = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=i)
    samples.append((X_train, X_test, y_train, y_test))

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
result = pd.DataFrame(columns=['Sample', 'Best Accuracy', 'Best Kernel', 'Best Nu', 'Best Epsilon'])

# Rest of the cells remain the same...