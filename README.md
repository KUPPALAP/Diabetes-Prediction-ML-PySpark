# Diabetes-Prediction-ML-PySpark


Diabetes Prediction and Classification using Machine Learning Models with PySpark

Project Overview

This project applies machine learning techniques to predict diabetes and related conditions using a structured healthcare dataset. Our team explored various classification models, including Logistic Regression, Random Forest, and K-Nearest Neighbors (KNN), comparing their effectiveness on diabetes prediction tasks by analyzing metrics like accuracy, precision, recall, F1 score, and specificity.

Team Members and Contributions

Karunakar Uppalapati: Implemented Logistic Regression model in PySpark

Varshitha Devarapalli: Implemented Random Forest model in PySpark

Saketha Kusu: Implemented K-Nearest Neighbors (KNN) model in Scikit-Learn

Objective

The goal of this project is to:

Predict diabetes outcomes using machine learning models.
Identify the best-performing model based on metrics like accuracy, precision, recall, F1 score, and specificity.
Gain insights into feature importance and model behavior in a healthcare dataset.

Dataset

The project dataset was a structured healthcare dataset related to diabetes. Key steps in the preprocessing stage included handling missing values, encoding categorical variables, and vectorizing numeric features.

Project Steps

1. Data Preprocessing

2. Handling Missing Values: Missing numeric values were replaced with median values to retain data consistency.

3. Column Normalization: Removed spaces from column names for easier handling.

4. Label Encoding: Categorical labels were converted to numeric format.

5. Feature Vector Assembly: All numeric features were assembled into a single vector to streamline PySpark ML pipeline processes.

6. 2. Model Implementation

  Three machine learning models were used, each tailored to different aspects of the dataset:

Logistic Regression (Karunakar Uppalapati)

Method: Applied using PySpark's LogisticRegression with 10 iterations for convergence.

Performance:
Accuracy: 64.17%
Weighted Precision: 64.17%
F1 Score: 64.08%
Specificity was high for some classes, indicating effectiveness in identifying true negatives.

Random Forest (Varshitha Devarapalli)

Method: PySpark's RandomForestClassifier with 100 trees and a maximum depth of 10.

Performance:
Accuracy: 80.74%
Weighted Precision: 83.28%
F1 Score: 80.50%
Specificity across classes was high, making it ideal for minimizing false positives.

K-Nearest Neighbors (KNN) (Saketha Kusu)

Method: Scikit-Learn’s KNeighborsClassifier after converting the dataset to Pandas for compatibility.

Performance:
Accuracy: 69.58%
Weighted Precision: 70.49%
F1 Score: 69.17%
Specificity was high for most classes, though performance varied with dataset size and K-value tuning.

3. Evaluation Metrics

4. Accuracy: Random Forest showed the highest accuracy, while Logistic Regression and KNN also performed well.
   
Precision, Recall, and F1 Score: Random Forest had the highest metrics, suggesting robustness in classification.

Specificity: High specificity was seen in Random Forest, reflecting strong performance in identifying true negatives.

Results Summary

Random Forest was the best-performing model with high accuracy, precision, recall, and specificity, making it suitable for the dataset’s complexity.

Logistic Regression showed reasonable performance, especially for simpler, linearly separable data.

K-Nearest Neighbors (KNN) provided decent results but required computational adjustments due to the dataset size.

Conclusion

Random Forest is recommended for complex patterns and large datasets.
KNN can be effective with smaller, clustered data but may need optimization for larger sets.
Logistic Regression is suitable for interpretable, linear data but may underperform on complex patterns.
   3. 
