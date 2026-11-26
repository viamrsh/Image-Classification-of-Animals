# Lung-Cancer-Survival-Prediction-System-
# 1. Introduction 
Lung cancer remains one of the leading causes of cancer-related deaths worldwide. Early prediction of 
patient survival can significantly support clinical decision-making, personalized treatment planning, 
and risk assessment. This project presents a machine learning–based Lung Cancer Survival Prediction 
System developed using Python. The system processes clinical patient data, performs exploratory 
analysis, applies preprocessing techniques, and evaluates multiple classification algorithms to predict 
survival outcomes. The objective is to design a robust, automated pipeline that can accurately classify 
patients into survival categories and provide meaningful risk insights for healthcare professionals. 
# 2. Methodology 
The methodology follows a structured end-to-end machine learning pipeline: 
# 2.1 Data Loading and Exploration 
The system reads a CSV dataset containing patient clinical information. It automatically detects the 
target variable (survival status) and handles various label formats such as numeric (0/1) or categorical 
(Yes/No). Initial data exploration includes: 
• Dataset shape and structure analysis 
• Missing value detection 
• Target distribution visualization 
• Correlation heatmap for numerical features 
# 2.2 Data Preprocessing 
The preprocessing stage ensures data quality and consistency: 
• Removal of ID or irrelevant columns 
• Handling missing values (median for numerical, most frequent for categorical) 
• Feature scaling using StandardScaler 
• Categorical encoding using OneHotEncoder 
• Automatic identification of numeric and categorical features 
A ColumnTransformer pipeline integrates all preprocessing steps to maintain reproducibility. 
# 2.3 Feature Engineering 
If date-related fields are present, the system extracts meaningful features such as: 
• Diagnosis year and month 
• Treatment duration (in days and months) 
This improves model learning by incorporating temporal insights. 
# 2.4 Model Training and Evaluation 
The dataset is split into training and testing sets using stratified sampling. The following classification 
models are trained and evaluated: 
• Logistic Regression 
• Random Forest 
• Gradient Boosting 
• Support Vector Machine (SVM) 
• Gaussian Naive Bayes 
Five-fold cross-validation is used to compute: 
• Accuracy 
• Precision 
• Recall 
• F1-score 
• AUC-ROC 
The best-performing model is selected based on F1-score and retrained on the full training data. 
# 3. Output Results 
The system produces the following outputs: 
• Target distribution visualization 
• Correlation heatmap 
• ROC curve for the best model 
• Confusion matrix values 
• Classification report including precision, recall, and F1-score 
• Model performance comparison table 
The selected best model demonstrates strong predictive capability based on F1-score and AUC values. 
Feature importance (if supported by the model) identifies key clinical factors influencing survival 
prediction. 
Additionally, the system supports: 
• Single patient prediction 
• Batch prediction for multiple patients 
• Probability-based risk assessment 
• Risk level categorization (Low, Medium, High) 
• Basic recommendation suggestions based on predicted risk 
The trained model and preprocessing pipeline are saved using pickle for future deployment. 
# 4. Conclusion 
The Lung Cancer Survival Prediction System successfully implements a complete machine learning 
pipeline for clinical survival prediction. By integrating data preprocessing, feature engineering, cross
validation, and multi-model comparison, the system ensures robust and reliable performance. The 
automated structure enhances reproducibility and ease of deployment. This project demonstrates how 
machine learning can support healthcare decision-making through risk stratification and predictive 
analytics. 
Future improvements may include hyperparameter tuning, handling class imbalance techniques (such 
as SMOTE), model interpretability methods like SHAP, and external clinical validation to improve 
reliability and real-world applicability.
