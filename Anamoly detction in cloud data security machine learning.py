import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Function to load data from a single CSV file
def load_data_from_file(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

# Step 1: Data Loading and Preprocessing
data_file = r'C:\Users\VKK\Downloads\labelled_2021may-ip-10-100-1-105-dns.csv'  
data = load_data_from_file(data_file)

# Preprocess the data: Handle missing values, if any
data = data.dropna()

# Select only numeric columns for anomaly detection
numeric_data = data.select_dtypes(include=[np.number])

# Check and handle zero variance columns
zero_variance_cols = numeric_data.columns[numeric_data.std() == 0]
if zero_variance_cols.any():
    print(f"Columns with zero variance: {zero_variance_cols}")
    numeric_data.drop(zero_variance_cols, axis=1, inplace=True)
    print("Dropped zero variance columns.")

# Feature engineering: Create additional features if necessary
numeric_data['interaction'] = numeric_data.iloc[:, 0] * numeric_data.iloc[:, 1]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(numeric_data)

# Step 2: Train-Test Split
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Step 3: Model Training using Isolation Forest
model = IsolationForest(random_state=42)
model.fit(X_train)

# Step 4: Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert predictions to binary labels (0: normal, 1: anomaly)
y_pred_train = np.where(y_pred_train == 1, 0, 1)
y_pred_test = np.where(y_pred_test == 1, 0, 1)

# Step 5: Evaluation Metrics and Further Exploration

# Since we don't have true labels for anomalies, we'll consider all test data as normal (y_true_test = 0)
y_true_test = np.zeros(X_test.shape[0])

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_test, y_pred_test)
print(f"Confusion Matrix:\n{conf_matrix}")

# Accuracy
accuracy = accuracy_score(y_true_test, y_pred_test)
print(f"Accuracy: {accuracy}")

# Precision
precision = precision_score(y_true_test, y_pred_test, zero_division=0)
print(f"Precision: {precision}")

# Recall
recall = recall_score(y_true_test, y_pred_test, zero_division=0)
print(f"Recall: {recall}")

# F1 Score
f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
print(f"F1 Score: {f1}")

# Classification Report
class_report = classification_report(y_true_test, y_pred_test, zero_division=0)
print("Classification Report:")
print(class_report)

# Model Loss (Negative Anomaly Scores)
train_loss = -model.score_samples(X_train).mean()
test_loss = -model.score_samples(X_test).mean()
print(f"Train Loss (Negative Anomaly Score): {train_loss}")
print(f"Test Loss (Negative Anomaly Score): {test_loss}")

# Additional Visualizations and Qualitative Analysis

# Visualize the distribution of anomaly scores
train_anomaly_scores = -model.score_samples(X_train)
test_anomaly_scores = -model.score_samples(X_test)

plt.figure(figsize=(10, 6))
plt.hist(train_anomaly_scores, bins=50, alpha=0.5, color='blue', label='Train Anomaly Scores', density=True)
plt.hist(test_anomaly_scores, bins=50, alpha=0.5, color='red', label='Test Anomaly Scores', density=True)
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.title('Distribution of Anomaly Scores')
plt.legend(loc='upper right')
plt.show()

# Pairplot for visualizing relationships (optional)
sns.pairplot(data=pd.DataFrame(X_train, columns=numeric_data.columns), diag_kind='kde')
plt.suptitle('Pairplot of Scaled Features', y=1.02)
plt.show()

# Heatmap of correlations (optional)
corr = np.corrcoef(X_train, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', xticklabels=numeric_data.columns, yticklabels=numeric_data.columns)
plt.title('Correlation Heatmap of Scaled Features')
plt.show()

# Analyze top anomalies (optional)
top_anomalies_indices = np.argsort(test_anomaly_scores)[-10:]
top_anomalies_data = data.iloc[top_anomalies_indices]
print("Top Anomalies:")
print(top_anomalies_data)

# Step 6: Model Deployment and Integration

# Save the model to disk using joblib
model_file = 'isolation_forest_model.pkl'
joblib.dump(model, model_file)
print(f"Model saved as {model_file}")

# Load the model later (if needed)
loaded_model = joblib.load(model_file)
print("Model loaded successfully.")
