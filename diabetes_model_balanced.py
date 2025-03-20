import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, accuracy_score, precision_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
import mlflow
import mlflow.sklearn
import joblib
import os
import tempfile

# Set MLflow tracking URI to local directory
mlflow.set_tracking_uri("file:./mlruns")

# Load data
data = pd.read_csv(r"diabetes_prediction_dataset.csv")

# Data exploration and preprocessing
sns.countplot(x='diabetes', data=data, hue="diabetes", palette="seismic_r")
plt.title('Distribution of Diabetes (0 = No, 1 = Yes)')
plt.show()

numeric_data = data.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

print(f"Duplicate rows: {data.duplicated().sum()}")
data = data.drop_duplicates()

# Separate features and target
X = data.drop(columns=['diabetes'])
y = data['diabetes']

# Define columns
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = ['gender', 'smoking_history']
binary_cols = ['hypertension', 'heart_disease']

# Encode categorical variables
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Scale numerical features
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

# Save feature names
feature_names = X_encoded.columns.tolist()

# Balance classes with SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_balanced, y_balanced = smote_tomek.fit_resample(X_encoded, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Define parameter grids for GridSearchCV
param_grids = {
    'knn': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1: Manhattan, 2: Euclidean
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'ExtraTrees': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    },
    'LogisticRegression': { 
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
}

# Define models
models = {
    'knn': KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(random_state=42),
    'ExtraTrees': ExtraTreesClassifier(random_state=42),
    'xgboost': XGBClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

# Plotting functions
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    temp_path = os.path.join(tempfile.gettempdir(), f'confusion_matrix_{model_name}.png')
    plt.savefig(temp_path)
    plt.close()
    return temp_path

def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    temp_path = os.path.join(tempfile.gettempdir(), f'roc_curve_{model_name}.png')
    plt.savefig(temp_path)
    plt.close()
    return temp_path

# Training and evaluation function
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    experiment_name = "Diabetes_Classification_Balanced_GridSearch"
    mlflow.set_experiment(experiment_name)

    results = {}

    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_GridSearch_Experiment") as run:
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                scoring='f1',
                cv=5,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'AUC': roc_auc_score(y_test, y_proba),
                'Recall': recall_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred),
                'Kappa': cohen_kappa_score(y_test, y_pred),
                'MCC': matthews_corrcoef(y_test, y_pred)
            }
            results[model_name] = metrics
            
            # Generate and log confusion matrix
            cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
            mlflow.log_artifact(cm_path, "confusion_matrix")
            
            # Generate and log ROC curve
            roc_path = plot_roc_curve(y_test, y_proba, model_name)
            mlflow.log_artifact(roc_path, "roc_curve")
            
            # Log classification report
            report = classification_report(y_test, y_pred)
            temp_report_path = os.path.join(tempfile.gettempdir(), f'classification_report_{model_name}.txt')
            with open(temp_report_path, 'w') as f:
                f.write(report)
            mlflow.log_artifact(temp_report_path, "classification_report")

            # Log best parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("model_name", model_name)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log the model
            mlflow.sklearn.log_model(best_model, f"model_{model_name}")
            
            # Save model data
            model_data = {
                'model': best_model,
                'scaler': scaler,
                'feature_names': feature_names
            }
            model_file = f"{model_name}_model.pkl"
            joblib.dump(model_data, model_file)
            mlflow.log_artifact(model_file)

            print(f"\nModel: {model_name}")
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Metrics: {metrics}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results).T

    print("\nAll Model Results:")
    print(results_df)

    return results_df

if __name__ == "__main__":
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)