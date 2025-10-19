# IRIS MLOps CI/CD Pipeline - Week 4 Assignment

**Name:** Vishwas Mehta  
**Roll No:** 22F3001150  
**GitHub Repository:** https://github.com/vishwas-mehta/iris-mlops-homework

## Project Overview

This project implements a complete MLOps CI/CD pipeline for IRIS flower classification using GitHub Actions, DVC, and CML.

## File Structure and Utilities

### **Source Code**

- **`train.py`**  
  Trains a RandomForestClassifier on the IRIS dataset and saves the model to `models/iris_model.joblib`

- **`evaluate.py`**  
  Loads the trained model and evaluates it on test data, printing accuracy and classification report

### **Unit Tests**

- **`test_eval.py`**  
  Contains 2 evaluation tests:
  - `test_model_accuracy_threshold`: Verifies model accuracy ≥ 90%
  - `test_model_prediction_integrity`: Validates prediction format and values

- **`test_data_validation.py`**  
  Contains 5 data validation tests:
  - `test_iris_data_shape`: Checks dataset has 150 samples and 5 columns
  - `test_iris_data_columns`: Validates column names
  - `test_iris_target_values`: Ensures target values are 0, 1, or 2
  - `test_iris_no_missing_values`: Checks for no missing data
  - `test_iris_feature_ranges`: Verifies all features are non-negative

### **CI/CD Configuration**

- **`ci-dev.yml`** (`.github/workflows/`)  
  GitHub Actions workflow for dev branch:
  - Triggers on push to dev or PR to main
  - Runs pytest tests
  - Pulls model from DVC
  - Evaluates model
  - Posts CML report as comment

- **`ci-main.yml`** (`.github/workflows/`)  
  GitHub Actions workflow for main branch:
  - Triggers on push to main
  - Runs sanity tests
  - Generates comprehensive test report
  - Posts CML report as commit comment

### **Dependencies**

- **`requirements.txt`**  
  Lists all Python dependencies:
  - scikit-learn (ML model)
  - pandas, numpy (data handling)
  - pytest (testing)
  - joblib (model serialization)
  - dvc, dvc-gs (model versioning)

### **DVC Configuration**

- **`.dvc/config`**  
  DVC remote storage configuration pointing to Google Cloud Storage bucket

- **`models/iris_model.joblib.dvc`**  
  DVC tracking file for the trained model (actual model stored in GCS)

## How It Works

1. **Development:** Code changes pushed to `dev` branch
2. **Testing:** CI pipeline runs automatically, testing code and model
3. **Pull Request:** Create PR from `dev` to `main`
4. **Review:** CI runs on PR, CML posts evaluation report
5. **Merge:** After approval, merge to `main`
6. **Sanity Test:** Main branch CI runs final verification

## Key Features

✅ Automated testing with pytest  
✅ Model versioning with DVC  
✅ CI/CD pipelines for both branches  
✅ Automatic evaluation reports with CML  
✅ Secure credential management with GitHub Secrets  
✅ Google Cloud Storage integration

## Running Tests Locally

