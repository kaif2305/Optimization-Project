# Churn Prediction Optimization

This project focuses on predicting customer churn for a telecommunications company using a RandomForestClassifier, with an emphasis on optimizing model performance through hyperparameter tuning.

## Project Structure

The project is structured around a Python script (`churn_prediction.py` - *assuming this is the name of the script based on the provided code*) that performs the following key steps:

1.  **Data Loading and Initial Exploration**: Loads the `Telco-Customer-Churn.csv` dataset and displays basic information, including data types, missing values, class distribution of the target variable ('Churn'), and a sample of the data.
2.  **Data Preprocessing**:
    * **Handling Missing Values**: Converts 'TotalCharges' to a numeric type and imputes missing values with the median.
    * **Encoding Categorical Variables**: Uses `LabelEncoder` to transform all categorical features (except 'customerID' if present and 'Churn' itself) into numerical representations. The 'Churn' target variable is also encoded.
    * **Feature Scaling**: Applies `StandardScaler` to numerical features ('tenure', 'MonthlyCharges', 'TotalCharges') to standardize their ranges.
3.  **Model Training and Initial Evaluation**:
    * Splits the data into training and testing sets (80% training, 20% testing).
    * Trains an initial `RandomForestClassifier` with default parameters.
    * Evaluates the initial model's performance using accuracy and a classification report on the test set.
4.  **Hyperparameter Optimization using RandomizedSearchCV**:
    * Defines a parameter distribution grid for `RandomForestClassifier` including `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
    * Utilizes `RandomizedSearchCV` to efficiently search for the best hyperparameter combination within the defined grid, performing 20 iterations with 5-fold cross-validation.
    * Identifies and prints the best parameters found by the search.
5.  **Tuned Model Training and Evaluation**:
    * Trains a new `RandomForestClassifier` using the best parameters obtained from `RandomizedSearchCV`.
    * Evaluates the tuned model's performance on the test set, comparing its accuracy and classification report to the initial model.
6.  **Cross-Validation of the Tuned Model**:
    * Performs 5-fold cross-validation on the entire dataset using the best model to provide a more robust estimate of its generalization performance.
    * Prints individual cross-validation scores and their mean.

## Setup and Usage

### Prerequisites

* Python 3.x
* pandas
* scikit-learn
* numpy

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn numpy