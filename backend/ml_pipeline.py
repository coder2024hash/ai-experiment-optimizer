import pandas as pd
import numpy as np
from pathlib import Path
import uuid
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report

def read_table(file_path):
    """
    Read CSV or Excel file
    
    Args:
        file_path: path to data file
    
    Returns:
        pandas DataFrame
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

def guess_target_candidates(df):
    """
    Suggest potential target columns (numeric columns)
    
    Args:
        df: pandas DataFrame
    
    Returns:
        list of column names
    """
    return df.select_dtypes(include=['number']).columns.tolist()

def detect_problem_type(y):
    """
    Automatically detect if problem is classification or regression
    
    Args:
        y: target variable series
    
    Returns:
        'classification' or 'regression'
    """
    unique_values = len(np.unique(y))
    total_values = len(y)
    
    # If unique values < 10% of total or <= 20, likely classification
    if unique_values <= 20 or (unique_values / total_values) < 0.1:
        return "classification"
    else:
        return "regression"

def preprocess_data(df, target_column):
    """
    Auto-preprocessing: handle missing values, encode categoricals, scale features
    
    Args:
        df: pandas DataFrame
        target_column: name of target column
    
    Returns:
        X_scaled: preprocessed features
        y: target variable
        problem_type: 'classification' or 'regression'
        scaler: fitted StandardScaler
        label_encoder: fitted LabelEncoder for target (if classification)
    """
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values in target
    if y.isnull().any():
        y = y.fillna(y.mode()[0] if detect_problem_type(y) == "classification" else y.mean())
    
    # Detect problem type
    problem_type = detect_problem_type(y)
    
    # Encode target if classification
    label_encoder = None
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))
    
    # Handle categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values in features
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, problem_type, scaler, label_encoder

def train_model(X, y, problem_type, n_estimators=100, max_depth=10, random_state=42):
    """
    Train RandomForest model based on problem type
    
    Args:
        X: features DataFrame
        y: target variable
        problem_type: 'classification' or 'regression'
        n_estimators: number of trees
        max_depth: maximum tree depth
        random_state: random seed
    
    Returns:
        model: trained model
        metrics: evaluation metrics
        feature_importance: top feature importances
    """
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Select model
    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    if problem_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "test_samples": len(y_test),
            "train_samples": len(y_train)
        }
    else:
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2_score": float(r2_score(y_test, y_pred)),
            "test_samples": len(y_test),
            "train_samples": len(y_train)
        }
    
    # Feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    feature_importance = {
        k: float(v) for k, v in sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
    }
    
    return model, metrics, feature_importance

def train_from_file(file_path, target_column, problem_type="auto", n_estimators=100, max_depth=10):
    """
    Complete training pipeline
    
    Args:
        file_path: path to data file
        target_column: target column name
        problem_type: 'auto', 'classification', or 'regression'
        n_estimators: number of trees
        max_depth: maximum tree depth
    
    Returns:
        meta: model metadata dictionary
        model_path: path to saved model file
    """
    
    # Read data
    df = read_table(file_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Preprocess
    X, y, detected_problem_type, scaler, label_encoder = preprocess_data(df, target_column)
    
    # Override if user specified
    if problem_type != "auto":
        detected_problem_type = problem_type
    
    # Train
    model, metrics, feature_importance = train_model(
        X, y, detected_problem_type, n_estimators, max_depth
    )
    
    # Save model
    model_id = str(uuid.uuid4())
    model_dir = Path(file_path).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = str(model_dir / f"{model_id}.pkl")
    
    # Save model and preprocessing objects
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_columns': list(X.columns)
        }, f)
    
    # Create metadata
    meta = {
        "id": model_id,
        "file_name": Path(file_path).name,
        "model_path": model_path,
        "target_column": target_column,
        "problem_type": detected_problem_type,
        "metrics": metrics,
        "feature_columns": list(X.columns),
        "feature_importance": feature_importance,
        "n_estimators": n_estimators,
        "max_depth": max_depth
    }
    
    return meta, model_path
if __name__ == "__main__":
    file_path = "crop_yield_experiment.csv"  # Adjust path if needed
    target_column = "Crop_Yield_tons"
    
    meta, model_path = train_from_file(
        file_path=file_path,
        target_column=target_column,
        problem_type="auto",
        n_estimators=100,
        max_depth=10
    )

    print("\n=== Training Summary ===")
    for key, val in meta.items():
        print(f"{key}: {val}")
