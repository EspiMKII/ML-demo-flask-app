import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import os
import io
import base64
import joblib

def get_model(model_id):
    """
    Load a regression model from disk based on model ID.
    
    Args:
        model_id (str): ID of the model to load
            'linear' - Linear Regression
            'ridge' - Ridge Regression
            'lasso' - Lasso Regression
            'svr' - Support Vector Regression
            'linear_ge' - Linear Regression for GE Stock
    
    Returns:
        Pipeline: Loaded scikit-learn pipeline
    """
    model_map = {
        "linear": "linear-regression.pkl",
        "ridge": "ridge-regression.pkl",
        "lasso": "lasso-regression.pkl",
        "svr": "support-vector-regression.pkl",
        "linear_ge": "linear-regression-GE.pkl"
    }
    
    # Get path to model file
    model_file = model_map.get(model_id)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_file)
    
    # Load the model
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise ValueError(f"Error loading model {model_id}: {str(e)}")

def prepare_data(time_step=1, stock='GOOG', feature_names=None):
    """
    Prepare stock data for regression model predictions.
    
    Args:
        time_step (int): Time step interval for data sampling
        stock (str): Stock symbol to use (GOOG or GE)
        feature_names (list): Specific feature names to use (for model compatibility)
    
    Returns:
        tuple: X_test, y_test DataFrames for prediction
    """
    # Load the data from pre-downloaded files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if stock.upper() == 'GE':
        data_path = os.path.join(script_dir, '..', '..', 'data', 'regression', 'regression-ge-data.csv')
    else:  # default to GOOG
        data_path = os.path.join(script_dir, '..', '..', 'data', 'regression', 'regression-goog-data.csv')
    
    # Skip the first 3 rows (they're header info) and use custom names
    df = pd.read_csv(data_path, skiprows=3, header=None, 
                    names=['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume'])
    
    # Convert Date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # If feature names are provided, use them in the exact order
    if feature_names is not None:
        # Create a dataframe with exactly the right features in the right order
        window = 2  # Assuming 2 lag features
        n = len(df['Close']) - window - 1
        
        # Create base features
        features = {}
        for i in range(window):
            # Create all possible lag features that the model might expect
            features[f'lag_{i+1}'] = df['Close'].shift(i).iloc[window-1:window-1+n].values
        
        # Create DataFrame with perfect feature name matching
        X = pd.DataFrame(index=df.index[window-1:window-1+n])
        
        # Add only the features needed by the model, in the exact order
        for name in feature_names:
            if name in features:
                X[name] = features[name]
            else:
                # If a required feature is missing, raise an error
                raise ValueError(f"Cannot create required feature: {name}")
    else:
        # Default to the original approach
        window = 2 
        n = len(df['Close']) - window - 1
        X = pd.DataFrame(
            [df['Close'].iloc[i:i + window].values.flatten() for i in range(n)],
            index=df.index[window - 1:window - 1 + n],
            columns=[f'lag_{j+1}' for j in range(window)]
        )
    
    # Create target variable (next day's closing price)
    X['target'] = df['Close'].shift(-1).iloc[window - 1:window - 1 + n]
    X = X.dropna()
    y = X.pop('target')
    
    # Split into train/test (using last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # Apply time step to reduce data points
    if time_step > 1:
        step_indices = list(range(0, len(X_test), time_step))
        X_test = X_test.iloc[step_indices]
        y_test = y_test.iloc[step_indices]
    
    return X_test, y_test

def run_model(model, time_step=1):
    """
    Run regression model prediction on test data.
    
    Args:
        model (Pipeline): Loaded scikit-learn pipeline
        time_step (int): Time step interval for data sampling
    
    Returns:
        dict: Results dictionary with predictions and metrics
    """
    # Determine which stock data to use based on the model type
    stock = 'GE' if 'ge' in str(model).lower() else 'GOOG'
    
    # Extract feature names from the model if possible
    feature_names = None
    try:
        # For direct models
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
            print(f"Using feature names from model: {feature_names}")
        # For pipelines
        elif hasattr(model, 'steps'):
            for _, step in model.steps:
                if hasattr(step, 'feature_names_in_'):
                    feature_names = step.feature_names_in_.tolist()
                    print(f"Using feature names from pipeline step: {feature_names}")
                    break
    except Exception as e:
        print(f"Could not extract feature names: {e}")
    
    # Prepare data with the exact feature names needed
    X_test, y_test = prepare_data(time_step, stock, feature_names)
    
    # Make predictions
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    # Return results
    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    }

def plot(model_name, y_test, y_pred, X_test, time_step=1, return_file=True):
    """
    Generate plot comparing actual vs predicted values.
    
    Args:
        model_name (str): Name of the model
        y_test (pd.Series): Actual values
        y_pred (pd.Series): Predicted values
        X_test (pd.DataFrame): Feature data
        time_step (int): Time step interval
        return_file (bool): Whether to return a base64 image
    
    Returns:
        str: Base64 encoded image if return_file=True
    """
    # Create DataFrame for plotting
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    }, index=y_test.index)
    
    # Plot settings
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Actual'], label='Actual', color='blue')
    plt.plot(results.index, results['Predicted'], label='Predicted', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{model_name} - Actual vs Predicted Prices (Every {time_step} Days)')
    plt.legend()
    plt.tight_layout()
    
    # Return base64 encoded image
    if return_file:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    else:
        plt.show()

# For testing
if __name__ == "__main__":
    # Test all regression models
    model_ids = ['linear', 'ridge', 'lasso', 'svr', 'linear_ge']
    model_names = {
        'linear': 'Linear Regression (GOOG)',
        'ridge': 'Ridge Regression',
        'lasso': 'Lasso Regression',
        'svr': 'Support Vector Regression',
        'linear_ge': 'Linear Regression (GE)'
    }
    
    for model_id in model_ids:
        print(f"\n{'='*50}")
        print(f"Testing model: {model_names[model_id]}")
        print(f"{'='*50}")
        
        try:
            # Load the model
            model = get_model(model_id)
            print(f"Loaded model: {type(model)}")
            
            # Test running predictions
            result = run_model(model, time_step=5)
            print(f"RMSE: {result['rmse']:.4f}")
            print(f"R²: {result['r2']:.4f}")
            print(f"MAPE: {result['mape']:.4f}%")
            
            # Test plotting (comment out to avoid displaying many plots)
            print("Generating plot...")
            plot(
                model_names[model_id], 
                result["y_test"], 
                result["y_pred"], 
                result["X_test"], 
                time_step=5,
                return_file=False  # Set to True to avoid displaying plots
            )
            
        except Exception as e:
            print(f"Error testing {model_id}: {str(e)}")