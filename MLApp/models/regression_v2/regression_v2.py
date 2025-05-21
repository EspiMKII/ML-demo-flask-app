import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from utils import *
import os
import io
import base64
import pickle as pkl

def get_model(model_id):
    """
    Load a saved regression model based on model ID.
    
    Args:
        model_id (str): ID of the model to load
            'ge_lr' - Linear Regression, GE stock
            'ge_ridge' - Ridge Regression, GE stock
            'ge_lasso' - Lasso Regression, GE stock
            'ge_svr' - Support Vector Regression, GE stock
            'gg_svr' - Suppport Vector Regression, GOOG stock
    
    Returns:
        Any: some scikit-learn model
    """
    # Get the path to the model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f'{model_id}_1.pkl')

    # print(model_path)

    # Load the model using pickle since models were saved with pickle.dump()
    with open(model_path, 'rb') as f:
        return pkl.load(f)

def prepare_data(time_step=1, stock='GE'):
    # heavily carried by utils.py
    """
    Prepare stock data for regression model predictions.
    
    Args:
        time_step (int): Time step interval for data sampling
        stock (str): Stock symbol to use (GOOG or GE)
        feature_names (list): Specific feature names to use (for model compatibility)
    
    Returns:
        tuple: X_test (DataFrame), y_test (Series) for prediction
    """

    # Get the path of the data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data', 'regression-v2')

    if stock == 'GE':
        data_path = os.path.join(data_dir, 'ge.us.txt')
    else: # GOOG
        data_path = os.path.join(data_dir, 'googl.us.txt')
    
    # load the data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).drop(columns=['OpenInt'])
    
    # creating the data sets from the data, and splitting thme
    df_X_1, df_y_1 = create_dataset(df, window=1, predicted_interval=time_step)
    df_X_1_train, df_X_1_test, df_y_1_train, df_y_1_test = split_dataset(df_X_1, df_y_1)

    return df_X_1_test, df_y_1_test

def get_stock_from_model(model):
    """Helper function to find which stock a model was trained on"""
    # Try to get the first feature name - models trained in notebook store this
    if hasattr(model, 'feature_names_in_'):
        first_feature = model.feature_names_in_[0]
        return 'GE' if 'ge' in first_feature.lower() else 'GOOG'
    
    # Fallback: check model's string representation
    model_str = str(model).lower()
    return 'GE' if 'ge' in model_str else 'GOOG'

def run_model(model, time_step=1, ):
    """
    Run regression model prediction on test data.
    
    Args:
        model (Any): Loaded scikit-learn model
        time_step (int): Time step interval for data sampling
    
    Returns:
        dict: Results dictionary with predictions and metrics
    """
    # Determine stock
    stock = get_stock_from_model(model)

    # Prepare data
    X_test, y_test = prepare_data(time_step, stock)
    print(X_test.info())

    # Make predictions & Calculate metrics
    y_pred = pd.Series(model.predict(X_test), index = X_test.index)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,        
    }

def plot(model_name, y_test, y_pred, time_step=1, return_file=True):
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
    day_str = "Day" if time_step == 1 else "Days"
    plt.title(f'{model_name} - Actual vs Predicted Prices After {time_step} ' + day_str) 
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
    # model_ids = ['ge_lr', 'ge_ridge', 'ge_lasso', 'ge_svr', 'gg_svr']
    model_ids = ['ge_lr', 'gg_svr']
    model_names = {
        'ge_lr': "Regression - Linear Regression (GE Stock)",
        # 'ge_ridge': "Regression - Ridge Regression (GE Stock)",
        # 'ge_lasso': "Regression - Lasso Regression (GE Stock)",
        # 'ge_svr': "Regression - Support Vector Regression (GE Stock)",
        'gg_svr':  "Regression - Linear Regression (GOOG Stock)"
    }
    
    for model_id in model_ids:
        print(f"\nTesting: {model_names[model_id]}")

        time_step = 10
        model = get_model(model_id)
        
        result = run_model(model, time_step)
        print(f"Metrics - RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}, MAPE: {result['mape']:.4f}%")
        
        print("Testing plot...", end=" ")
        plot(
            model_names[model_id], 
            result["y_test"], 
            result["y_pred"], 
            time_step,
            return_file=False
        )
        print()