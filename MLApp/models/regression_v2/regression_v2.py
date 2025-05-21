import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from .utils import *
import os
import io
import base64
import pickle as pkl

def get_model(model_id, ticker='GOOG', time_step=10):
    """
    Load a saved regression model based on model ID, ticker, and time step.
    
    Args:
        model_id (str): ID of the model to load
            'lr': Linear Regression
            'lasso': Lasso Regression
            'ridge': Ridge Regression
            'svr': Support Vector Regression
        ticker (str): 'GOOG' or 'GE', default 'GOOG'
        time-step (int): time step, default 10
    Returns:
        Any: some scikit-learn model
    """
    # Get the path to the model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ticker_name = 'gg' if ticker == 'GOOG' else 'ge'
    model_name = ticker_name + '_' + model_id + '_' + str(time_step) + '.pkl'
    model_path = os.path.join(script_dir, model_name)

    # Load the model using pickle since models were saved with pickle.dump()
    with open(model_path, 'rb') as f:
        return pkl.load(f)

def prepare_data(time_step=10, ticker='GOOG'):
    # heavily carried by utils.py
    """
    Prepare ticker data for regression model predictions.
    
    Args:
        time_step (int): Time step, default 10
        ticker (str): 'GOOG' or 'GE', default 'GOOG'
    
    Returns:
        tuple: X_test (DataFrame), y_test (Series) for prediction
    """

    # Get the path of the data file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data', 'regression-v2')

    if ticker == 'GOOG':
        data_path = os.path.join(data_dir, 'googl.us.txt')
    else: # GE
        data_path = os.path.join(data_dir, 'ge.us.txt')
    
    # load the data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).drop(columns=['OpenInt'])
    
    # creating the data sets from the data, and splitting thme
    df_X_1, df_y_1 = create_dataset(df, window=1, predicted_interval=time_step)
    df_X_1_train, df_X_1_test, df_y_1_train, df_y_1_test = split_dataset(df_X_1, df_y_1)

    return df_X_1_test, df_y_1_test

def get_ticker_from_model(model):
    """Helper function to find which ticker a model was trained on"""
    # Just check the model ID in the file name
    model_str = str(model).lower()
    return 'GOOG' if 'gg_' in model_str else 'GE'

def run_model(model, time_step=1):
    """
    Run regression model prediction on test data.
    
    Args:
        model (Any): Loaded scikit-learn model
        time_step (int): Time step
    
    Returns:
        dict: Results dictionary with predictions and metrics
    """
    # Determine ticker
    ticker = get_ticker_from_model(model)

    # Prepare data
    X_test, y_test = prepare_data(time_step, ticker)

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

def plot(model_name, y_test, y_pred, ticker, time_step=10, return_file=True):
    """
    Generate plot comparing actual vs predicted values.
    
    Args:
        model_name (str): Name of the model
        y_test (pd.Series): Actual values
        y_pred (pd.Series): Predicted values
        ticker (str): ticker
        time_step (int): Time step
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
    plt.xlabel('Time')
    plt.ylabel('Ticker Price')
    day_str = "Day" if time_step == 1 else "Days"
    ticker_str = ticker + ' Ticker '
    plt.title(f'{model_name} - Actual vs Predicted ' + ticker_str + f'Prices After {time_step} ' + day_str) 
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
    model_ids = ['lr', 'ridge', 'lasso', 'svr']
    time_step, ticker = 10, 'GOOG'
    model_names = {model_id:ticker + '_' + model_id + '_' + str(time_step) for model_id in model_ids}
    
    for model_id in model_ids:
        print(f"\nTesting: {model_names[model_id]}")
        
        model = get_model(model_id, ticker, time_step)
        
        result = run_model(model, time_step)
        print(f"Metrics - RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}, MAPE: {result['mape']:.4f}%")
        
        print("Testing plot...", end=" ")
        plot(
            model_names[model_id], 
            result["y_test"], 
            result["y_pred"], 
            ticker,
            time_step,
            return_file=False
        )
        print()