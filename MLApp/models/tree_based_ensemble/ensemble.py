from joblib import parallel_backend, load
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import parallel_backend
from xgboost import XGBRegressor
import seaborn as sns
import sklearn
import os

def prepare_data(time_step=10):
    """
    Prepare stock data for ensemble model predictions

    Returns: 
        tuple: X_test, y_test DataFrames for prediction
    """

    # Load the data from pre-downloaded files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "../.."))
    csv_path = os.path.join(workspace_root, "data/tree-based-ensemble/ensemble-data.csv")
    df = pd.read_csv(csv_path)
    
    # These were done in the notebook, so I'm doing it here too
    df.drop(['OpenInt', 'Volume', 'Unnamed: 0'], axis=1, inplace=True)
    
    # Convert Date to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Feature engineering 
    # avg
    df['Average_price'] = (df['Close'] + df['Open']) / 2

    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    # EMAs
    df['EMA_5'] = df['Close'].rolling(window=5).mean()
    df['EMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_20'] = df['Close'].rolling(window=20).mean()

    df['STD_5'] = df['Close'].rolling(window=5).std()
    df['STD_10'] = df['Close'].rolling(window=10).std()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    # Lags
    df['Close_t-1'] = df['Close'].shift(1)
    df['Close_t-2'] = df['Close'].shift(2)
    df['Close_t-3'] = df['Close'].shift(3)
    
    # Creating target variable (next n-day's closing price)
    prediction_days = time_step
    df['Target'] = df['Close'].shift(-prediction_days)
    df.dropna(inplace=True)

    X = df.drop(columns=['Target'])
    y = df['Target']

    # Splitting into train/test (last 20% as test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=False)

    return X_test, y_test

def get_model(model_id):
    """
    Load a saved ensemble model based on model ID

    Args: 
        model_id (str): ID of the model to load
            'tree' - Decision Tree
            'forest' - Random Forest
            'grad' - Gradient Boosting
            'vr' - Voting Regressor
            'xg' - XGBoost

    Returns:
        Any: Some scikit-learn model
    """
    model_map = {
        "tree": "decision-tree.pkl",
        "grad": "gradient-boosting.pkl",
        "forest": "random-forest.pkl",
        "vr": "voting-regressor.pkl",
        "xg": "xgboosting.pkl"
    }
    # Get the path to model file
    model_path = model_map[model_id]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_path)

    # Load the model
    model = load(model_path)
    return model

def run_model(model, time_step=1):
    """
    Run ensemble model prediction on test data

    Args:
        model (Any): Loaded scikit-learn model
        time_step (int): Time step interval for data sampling
    
    Return:
        dict: results dictionary with prediction and metrics
    """
    # Prepare the data to predict
    X_test, y_test = prepare_data(time_step)
    
    # # Planting time steps
    # step_indices = range(0, len(X_test), time_step)
    # X_test_stepped = X_test.iloc[step_indices]
    # y_test_stepped = y_test.iloc[step_indices]
    
    # Make predictions & Calculating metrics
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    }

def plot(model_name, y_test, y_pred, X_test, time_step=10, return_file=True):
    '''
    Generate plot comparing actual vs predicted values

    Args:
        model_name (str): Name of the model
        y_test (pd.Series): Actual values
        y_pred (np.ndarray): Predicted values
        X_test (pd.DataFrame): Feature data
        time_step (int): Time step interval
        return_file (bool): Whether to return a base64 image
    
    Returns:
        str: Base64 encoded image if return_file=True
    '''
    # # Calculate the step indices used for predictions
    # step_indices = list(range(0, len(X_test), time_step))
    
    # # Get the indices for the stepped data
    # pred_indices = X_test.index[step_indices]
    
    # Create plot with actual data
    plt.figure(figsize=(12, 6))
    plt.plot(X_test.index, y_test, label='Actual', color='blue')
    
    # Plot predicted data - make sure to use only the stepped indices
    plt.plot(X_test.index, y_pred, label='Predicted', color='orange')
    
    plt.xlabel('Date')
    plt.ylabel('Stock Closing Price')
    day_str = "Day" if time_step == 1 else "Days"
    plt.title(model_name + f' - Actual vs Predicted Closing Prices After {time_step} ' + day_str)
    plt.legend()
    plt.tight_layout()
    
    # Return base64 encoded image
    if return_file:
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        
        import base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_str}"
    else: 
        plt.show()
    
# For testing
if __name__ == "__main__":
    # Test all ensemble models
    model_ids = ['tree', 'forest', 'grad', 'vr', 'xg']
    model_names = {
        'tree':  "Ensemble - Decision Tree (GE Stock)",
        'forest': "Ensemble - Random Forest (GE Stock)",
        'grad': "Ensemble - Gradient Boosting (GE Stock)",
        'vr': "Ensemble - Voting Regressor (GE Stock)",
        'xg':  "Ensemble - XGBoost (GE Stock)"
    }

    for model_id in model_ids:
        print(f"Testing model: {model_names[model_id]}")
        
        time_step = 10
        model = get_model(model_id)     

        result = run_model(model, time_step)
        # print(f"RMSE: {result['rmse']:.4f}")
        # print(f"R²: {result['r2']:.4f}")
        # print(f"MAPE: {result['mape']:.4f}%")
        print(f"Metrics - RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}, MAPE: {result['mape']:.4f}%")
        
        print("Testing plot...", end=" ")
        plot(
            model_names[model_id], 
            result["y_test"], 
            result["y_pred"], 
            result["X_test"], 
            time_step,
            return_file=False
        )
        print()