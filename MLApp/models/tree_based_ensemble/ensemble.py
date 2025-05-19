from joblib import parallel_backend, load
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import parallel_backend
from xgboost import XGBRegressor
import seaborn as sns
import sklearn
import os

def prepare_data():
    '''returns: a test set, and the true values of that set'''

    '''stuff i guess buh'''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "../.."))
    csv_path = os.path.join(workspace_root, "data/tree-based-ensemble/ensemble-data.csv")
    df = pd.read_csv(csv_path)
    
    df.drop(['OpenInt', 'Volume'], axis=1, inplace=True)
    df = df.tail(7500)

    '''feature engineering'''

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

    df.drop(['Date', 'Unnamed: 0'], axis=1, inplace=True)
    '''splitting data set'''

    prediction_days = 10
    df['Target'] = df['Close'].shift(-prediction_days)
    df.dropna(inplace=True)

    X = df.drop(columns=['Target'])
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=False)

    return X_test, y_test

def get_model(name):
    model_path = {
        "tree": "decision-tree.pkl",
        "grad": "gradient-boosting.pkl",
        "forest": "random-forest.pkl",
        "vr": "voting-regressor.pkl",
        "xg": "xgboosting.pkl"
    }[name]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_path)

    model = load(model_path)
    return model

def run_model(model, time_step=1):
    X_test, y_test = prepare_data()
    
    # time steps
    step_indices = range(0, len(X_test), time_step)
    X_test_stepped = X_test.iloc[step_indices]
    y_test_stepped = y_test.iloc[step_indices]
    
    y_pred = model.predict(X_test_stepped)
    
    result = {
        "X_test": X_test_stepped,
        "y_pred": y_pred,
        "y_test": y_test_stepped,
        "time_step": time_step
    }
    
    rmse = np.sqrt(mean_squared_error(y_test_stepped, y_pred))
    r2 = r2_score(y_test_stepped, y_pred)
    mape = mean_absolute_percentage_error(y_test_stepped, y_pred)
    result["rmse"] = rmse
    result["r2"] = r2
    result["mape"] = mape
    
    return result

def plot(model_name, y_test, y_pred, X_test, time_step=1, return_file=True):
    '''
    it doesn't just plot,
    it also returns a base64 encoded image (str), to be displayed in html
    power of HTML BABY
    '''
    results = X_test.copy()
    results['Actual'] = y_test
    results['Predicted'] = y_pred
    results = results[['Actual', 'Predicted']]

    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Actual'], label='Actual', color='blue')
    plt.plot(results.index, results['Predicted'], label='Predicted', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Stock Closing Price')
    plt.title(model_name + f' - Actual vs Predicted Closing Prices (Every {time_step} Days)')
    plt.legend()
    plt.tight_layout()

    if return_file:
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)

        import base64
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

        return f"data:image/png;base64,{img_str}"
    else: plt.show()
    

if __name__ == "__main__":
    tree = get_model("tree")
    forest = get_model("forest")
    grad = get_model("grad")
    vr = get_model("vr")
    xg = get_model("xg")

    time_step = int(input("Enter time step (1, 5, 10 or 365): ") or 1)

    result_tree = run_model(tree, time_step)
    result_forest = run_model(forest, time_step)
    result_grad = run_model(grad, time_step)
    result_vr = run_model(vr, time_step)
    result_xg = run_model(xg, time_step)

    results = {
        "Decision Tree": result_tree,
        "Random Forest": result_forest,
        "Gradient Boosting": result_grad,
        "Voting Regressor": result_vr,
        "XGBoost": result_xg
    }

    for name, result in results.items():
        print(name)
        print(f"RMSE: {result['rmse']}")
        print(f"MAPE: {result['mape']}")
        print(f"R^2: {result['r2']}")
        print()
        plot(name, result["y_test"], result["y_pred"], result["X_test"], time_step, return_file=False)