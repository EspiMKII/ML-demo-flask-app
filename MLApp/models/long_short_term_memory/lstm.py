import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import keras
import matplotlib.dates as mdates
import os
import io
import base64

def get_model(ticker='GE', time_step=10):
    """
    Load a saved regression model based on model ID, ticker, and time step.
    
    Args:
        ticker (str): 'GOOG' or 'GE', default 'GE'
        time-step (int): time step, default 10
    Returns:
        Any: some keras model
    """

    # Get the path to the model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = ticker.lower() + '_' + str(time_step) + '.keras'
    model_path = os.path.join(script_dir, model_name)

    # Load the model with keras.load_model()
    return keras.saving.load_model(model_path)

def prepare_data(time_step=10, ticker='GE'):
    """
    Prepare ticker data for lstm model predictions.
    
    Args:
        time_step (int): Time step, default 10
        ticker (str): 'GOOG' or 'GE', default 'GOOG'
    
    Returns:
        tuple: 
            - X_test (): for prediction, normalized
            - y_test (pd.Series): for actual data
            - scaler_list (list): to denormalize a model's normalized predictions
    """
    # Some parameters
    train_percentage = 0.75
    valid_percentage = 0.05

    input_seq_length = 60
    output_offset = time_step

    # get the path of the data file and load the data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data', 'long-short-term-memory')

    if ticker == 'GE':
        data_path = os.path.join(data_dir, 'ge.us.txt')
    else: # 'GOOG'
        data_path = os.path.join(data_dir, 'googl.us.txt')

    df = pd.read_csv(data_path, delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
    df = df.sort_values('Date')
    df = df.set_index('Date')

    # close_prices will be tremendously useful
    close_prices = df.loc[:, 'Close']

    # Here is 1 of the outputs; fortunately, it has Date as index, good for plotting!
    valid_breakpoint = round(len(close_prices) * (train_percentage + valid_percentage))
    y_test = close_prices[valid_breakpoint:]

    """and now all the df gymnastics to obtain X_series...
    ripped straight from the notebook because i don't have the brain to think about it"""
    inputs = close_prices[len(close_prices) - len(y_test) - input_seq_length - output_offset:].values
    # print(inputs.shape) returns 1D array (2872, )
    inputs = inputs.reshape(-1, 1)
    # print(inputs.shape) returns 2D array (2872, 1) so like [ [a], [b], [c], ....]
    X_test, scaler_list = [], []
    # We are going to normalize each chunk of inputs at a time, and scaler_list helps us store data on how we normalized the chunk
    # so we can denormalize them later
    for i in range(input_seq_length + output_offset, len(inputs)):
        temp = MinMaxScaler()
        temp_array = inputs[i - input_seq_length - output_offset:i - output_offset, 0].reshape(-1, 1)
        # for each iteration, take a chunk out of inputs with size input_seq_length, and shape it into a 1D array
        X_test.append(temp.fit_transform(temp_array).reshape(-1))
        # so after the for loop, X_test would be a 2D LIST (not np.array yet)
        # the chunk we just took out needs to be normalized, of course, this is lstm
        scaler_list.append(temp)
        # the state of the normalizer normalizing the current chunk gets saved
    X_test = np.array(X_test)
    # NOW X_test is a 2D np.array
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    # we turn X_test into a 3D array, FINALLY the thing that the model actually accepts

    return X_test, y_test, scaler_list

def run_model(model, time_step=10, ticker='GE'):
    """
    Run lstm model prediction on test data.
    
    Args:
        model (Any): Loaded scikit-learn model
        time_step (int): Time step
        ticker (str): 'GOOG' or 'GE', default 'GE'
    
    Returns:
        dict: Results dictionary with predictions and metrics
    """
    # Prepare data
    X_test, y_test, scaler_list = prepare_data(time_step, ticker)

    # Make predictions & Calculate metrics
    y_pred = model.predict(X_test)
    for v in range(len(y_pred)):
        y_pred[v] = scaler_list[v].inverse_transform(np.array(y_pred[v]).reshape(-1, 1)).reshape(-1)
        # and this is the denormalization!
    y_pred = np.array(y_pred).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = r2_score(y_test, y_pred)

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
    
    # # Plot settings
    # plt.figure(figsize=(12, 6))
    # plt.plot(results.index, results['Actual'], label='Actual', color='blue')
    # plt.plot(results.index, results['Predicted'], label='Predicted', color='orange')
    # plt.xlabel('Time')
    # plt.ylabel('Ticker Price')
    # day_str = "Day" if time_step == 1 else "Days"
    # ticker_str = ticker + ' Ticker '
    # plt.title(f'{model_name} - Actual vs Predicted ' + ticker_str + f'Prices After {time_step} ' + day_str) 
    # plt.legend()
    # plt.tight_layout()

    # # Plot settings
    # plt.figure(figsize=(12, 6))

    # Create single figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert index to datetime if it isn't already
    results.index = pd.to_datetime(results.index)
    
    # Set up the plot
    ax.plot(results.index, results['Actual'], label='Actual', color='blue')
    ax.plot(results.index, results['Predicted'], label='Predicted', color='orange')
    
    # Format x-axis to show dates properly
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.xticks(rotation=45)
    ax.tick_params(axis='x', rotation=45)
    
    # plt.xlabel('Time')
    # plt.ylabel('Ticker Price')
    # day_str = "Day" if time_step == 1 else "Days"
    # ticker_str = ticker + ' Ticker '
    # plt.title(f'{model_name} - Actual vs Predicted ' + ticker_str + f'Prices After {time_step} ' + day_str)
    # plt.legend()
    # plt.tight_layout()
    ax.set_xlabel('Time')
    ax.set_ylabel('Ticker Price')
    day_str = "Day" if time_step == 1 else "Days"
    ticker_str = ticker + ' Ticker '
    ax.set_title(f'{model_name} - Actual vs Predicted ' + ticker_str + f'Prices After {time_step} ' + day_str)
    ax.legend()
    fig.tight_layout()
    
    # Return base64 encoded image
    if return_file:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    else:
        plt.show()

# For Testing
if __name__ == "__main__":
    time_steps = [1, 10, 30]
    tickers = ['GOOG', 'GE']
    model_name = 'LSTM'
    from itertools import product
    for (time_step, ticker) in product(time_steps, tickers):
        print(f'Testing with times steps = {time_step}, ticker = ' + ticker)

        model = get_model(ticker, time_step)

        result = run_model(model, time_step, ticker)
        print(f"Metrics - RMSE: {result['rmse']:.4f}, R²: {result['r2']:.4f}, MAPE: {result['mape']:.4f}%")

        
        print("Testing plot...", end=" ")
        plot(
            model_name, 
            result["y_test"], 
            result["y_pred"], 
            ticker,
            time_step,
            return_file=False
        )
        print()