from flask import (
    Blueprint, redirect, render_template, url_for, session
)
import os

bp = Blueprint('output', __name__, url_prefix='/output')

@bp.route('/results', methods=('GET',))
def results():
    pass
    # this is why we used session baby!!!!!
    model_name = session.get('model')
    time_step = session.get('time_step', 1)
    metrics = session.get('metrics', [])
    ticker = session.get('ticker')

    # need this so if none of the metrics are selected, the app wouldn't crash
    rmse, mape, r2, chart_image = None, None, None, None

    # nah im not doing oop in my python we rawdogging 3 types of models
    # 1st is Ensemble models
    if "Ensemble" in model_name:
        from .models.tree_based_ensemble import ensemble
        model_id_map = {
        "Ensemble - Decision Tree": 'tree',
        "Ensemble - Random Forest": 'forest',
        "Ensemble - Gradient Boosting": 'grad',
        "Ensemble - Voting Regressor": 'vr',
        "Ensemble - XGBoost": 'xg',
        }
        model = ensemble.get_model(model_id_map[model_name])
        result = ensemble.run_model(model, time_step)

        chart_image = ensemble.plot(model_name, 
                                    result['y_test'],
                                    result['y_pred'], 
                                    result['X_test'],
                                    time_step)
    
    # 2nd is regression
    elif "Regression" in model_name:
        from .models.regression_v2 import regression_v2
        model_id_map = {
            "Regression - Linear Regression": 'lr',
            "Regression - Ridge Regression": 'ridge',
            "Regression - Lasso Regression": 'lasso',
            "Regression - Support Vector Regression": 'svr'
        }
        model = regression_v2.get_model(model_id_map[model_name], ticker, time_step)
        result = regression_v2.run_model(model, time_step)

        chart_image = regression_v2.plot(model_name, 
                                         result['y_test'], 
                                         result['y_pred'],
                                         ticker,
                                         time_step)

    # 3rd is lstm
    else: # "Long-Short Term Memory"
        from .models.long_short_term_memory import lstm
        model = lstm.get_model(ticker, time_step)
        result = lstm.run_model(model, time_step, ticker)

        chart_image = lstm.plot("Long-Short Term Memory",
                               result['y_test'],
                               result['y_pred'],
                               ticker,
                               time_step)

    if 'rmse' in metrics: rmse = result['rmse']
    if 'mape' in metrics: mape = result['mape']
    if 'r2' in metrics: r2 = result['r2']
    
    return render_template('output.html',
                           model_name=model_name,
                           time_step=time_step,
                           chart_image=chart_image,
                           rmse=rmse,
                           mape=mape,
                           r2=r2)