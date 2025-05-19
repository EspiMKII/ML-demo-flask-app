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

    # need this so if none of the metrics are selected, the app wouldn't crash
    rmse, mape, r2, chart_image = None, None, None, None

    # nah im not doing oop in my python we rawdogging 3 types of models
    # 1st is Ensemble models
    if "Ensemble" in model_name:
        from .models.tree_based_ensemble import ensemble
        model_id_map = {
        "Ensemble - Decision Tree": "tree",
        "Ensemble - Random Forest": "forest",
        "Ensemble - Gradient Boosting": "grad",
        "Ensemble - Voting Regressor": "vr",
        "Ensemble - XGBoost": "xg",
        }
        model = ensemble.get_model(model_id_map[model_name])
        result = ensemble.run_model(model, time_step)

        X_test = result['X_test']
        y_test = result['y_test']
        y_pred = result['y_pred']

        chart_image = ensemble.plot(model_name, y_test, y_pred, X_test, time_step)

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