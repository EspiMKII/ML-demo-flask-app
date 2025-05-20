from flask import (
    Blueprint, redirect, render_template, url_for, request, session
)

bp = Blueprint('param-input', __name__, url_prefix = '/param-input')

@bp.route('/', methods=('GET', 'POST'))
def index():
    models = [
        "Ensemble - Decision Tree (GE Stock)",
        "Ensemble - Random Forest (GE Stock)",
        "Ensemble - Gradient Boosting (GE Stock)",
        "Ensemble - Voting Regressor (GE Stock)",
        "Ensemble - XGBoost (GE Stock)",
        "Regression - Linear Regression (GE Stock)",
        "Regression - Ridge Regression (GE Stock)",
        "Regression - Lasso Regression (GE Stock)",
        "Regression - Support Vector Regression (GE Stock)",
        "Regression - Linear Regression (GOOG Stock)",
        "Long-Short Term Memory"
    ]
    time_step = [1, 5, 10, 365]
    metrics = [
        {'id': 'rmse', 'name': 'Root Mean Square Error'},
        {'id': 'r2', 'name': 'R^2 Score'},
        {'id': 'mape', 'name': 'Mean Absolute Percentage Error'}
    ]

    if request.method == 'POST':
        pass
        session['model'] = request.form.get('model')
        session['time_step'] = int(request.form.get('time_step'))
        session['metrics'] = request.form.getlist('metrics')

        return redirect(url_for('output.results'))

    return render_template('input.html',
                           models=models,
                           time_step=time_step,
                           metrics=metrics
                           )