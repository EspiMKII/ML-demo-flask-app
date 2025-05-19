import os
from flask import Flask, redirect, url_for

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev'
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import param_input
    app.register_blueprint(param_input.bp)

    from . import output
    app.register_blueprint(output.bp)

    @app.route('/')
    def index():
        return redirect(url_for('param-input.index'))

    return app