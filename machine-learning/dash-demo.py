# some toy data
import numpy as np
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn import metrics
from sklearn.datasets import make_moons

X_train, y_train = make_moons(n_samples=20000, shuffle=True, noise=0.5, random_state=10)
X_test, y_test = make_moons(n_samples=10000, shuffle=True, noise=0.5, random_state=10)
xgb.set_config(verbosity=0)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)


from time import sleep

import dash
import plotly
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly import graph_objects as go
from sklearn.metrics import f1_score

acc_fig, f2_fig, recall_fig = None, None, None
accs, f2s, recalls = [], [], []
from jupyter_dash import JupyterDash

app = JupyterDash(__name__)

app.layout = html.Div(
    [
        html.H1("Evolution of Hyperparameter Tuning"),
        html.Div(id="live-output"),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="alpha", figure={}),
                    ],
                    style={"width": "33%", "display": "inline-block"},
                ),
                html.Div(
                    [dcc.Graph(id="eta", figure={})],
                    style={"width": "33%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="max_depth", figure={}),
                    ],
                    style={"width": "33%", "display": "inline-block"},
                ),
            ],
            className="row",
        ),
        html.Div([dcc.Graph(id="F1", figure={})]),
        # relationshiups
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="alpha_loss", figure={}),
                    ],
                    style={"width": "33%", "display": "inline-block"},
                ),
                html.Div(
                    [dcc.Graph(id="eta_loss", figure={})],
                    style={"width": "33%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="max_depth_loss", figure={}),
                    ],
                    style={"width": "33%", "display": "inline-block"},
                ),
            ],
            className="row",
        ),
        dcc.Interval(id="interval-component", interval=100, n_intervals=0),
    ]
)
SPACE = {
    "n_jobs": 0,
    "objective": "binary:logistic",
    "subsample": hp.uniform("subsample", 0.5, 1),
    "min_child_weight": hp.uniform("min_child_weight", 1, 10),
    "eta": hp.uniform("eta", 0, 1),
    "max_depth": scope.int(hp.quniform("max_depth", 1, 12, 1)),
    "min_split_loss": hp.uniform("min_split_loss", 0, 0.2),
    "num_parallel_tree": hp.choice("n_estimators", np.arange(1, 10, 1)),
    "lambda": hp.uniform("lambda", 0, 1),
    "alpha": hp.uniform("alpha", 0, 1),
    "booster": hp.choice("booster", ["gbtree", "gblinear", "dart"]),
    "tree_method": hp.choice("tree_method", ("approx", "hist")),
}


def objective(space):

    # this is a "hack" since I want to pass obj in as
    #   a member of the search space
    #   but treat it ALONE as a keyword argument
    #   may increase computation time ever-so-slightly
    params = {}
    for k, v in space.items():
        if k != "obj":
            params[k] = v

    # train the classifier
    booster = xgb.train(
        params,
        dtrain,
    )

    y_pred = booster.predict(dtest)
    y_pred[y_pred < 0.5] = 0
    y_pred[y_pred >= 0.5] = 1

    # evaluate and return
    # note we want to maximize F1 and hence MINIMIZE NEGATIVE F1
    return {
        "loss": -f1_score(y_pred, y_test),
        "status": STATUS_OK,
        "accuracy": metrics.accuracy_score(dtest.get_label(), y_pred),
        "recall": metrics.recall_score(dtest.get_label(), y_pred),
        "f2": metrics.fbeta_score(dtest.get_label(), y_pred, beta=2),
    }


trials = Trials()


@app.callback(
    [
        Output("live-output", "children"),
        Output("alpha", "figure"),
        Output("eta", "figure"),
        Output("max_depth", "figure"),
        Output("F1", "figure"),
        Output("alpha_loss", "figure"),
        Output("eta_loss", "figure"),
        Output("max_depth_loss", "figure"),
    ],
    [Input("interval-component", "n_intervals")],
)
def update_graph(n):

    fig1, fig2, fig3 = None, None, None
    fig1_loss, fig2_loss, fig3_loss = None, None, None
    fig_loss = None

    losses = []
    try:

        n_trials = len(trials)
        alphas = [i["misc"]["vals"]["alpha"][0] for i in trials.trials]
        lambdas = [i["misc"]["vals"]["lambda"][0] for i in trials.trials]
        max_depths = [i["misc"]["vals"]["max_depth"][0] for i in trials.trials]
        etas = [i["misc"]["vals"]["eta"][0] for i in trials.trials]

        # fig = go.Figure(go.Scatter(x=np.linspace(0, n_trials), y=alphas), layout=go.Layout(autosize=False, width=600, height=600, title='Alpha'))
        fig1 = go.Figure(
            go.Scatter(x=np.linspace(0, n_trials), y=alphas, mode="lines+markers"),
            layout=go.Layout(
                title="Alpha Evolution",
                xaxis=dict(title="n_trials"),
                yaxis=dict(title="alpha"),
            ),
        )
        fig2 = go.Figure(
            go.Scatter(x=np.linspace(0, n_trials), y=etas, mode="lines+markers"),
            layout=go.Layout(
                title="ETA Evolution",
                xaxis=dict(title="n_trials"),
                yaxis=dict(title="Learning Rate"),
            ),
        )
        fig3 = go.Figure(
            go.Scatter(x=np.linspace(0, n_trials), y=max_depths, mode="lines+markers"),
            layout=go.Layout(
                title="Depth Evolution",
                xaxis=dict(title="n_trials"),
                yaxis=dict(title="Max Depth"),
            ),
        )

        losses = trials.losses()

        fig_loss = go.Figure(
            go.Scatter(
                x=np.linspace(0, n_trials),
                y=losses,
                mode="lines+markers",
            ),
            layout=go.Layout(
                xaxis=dict(title="n_trials"), yaxis=dict(title="F2 Metric")
            ),
        )

        fig1_loss = go.Figure(
            go.Scatter(x=alphas, y=losses, mode="markers"),
            layout=go.Layout(
                title="F2 vs Alpha", xaxis=dict(title="alpha"), yaxis=dict(title="F2")
            ),
        )
        fig2_loss = go.Figure(
            go.Scatter(x=etas, y=losses, mode="markers"),
            layout=go.Layout(
                title="F2 vs Learning Rate",
                xaxis=dict(title="eta"),
                yaxis=dict(title="F2"),
            ),
        )
        fig3_loss = go.Figure(
            go.Scatter(x=max_depths, y=losses, mode="markers"),
            layout=go.Layout(
                title="F2 vs Depth",
                xaxis=dict(title="max_depth"),
                yaxis=dict(title="F2"),
            ),
        )
    except Exception as e:
        print(e)

    return (
        html.Span(f"Current Loss: {trials.losses()[-1]}"),
        fig1,
        fig2,
        fig3,
        fig_loss,
        fig1_loss,
        fig2_loss,
        fig3_loss,
    )


app.run_server()

best_hyperparams = fmin(
    space=SPACE,
    algo=tpe.suggest,
    fn=objective,
    max_evals=2000,  # this would be 100, 500 or something higher when actually optimizing
    trials=trials,
)
