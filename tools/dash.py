import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Dash
from dash.dependencies import Input, Output

app = Dash(__name__)
df = pd.read_csv('./airline_2m.csv')
app.layout = html.Div([
    dcc.Graph(id='my-bar-graph'),
    dcc.Dropdown(id='my-dropdown', options= df.state.unique())
])

@app.callback(
    Output(component_id='my-bar-graph', component_property='figure'),
    Input(component_id='my-dropdown', component_property='value')
)
def update_state(my_dropdown_value):
    dff = df[df.state==my_dropdown_value]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=dff.airport,
            y=dff.cnt
        )
    )

    return fig
app.run_server(port=1080, debug=True)
