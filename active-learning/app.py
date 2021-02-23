import io
import base64
from io import BytesIO
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.dependencies import Input, Output, State
import pickle
import plotly.express as px
import numpy as np

import lime
import lime.lime_tabular

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

label_mapping = {
    1: "Class 1",
    2: "Class 2",
    3: "Class 3"
}

# train and test switch roles to present active learning
wine_test = pd.read_csv("./wine/wine-test.csv", sep=';')
X_train = wine_test.iloc[:,1:]
y_train = wine_test.iloc[:,0]

wine_train = pd.read_csv("./wine/wine-train.csv", sep=';')
X_test = wine_train.iloc[:,1:]
y_test = wine_train.iloc[:,0]

svc = make_pipeline(
    StandardScaler(),
    SVC(probability=True)
)

svc.fit(X_train, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names=list(X_train.columns), class_names=label_mapping.values(), discretize_continuous=True)

intro_text = """
This app demonstrates active learning process.
"""

pca = PCA(
    n_components=2
)

def create_pca_graph(data):
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    colors = px.colors.qualitative.Pastel
    traces = []
    for i, key in enumerate(label_mapping.keys()):
        # Labeled data
        idx = np.where(y_train == key)
        x = X_train_pca[idx, 0].flatten()
        y = X_train_pca[idx, 1].flatten()
        if data in ["Labeled", "All"]:
            opacity = 0.9
            hoverinfo = "all"
            showlegend = True
            visible = True
        else:
            opacity = 0.5
            hoverinfo = "none"
            showlegend = False
            visible = "legendonly"
        trace = {
            "x": x,
            "y": y,
            "mode": "markers",
            "type": "scattergl",
            "marker": {"color": colors[i], "size": 10},
            "name": label_mapping[key],
            "text": label_mapping[key],
            "customdata": list(zip([True] * len(idx[0]), idx[0])),
            "opacity": opacity,
            "hoverinfo": hoverinfo,
            "visible": visible,
            "showlegend": showlegend,
            "selected": {"marker": {"size": 25, "color": colors[i]}},
        }
        traces.append(trace)

    for i, key in enumerate(label_mapping.keys()):
        # Not labeled data
        idx = np.where(y_test == key)
        x = X_test_pca[idx, 0].flatten()
        y = X_test_pca[idx, 1].flatten()
        if data in ["Not labeled", "All"]:
            hoverinfo = "all"
            showlegend = True if data == "Not labeled" else False
            visible = True
        else:
            hoverinfo = "none"
            showlegend = False
            visible = "legendonly"
        trace = {
            "x": x,
            "y": y,
            "mode": "markers",
            "type": "scattergl",
            "marker": {
                "color": colors[i],
                "size": 10
            },
            "name": label_mapping[key],
            "text": label_mapping[key],
            "customdata": list(zip([False] * len(idx[0]), idx[0])),
            "opacity": 0.3,
            "hoverinfo": hoverinfo,
            "visible": visible,
            "showlegend": showlegend,
            "selected": {"marker": {"size": 25, "color": colors[i]}},
        }
        traces.append(trace)


    layout = {
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "clickmode": "event+select"
    }
    figure = {"data": traces, "layout": layout}
    return figure


app = dash.Dash(name=__name__)

server = app.server

app.css.config.serve_locally = False
app.config.suppress_callback_exceptions = True

header = html.Div(
    id="app-header",
    children=[
        html.Img(src=app.get_asset_url("dash-logo.png"), className="logo"),
        "Active Learning with LIME: wine dataset",
    ],
)

app.layout = html.Div(
    children=[
        header,
        html.Br(),
        html.Details(
            id="intro-text",
            children=[html.Summary(html.B("About This App")), dcc.Markdown(intro_text)],
        ),
        html.Div(
            id="app-body",
            children=[
                html.Div(
                    id="control-card",
                    children=[
                        html.Span(
                            className="control-label",
                            children="Display Train or Test Data",
                        ),
                        dcc.Dropdown(
                            id="train-test-dropdown",
                            className="control-dropdown",
                            options=[
                                {"label": i, "value": i}
                                for i in ["Labeled", "Not labeled", "All"]
                            ],
                            value="Labeled",
                        ),
                        html.Span(
                            className="control-label", children="Upload data (csv)"
                        ),
                        dcc.Upload(
                            id="img-upload",
                            className="upload-component",
                            children=html.Div(
                                ["Drag and Drop or ", html.A("Select Files")]
                            ),
                        ),
                        html.Div(id="output-img-upload"),
                    ],
                ),
                html.Div(
                    style={"width": "60vw"},
                    children=[
                        html.Div(
                            id="pca-graph-div",
                            children=[
                                html.Div(
                                    id="pca-graph-outer",
                                    children=[
                                        # html.Div(
                                        # id="intro-text",
                                        # children=dcc.Markdown(intro_text),
                                        # ),
                                        html.H3(
                                            className="graph-title",
                                            children="Wine Dataset Reduced to 2 Dimensions with PCA",
                                        ),
                                        dcc.Graph(
                                            id="pca-graph",
                                            figure=create_pca_graph("Not labeled"),
                                        ),
                                    ],
                                )
                            ],
                        ),
                        html.Div(
                            id="image-card-div",
                            style={"width": "75vw"},
                            children=[
                                html.Div(
                                    id="prediction-div",
                                    className="img-card",
                                    children=[
                                        html.Div(id="choose-label"),
                                        html.Div(
                                            id="selected-data-graph-outer",
                                            children=[
                                                html.Div(
                                                    children=[
                                                        html.Div(
                                                            id="prediction",
                                                            children=[
                                                                "Click on a point to display the prediction",
                                                                html.Br()
                                                            ],
                                                        ),
                                                    ],
                                                    style={"height": "20%"},
                                                ),
                                                html.Br()
                                            ],
                                        ),
                                        html.Div(
                                            id="explanation-graphs"
                                        )
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                html.Div(
                    style={"width": "15vw"},
                    id="hover-point-outer",
                    className="img-card",
                    children=[
                        html.B(
                            "Hover Point:", style={"height": "20%"}
                        ),
                        html.Br(),
                        html.Div(
                            id="hover-point-graph"
                        ),
                    ],
                )
            ],
        ),
    ]
)


@app.callback(
    Output("output-img-upload", "children"),
    [Input("img-upload", "contents")],
    [State("img-upload", "filename"), State("img-upload", "last_modified")],
)
def display_uploaded_csv(contents, fname, date):
    if contents is not None:
        content_type, content_string = contents.split(",")
        csv = io.BytesIO(base64.b64decode(content_string))
        csv = csv.read()
        csv = csv.decode('UTF-8')
        csv = io.StringIO(csv)
        data_frame = pd.read_csv(csv, sep=";")
        
        children = [
            "Your uploaded data: ",
            dash_table.DataTable(
                id="upload_table",
                style_table={'overflowX': 'auto', 'overflowY': 'auto'},
                editable=True,
                page_size=15,
                columns=[{'name': c, 'id':c } for c in data_frame.columns],
                data=data_frame.to_dict(orient='records')),
            f"({len(data_frame)} records)",
            html.Button(
                id="add-labeled", children="Add as Labeled", n_clicks=0
            ),
            html.Button(
                id="add-not-labeled", children="Add as Not labeled", n_clicks=0
            ),
            html.Button(
                id="clear-button", children="Remove Uploaded CSV", n_clicks=0
            ),
        ]
        return children


@app.callback(
	    Output("img-upload", "contents"), 
	    [Input("clear-button", "n_clicks"), Input("add-labeled", "n_clicks"), Input("add-not-labeled", "n_clicks"), 
	    Input("upload_table","data"), Input("upload_table","columns")])
def clear_upload(n_clicks_clear, n_clicks_add_labeled, n_clicks_add_not_labeled, rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    if n_clicks_add_labeled >= 1:
        add_to_labeled(df)
    if n_clicks_add_not_labeled >= 1:
        add_to_not_labeled(df)
    if n_clicks_clear >= 1 or n_clicks_add_labeled >= 1 or n_clicks_add_not_labeled >= 1:
        return None
    raise dash.exceptions.PreventUpdate
    
def add_to_labeled(df):
    global X_train, y_train, explainer
    
    x = df.iloc[:,1:]
    y = df.iloc[:,0]
    X_train = X_train.append(x, ignore_index=True)
    y_train = y_train.append(y, ignore_index=True)
    svc.fit(X_train, y_train)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names=list(X_train.columns), class_names=label_mapping.values(), discretize_continuous=True)

def add_to_not_labeled(df):
    global X_test, y_test
    
    x = df.iloc[:,1:]
    y = df.iloc[:,0]
    X_test = X_test.append(x, ignore_index=True)
    y_test = y_test.append(y, ignore_index=True)

    
@app.callback(
    Output("pca-graph", "figure"),
    [Input("train-test-dropdown", "value")]
)
def display_train_test(value):
    return create_pca_graph(value)


@app.callback(Output("hover-point-graph", "children"), [Input("pca-graph", "hoverData")])
def display_selected_point(hoverData):
    if not hoverData:
        return printPoint(X_train.iloc[0,:])
    is_train, idx = hoverData["points"][0]["customdata"]
    X = X_train if is_train else X_test
    return printPoint(X.iloc[idx,:])

def printPoint(point):
    displayable = point.to_string()
    with_br = []
    for line in displayable.split("\n"):
    	with_br.append(line)
    	with_br.append(html.Br())
    return with_br

@app.callback(
    [Output("choose-label", "children"), Output("explanation-graphs", "children"), Output("prediction", "children")],
    [Input("pca-graph", "clickData")],
)
def display_selected_point(clickData):
    if not clickData:
        raise dash.exceptions.PreventUpdate

    is_train, idx = clickData["points"][0]["customdata"]
    X = X_train if is_train else X_test
    x = X.iloc[idx,:]

    exp = explainer.explain_instance(x, svc.predict_proba, num_features=len(X_train.columns), top_labels=len(label_mapping), num_samples=100)
    
    obj = html.Iframe(
            # Javascript is disabled from running in an Iframe for security reasons
            # Static HTML only!!!
            srcDoc=exp.as_html(show_table=False),
            width='100%',
            height='450px',
            style={'border': '0'},
        )
    
    drop = dcc.Dropdown(
        id='label-dropdown',
        options=[
            {'label': v, 'value': k}
            for k, v in label_mapping.items()
        ],
        value=None
    )
    
    return [
        drop,
        obj,
        [
            f"True label: {label_mapping[y_train.iloc[idx]] if is_train else 'unknown'}" 
        ]
    ]

@app.callback(Output('label-dropdown', 'value'), [Input("pca-graph", "clickData"), Input('label-dropdown', 'value')])
def update_output(clickData, value):
    if not clickData:
        raise dash.exceptions.PreventUpdate
    
    if value is None:
        return value
    
    is_train, idx = clickData["points"][0]["customdata"]
    
    y = y_train if is_train else y_test
    X = X_train if is_train else X_test
    
    y.drop(idx, inplace=True)
    X.drop(idx, inplace=True)
    
    return value


if __name__ == "__main__":
    app.run_server(debug=True)
