import base64
from io import BytesIO

import numpy as np
from keras.models import load_model
from PIL import Image
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import pickle
import plotly.express as px

import lime
import lime.lime_tabular

label_mapping = {
    1: "Class 1",
    2: "Class 2",
    3: "Class 3"
}

#label_mapping = { val: f'Quality {val}' for val in range(11)}

import pandas as pd
wine = pd.read_csv("./wine/wine.data", sep=';')
X = wine.iloc[:,1:]
Y = wine.iloc[:,0]

#wine = pd.read_csv("./wine/winequality-white.csv", sep=';')
#X = wine.iloc[:,:11]
#Y = wine.iloc[:,11]

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

svc = make_pipeline(
    MinMaxScaler(),
    SVC(probability=True)
)

svc.fit(X_train, y_train)

explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names=list(X_train.columns),    class_names=label_mapping.values(), discretize_continuous=True)

intro_text = """
This app demonstrates active learning process. //TODO
"""

import numpy as np
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2
)

X_hat = tsne.fit_transform(X)

def create_tsne_graph(data, uploaded_point=None):
    colors = px.colors.qualitative.Pastel
    traces = []
    for i, key in enumerate(label_mapping.keys()):
        # Labeled data
        idx = np.where(y_train == key)
        x = X_hat[idx, 0].flatten()
        y = X_hat[idx, 1].flatten()
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
            "marker": {"color": colors[i], "size": 3},
            "name": label_mapping[key],
            "text": label_mapping[key],
            "customdata": idx[0],
            "opacity": opacity,
            "hoverinfo": hoverinfo,
            "visible": visible,
            "showlegend": showlegend,
            "selected": {"marker": {"size": 10, "color": "black"}},
        }
        traces.append(trace)

    for i, key in enumerate(label_mapping.keys()):
        # Not labeled data
        idx = np.where(y_test == key)
        x = X_hat[idx, 0].flatten()
        y = X_hat[idx, 1].flatten()
        if data in ["Not labeled", "All"]:
            opacity = 0.9
            hoverinfo = "all"
            showlegend = True if data == "Not labeled" else False
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
            "marker": {"color": colors[i], "size": 3},
            "name": label_mapping[key],
            "text": label_mapping[key],
            "customdata": idx[0],
            "opacity": opacity,
            "hoverinfo": hoverinfo,
            "visible": visible,
            "showlegend": showlegend,
            "selected": {"marker": {"size": 10, "color": "black"}},
        }
        traces.append(trace)

    annotation = []

    if uploaded_point:
        annotation.append(
            {
                "x": uploaded_point[0][0],
                "y": uploaded_point[0][1],
                "xref": "x",
                "yref": "y",
                "text": "Predicted Embedding for Uploaded Image",
                "showarrow": True,
                "arrowhead": 1,
                "ax": 10,
                "ay": -40,
                "font": {"size": 20},
            }
        )

    layout = {
        "xaxis": {"visible": False},
        "yaxis": {"visible": False},
        "clickmode": "event+select",
        "annotations": annotation,
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
                            id="tsne-graph-div",
                            children=[
                                html.Div(
                                    id="tsne-graph-outer",
                                    children=[
                                        # html.Div(
                                        # id="intro-text",
                                        # children=dcc.Markdown(intro_text),
                                        # ),
                                        html.H3(
                                            className="graph-title",
                                            children="Wine Dataset Reduced to 2 Dimensions with T-SNE",
                                        ),
                                        dcc.Graph(
                                            id="tsne-graph",
                                            figure=create_tsne_graph("Not labeled"),
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
def display_uploaded_img(contents, fname, date):
    if contents is not None:
        original_img, resized_img = parse_image(contents, fname, date)

        img = np.expand_dims(resized_img, axis=0)
        prediction_array = model.predict(img)
        prediction = np.argmax(prediction_array)

        children = [
            "Your uploaded image: ",
            html.Img(className="image", src=original_img),
            "Image fed the model: ",
            html.Img(className="image", src=create_img(resized_img)),
            f"The model thinks this is a {label_mapping[prediction]}",
            html.Button(
                id="clear-button", children="Remove Uploaded Image", n_clicks=0
            ),
        ]
        return children


@app.callback(Output("img-upload", "contents"), [Input("clear-button", "n_clicks")])
def clear_upload(n_clicks):
    if n_clicks >= 1:
        return None
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("tsne-graph", "figure"),
    [Input("train-test-dropdown", "value"), Input("img-upload", "contents")],
    [State("img-upload", "filename"), State("img-upload", "last_modified")],
)
def display_train_test(value, contents, fname, date):
    embedding_prediction = None
    if contents is not None:
        original_img, resized_img = parse_image(contents, fname, date)
        linear_model = pickle.load(
            open("trained_data/linear_model_embeddings.sav", "rb")
        )
        embedding_prediction = linear_model.predict(resized_img.reshape(1, -1)).tolist()
    return create_tsne_graph(value, embedding_prediction)


@app.callback(Output("hover-point-graph", "children"), [Input("tsne-graph", "hoverData")])
def display_selected_point(hoverData):
    if not hoverData:
        return printPoint(X_train.iloc[0,:])
    idx = hoverData["points"][0]["customdata"]
    return printPoint(X.iloc[idx,:])

def printPoint(point):
    displayable = point.to_string()
    with_br = []
    for line in displayable.split("\n"):
    	with_br.append(line)
    	with_br.append(html.Br())
    return with_br

@app.callback(
    [Output("explanation-graphs", "children"), Output("prediction", "children")],
    [Input("tsne-graph", "clickData")],
)
def display_selected_point(clickData):
    if not clickData:
        raise dash.exceptions.PreventUpdate

    idx = clickData["points"][0]["customdata"]
    x = X.iloc[idx,:]

    exp = explainer.explain_instance(x, svc.predict_proba, num_features=len(X.columns), top_labels=len(label_mapping))
    
    obj = html.Iframe(
            # Javascript is disabled from running in an Iframe for security reasons
            # Static HTML only!!!
            srcDoc=exp.as_html(show_table=False),
            width='100%',
            height='450px',
            style={'border': '0'},
        )
    
    return [
        obj,
        [
            f"True label: {label_mapping[Y.iloc[idx]]}" 
        ]
    ]


if __name__ == "__main__":
    app.run_server(debug=False)
