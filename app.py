from turtle import bgcolor
from dash import Dash, html, dcc
from matplotlib.pyplot import autoscale, legend
import plotly.express as px
import pandas as pd
from stops import stop_words
import re
import nltk
from nltk.util import bigrams, trigrams
from wordcloud import WordCloud
import numpy as np
from PIL import Image

from io import BytesIO
import base64

from states import us_state


app = Dash(__name__)

## data process
ufo = pd.read_csv(
    "data/ufo-scrubbed-geocoded-time-standardized.csv",
    names=[
        "date",
        "city",
        "state",
        "country",
        "shape",
        "distance",
        "duration",
        "account",
        "dateAcct",
        "lat",
        "lon",
    ],
)
ufo_usa = pd.DataFrame(ufo[ufo.country == "us"])
ufo_usa.loc[ufo.lat == "33q.200088", "lat"] = 33.200088
ufo_usa["lat"] = ufo_usa["lat"].apply(pd.to_numeric)
ufo_usa.account = ufo_usa["account"].str.wrap(30)
ufo_usa["state"] = ufo_usa["state"].str.upper()
ufo_usa["city"] = ufo_usa["city"].str.upper()

cleanList = {
    "\\n": "<br>",
    "&quot;": "'",
    "&#39": "'",
    "&#44": '"',
    "&#33": "!",
    "&#63": "?",
}
for key, value in cleanList.items():
    print(value)
    ufo_usa.account = ufo_usa.account.str.replace(key, value, regex=True)


## clean the accounts


usa_state = "AZ"


def shapes(state):
    data = ufo_usa[ufo_usa.state == state]
    return data["shape"].value_counts().head(7)


def nlp(state):
    data = ufo_usa[ufo_usa.state == state]
    big_string = ""
    for text in data.account:
        if type(text) == str:
            text_filter = re.sub("[^a-zA-Z0-9\s]", " ", text)
            big_string += text_filter.lower()

    return [word for word in big_string.split() if word not in stop_words]


bigram_fb = list(bigrams(nlp(usa_state)))
trigram_fb = list(trigrams(nlp(usa_state)))

bigram_list = [" ".join(tup) for tup in bigram_fb]
trigram_list = [" ".join(tup) for tup in trigram_fb]

words_singles = pd.Series(nlp(usa_state))
words_2 = pd.Series(bigram_list)
words_3 = pd.Series(trigram_list)


topN = 10

s1 = words_singles.value_counts().head(topN)
s2 = words_2.value_counts().head(topN)
s3 = words_3.value_counts().head(3)


def usa_map(state):
    map_data = ufo_usa[ufo_usa.state == state]
    fig1 = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        hover_name="city",
        zoom=5,
        hover_data={
            "lat": False,
            "lon": False,
            "account": True,
        },
        color_discrete_sequence=["rgb(110, 197, 116)"],
        template="plotly_dark",
    )
    fig1.update_layout(mapbox_style="carto-darkmatter")
    fig1.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 30})
    fig1.update_layout(hoverlabel=dict(font_family="Fira Sans"))
    return fig1


fig3 = px.bar(
    s1,
    orientation="h",
    color_discrete_sequence=px.colors.sequential.Aggrnyl,
    template="plotly_dark",
    title="Unigram Distribution",
)
fig3.update_layout(
    font=dict(
        size=16,
    )
)
fig3.update_yaxes(autorange="reversed")
fig3a = px.treemap(
    title="Unigram Distribution",
    path=[px.Constant("all ngrams"), s1.index],
    color=s1,
    values=s1,
    color_continuous_scale="Aggrnyl",
    template="plotly_dark",
)
fig3a.update_layout(
    font=dict(
        size=16,
    )
)
fig4 = px.bar(
    s2,
    orientation="h",
    color_discrete_sequence=px.colors.sequential.Aggrnyl,
    template="plotly_dark",
    title="Bigram Distribution",
)
fig4.update_layout(
    font=dict(
        size=16,
    )
)
fig4.update_yaxes(autorange="reversed")
fig4a = px.treemap(
    template="plotly_dark",
    path=[px.Constant("all ngrams"), s2.index],
    color=s2,
    values=s2,
    color_continuous_scale="Aggrnyl",
    title="Bigram Distribution",
)
fig4a.update_layout(
    font=dict(
        size=16,
    )
)


def plot_wordcloud(data):
    mask = np.array(Image.open("assets/img/ufo_mask.jpg"))
    word_cloud = WordCloud(
        background_color="rgb(17, 17, 17)",
        max_words=2000,
        width=mask.shape[1],
        height=mask.shape[0],
        random_state=42,
        mask=mask,
        colormap="YlGn",
        font_path="assets/fonts/FiraSans-Black.ttf",
    )

    word_cloud.generate(",".join(data))

    return word_cloud.to_image()


def make_image(words):
    img = BytesIO()
    plot_wordcloud(words).save(img, format="PNG")
    return "data:image/png;base64,{}".format(base64.b64encode(img.getvalue()).decode())


make_image(nlp(usa_state))

## app layout

app.layout = html.Div(
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.H1(children="UFO Sightings in the United States "),
                                html.Div(
                                    """
      Map and text analysis of NUFORC ufo reports (1949-2013) 
     """
                                ),
                            ],
                            className="header-txt",
                        ),
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div("Select US State"),
                                        dcc.Dropdown(
                                            options=us_state,
                                            value="Texas",
                                            id="stateDrop",
                                        ),
                                    ], id='state-input'
                                )
                            ]
                        ),
                    ],
                    className="header",
                ),
            ],
            className="wrapper",
        ),
        html.Div(
            children=[
                dcc.Graph(id="map", figure=usa_map(usa_state)),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.H2("UFO Wordcloud"),
                                        html.Img(
                                            id="wordcloud",
                                            src=make_image(nlp(usa_state)),
                                        ),
                                    ],
                                    className="img-wrap",
                                ),
                                # dcc.Graph(id="fig2", figure=fig2),
                            ],
                            className="section-2",
                        ),
                        html.Div(
                            children=[
                                dcc.Tabs(
                                    parent_className="custom-tabs",
                                    className="custom-tabs-container",
                                    children=[
                                        dcc.Tab(
                                            label="Bar Chart",
                                            children=[
                                                dcc.Graph(id="fig3", figure=fig3),
                                            ],
                                            className="custom-tab",
                                            selected_className="custom-tab--selected",
                                        ),
                                        dcc.Tab(
                                            label="Tree Diagram",
                                            children=[
                                                dcc.Graph(id="fig3a", figure=fig3a),
                                            ],
                                            className="custom-tab",
                                            selected_className="custom-tab--selected",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            children=[
                                dcc.Tabs(
                                    parent_className="custom-tabs",
                                    className="custom-tabs-container",
                                    children=[
                                        dcc.Tab(
                                            label="Bar Chart",
                                            children=[
                                                dcc.Graph(id="fig4", figure=fig4),
                                            ],
                                            className="custom-tab",
                                            selected_className="custom-tab--selected",
                                        ),
                                        dcc.Tab(
                                            label="Tree Diagram",
                                            children=[
                                                dcc.Graph(id="fig4a", figure=fig4a),
                                            ],
                                            className="custom-tab",
                                            selected_className="custom-tab--selected",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                    className="wrapper",
                ),
            ],
            className="content",
        ),
    ],
    className="app",
)

if __name__ == "__main__":
    app.run_server(debug=True)
