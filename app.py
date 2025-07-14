# app.py
import re
from pathlib import Path
import openai

import pandas as pd
import umap.umap_ as umap
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

import dash
from dash import dcc, html, dash_table, no_update
from dash.dependencies import Input, Output
import plotly.express as px

from helpers import (
    drop_empty_rows,
    drop_empty_columns,
    acronyms_csv_to_dict,
    make_acronym_regex,
    make_theme_title
)

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
REPORT_CSV = DATA_DIR / "airbus_component_failures.csv"
ACRO_CSV   = DATA_DIR / "ASRS_Definitions.csv"
N_TOPICS   = 4

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent /"data"
REPORT_CSV = DATA_DIR /"airbus_component_failures.csv"
ACRO_CSV   = DATA_DIR /"ASRS_Definitions.csv"
N_TOPICS   = 4

# ─── SMALL UTILITIES ────────────────────────────────────────────────────────────
# def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
#     """Remove rows that are entirely NaN or blank."""
#     return df.dropna(how="all")

# def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """Remove columns that are entirely NaN or blank."""
#     return df.dropna(axis=1, how="all")

# def acronyms_csv_to_dict(path: Path) -> dict:
#     df = pd.read_csv(path)
#     return dict(zip(df['Acronym'], df['Definition']))

# def make_acronym_regex(mapping: dict) -> re.Pattern:
#     pat = r'\b(?:' + '|'.join(re.escape(a) for a in mapping) + r')\b'
#     return re.compile(pat)

# def make_theme_title(weights, feature_names):
#     comp_pat = re.compile(r"(^[A-Z]{3,}$)|[\d/#\\-]")
#     idx = weights.argsort()[-15:][::-1]
#     toks = [feature_names[i] for i in idx]
#     important = [t for t in toks if comp_pat.search(t)] + [t for t in toks if not comp_pat.search(t)]
#     return important[0].title() if important else "Miscellaneous"

# ─── LOAD & PREPROCESS RAW CSV ──────────────────────────────────────────────────
df_raw = pd.read_csv(REPORT_CSV)

# Drop artifact header / index column if present
if "Unnamed: 0" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Unnamed: 0"])

# Reset the index, then rename the first real column to 'ACN'
df_raw.reset_index(drop=True, inplace=True)
if df_raw.columns.size >= 1:
    df_raw.rename(columns={df_raw.columns[0]: "ACN"}, inplace=True)

# Drop any rows or columns that are entirely empty
df_raw = drop_empty_rows(df_raw)
df_raw = drop_empty_columns(df_raw)

# ─── EXTRACT YOUR “narrative” TEXTS ─────────────────────────────────────────────
docs = (
    df_raw['Report 1']
      .dropna()
      .loc[lambda s: s.str.lower() != 'narrative']
      .reset_index(drop=True)
)

# Expand acronyms
mapping = acronyms_csv_to_dict(ACRO_CSV)
acro_pat = make_acronym_regex(mapping)
docs = docs.map(lambda txt: acro_pat.sub(lambda m: mapping[m.group(0)], txt))

# ─── LOAD & PREPROCESS ──────────────────────────────────────────────────────────
df_raw = pd.read_csv(REPORT_CSV)

# drop empty / placeholder rows
docs = (
    df_raw['Report 1']
    .dropna()
    .loc[lambda s: s.str.lower() != 'narrative']
    .reset_index(drop=True)
)

# expand acronyms
mapping = acronyms_csv_to_dict(ACRO_CSV)
acro_pat = make_acronym_regex(mapping)
docs = docs.map(lambda txt: acro_pat.sub(lambda m: mapping[m.group(0)], txt))

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english', ngram_range=(1,2),
    max_df=0.5, min_df=5
)
X = vectorizer.fit_transform(docs)

# NMF → topics
nmf = NMF(n_components=N_TOPICS, init="nndsvd", random_state=0)
W = nmf.fit_transform(X)
H = nmf.components_
feature_names = vectorizer.get_feature_names_out()
titles = [make_theme_title(row, feature_names) for row in H]
doc_topics = W.argmax(axis=1)

# UMAP → 2D embedding
reducer = umap.UMAP(
    n_neighbors=15, min_dist=0.1,
    metric='cosine', random_state=0
)
embedding = reducer.fit_transform(X)

# build the DataFrame we’ll plot and table-view
df_plot = pd.DataFrame({
    'UMAP-1':  embedding[:,0],
    'UMAP-2':  embedding[:,1],
    'Theme #': (doc_topics + 1).astype(str),
    'Theme':   [titles[i] for i in doc_topics],
    'Snippet': docs.str.slice(0,120) + '…'
})

df_table = pd.DataFrame({
    'Theme #': (doc_topics + 1),
    'Theme':   [titles[i] for i in doc_topics],
    'Report':  docs,
})

# ─── DASH APP ──────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("ASRS Narrative Clusters (UMAP)"),

    dcc.Graph(id='scatter'),

    html.H2("Selected Reports"),
    dash_table.DataTable(
        id='table',
        columns=[{'name': c, 'id': c} for c in df_table.columns],
        data=df_table.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '5px'},
    ),

    html.H2("Summary of Selected Reports"),
    html.Div(id='summary', style={'whiteSpace': 'pre-wrap', 'padding': '10px', 'border': '1px solid #ccc'}),
])


df_plot['orig_index'] = df_table.index

# initial scatter figure
fig = px.scatter(
    df_plot, x='UMAP-1', y='UMAP-2',
    color='Theme #',
    hover_data=['Theme','Snippet'],
    custom_data=['orig_index'],   
    title="Box-select or lasso-select to filter the table below"
)
# enable box-select by default
fig.update_layout(dragmode='select')

app.layout.children[1].figure = fig

# ─── CALLBACK ─────────────────────────────────────────────────────────────────
from dash import Output, Input

@app.callback(
    [Output('table', 'data'),
     Output('summary', 'children')],
    [Input('scatter', 'selectedData')]
)

def update_table_and_summary(selectedData):
    # 1) Always build your filtered_df
    if not selectedData or 'points' not in selectedData:
        filtered_df = df_table
    else:
        orig_idxs = [p['customdata'][0] for p in selectedData['points']]
        filtered_df = df_table.loc[orig_idxs]

    table_data = filtered_df.to_dict('records')

    # 2) Only call the API if they actually selected something
    if not selectedData or 'points' not in selectedData:
        # don’t overwrite whatever’s in the summary div
        return table_data, no_update

    reports = filtered_df['Report'].tolist()
    if not reports:
        return table_data, "No reports to summarize."

    # 3) Now that we know we have a non-empty selection, call the LLM
    prompt = (
        "You are an expert aviation safety analyst.  "
        "Summarize the following incident reports into a concise executive summary highlighting root causes, DO NOT provide suggestions:\n\n"
            + "\n\n".join(reports)
    )

    openai.api_key = #Place your OpenAI API key here
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
          {"role":"system", "content":"You are a helpful assistant."},
          {"role":"user",   "content":prompt}
        ],
        temperature=0.3,
    )
    summary = resp.choices[0].message.content.strip()

    return table_data, summary


if __name__ == '__main__':
    app.run(debug=True)
