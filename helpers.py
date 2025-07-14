import pandas as pd
import re
from pathlib import Path

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that are entirely NaN or blank."""
    return df.dropna(how="all")

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are entirely NaN or blank."""
    return df.dropna(axis=1, how="all")

def acronyms_csv_to_dict(path: Path) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df['Acronym'], df['Definition']))

def make_acronym_regex(mapping: dict) -> re.Pattern:
    pat = r'\b(?:' + '|'.join(re.escape(a) for a in mapping) + r')\b'
    return re.compile(pat)

def make_theme_title(weights, feature_names):
    comp_pat = re.compile(r"(^[A-Z]{3,}$)|[\d/#\\-]")
    idx = weights.argsort()[-15:][::-1]
    toks = [feature_names[i] for i in idx]
    important = [t for t in toks if comp_pat.search(t)] + [t for t in toks if not comp_pat.search(t)]
    return important[0].title() if important else "Miscellaneous"


# ─── SAFEAeroBert ──────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel
from sentence_splitter import SentenceSplitter
from sklearn.cluster import KMeans
import numpy as np
import torch


# ## 3. Load SafeAeroBERT model and tokenizer as an embedding extractor
model_name = "NASA-AIML/MIKA_SafeAeroBERT" #Bertfast Tokenizer 

tokenizer = AutoTokenizer.from_pretrained(model_name)
# We load the base model (encoder-only) to get sentence embeddings
model = AutoModel.from_pretrained(model_name)
model.eval()

# ## 4. Define helper functions

def embed_sentence(sentence, max_length=128):
    """Returns the CLS token embedding for a sentence."""
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS embedding
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def extractive_summarize(text, num_sentences=3):
    """Performs extractive summarization by clustering sentence embeddings and
    selecting the nearest sentence to each cluster center."""
    splitter = SentenceSplitter(language='en')
    sentences = splitter.split(text)
    if len(sentences) <= num_sentences:
        return sentences

    # Compute embeddings for each sentence
    embeddings = np.array([embed_sentence(s) for s in sentences])

    # Cluster embeddings
    kmeans = KMeans(n_clusters=num_sentences, random_state=42).fit(embeddings)
    centers = kmeans.cluster_centers_

    # Find the closest sentence to each cluster center
    summary = []
    for center in centers:
        idx = np.argmin(np.linalg.norm(embeddings - center, axis=1))
        summary.append(sentences[idx])
    return summary
