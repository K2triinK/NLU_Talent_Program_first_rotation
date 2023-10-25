import numpy as np
from umap import UMAP
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import plotly.graph_objects as go


def map_to_topics(topics_df, file_path, lang):
    
    # Get projects to map
    df_read = pd.read_excel(file_path.name)
    texts = df_read.text.tolist()
    if 'title' in df_read.columns:
        titles = df_read.title.tolist()
    else:
        titles = None
        
    # Get topics
    topic_descs = []
    topic_ids = []
    for i, row in topics_df.iterrows():
        topic_descs.append(row['Topic title'] + ' ' + row['Example text or topic summary'])
    
    # Choose language model
    if lang == 'EN':
        model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        #model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')
    
    # Embed topics and corpus
    topic_embeddings = model.encode(topic_descs, convert_to_tensor=True)
    corpus_embeddings = model.encode(texts, convert_to_tensor=True)
    
    #Compute cosine similarities
    cosine_scores = util.cos_sim(corpus_embeddings, topic_embeddings)
    
    for i, emb in enumerate(corpus_embeddings):
        topic_ids.append(cosine_scores[i].argmax().item())
    
    topic_per_doc = topic_ids
    sample = 1
    if titles:
        docs = titles
    else:
        docs = texts

    # Get indices
    indices = []
    for topic in set(topic_per_doc):
        s = np.where(np.array(topic_per_doc) == topic)[0]
        size = len(s) if len(s) < 100 else int(len(s) * sample)
        indices.extend(np.random.choice(s, size=size, replace=False))
    indices = np.array(indices)

    df = pd.DataFrame({"topic": np.array(topic_per_doc)[indices]})
    df["doc"] = [docs[index] for index in indices]
    df["topic"] = [topic_per_doc[index] for index in indices]
    
    umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit(corpus_embeddings.cpu())
    embeddings_2d = umap_model.embedding_

    unique_topics = set(topic_per_doc)
    topics = unique_topics

    # Combine data
    df["x"] = embeddings_2d[:, 0]
    df["y"] = embeddings_2d[:, 1]

    names = [f"{row['Topic title']} ({topic_per_doc.count(i)})" for i, row in topics_df.iterrows()]
    
    # Visualize
    fig = go.Figure()

    # Selected topics
    for name, topic in zip(names, unique_topics):
        selection = df.loc[df.topic == topic, :]
        selection["text"] = ""

        selection.loc[len(selection), :] = [None, None, selection.x.mean(), selection.y.mean(), name]

        fig.add_trace(
            go.Scattergl(
                x=selection.x,
                y=selection.y,
                hovertext=selection.doc,
                hoverinfo="text",
                text=selection.text,
                mode='markers+text',
                name=name,
                textfont=dict(
                    size=12,
                ),
                marker=dict(size=5, opacity=0.5)
             )
        )
    
    width = 1200
    height = 750
    
    # Add grid in a 'plus' shape
    x_range = (df.x.min() - abs((df.x.min()) * .15), df.x.max() + abs((df.x.max()) * .15))
    y_range = (df.y.min() - abs((df.y.min()) * .15), df.y.max() + abs((df.y.max()) * .15))
    fig.add_shape(type="line",
                  x0=sum(x_range) / 2, y0=y_range[0], x1=sum(x_range) / 2, y1=y_range[1],
                  line=dict(color="#CFD8DC", width=2))
    fig.add_shape(type="line",
                  x0=x_range[0], y0=sum(y_range) / 2, x1=x_range[1], y1=sum(y_range) / 2,
                  line=dict(color="#9E9E9E", width=2))
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)

    # Stylize layout
    fig.update_layout(
        template="simple_white",
        title={
            'text': "<b>Documents and Topics",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig