"""
Ejercicio 4 - Construccion de la red bipartita autor-video.

Nodos: autores (author_channel_id) y videos (video_id).
Arista: existe si un autor comento en un video; el peso es el numero de
comentarios que ese autor publico en ese video especifico. La red es NO
dirigida y no bipartita-ponderada en el sentido de "conversacion": una
arista solo indica co-presencia de comentarios, nunca una respuesta directa
entre personas (ver aclaracion del enunciado sobre reply_count).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from config import VIDEOS_CLEAN_CSV, COMMENTS_CLEAN_CSV, NETWORK_DIR, FIGURES_DIR

plt.rcParams["figure.dpi"] = 130


def build_bipartite(videos: pd.DataFrame, comments: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    video_stats = comments.groupby("video_id").agg(
        n_comentarios=("comment_id", "count"),
        n_autores_unicos=("author_channel_id", "nunique"),
    )
    videos_commented = videos[videos["video_id"].isin(comments["video_id"])].merge(
        video_stats, left_on="video_id", right_index=True
    )

    for _, row in videos_commented.iterrows():
        G.add_node(
            row["video_id"],
            bipartite="video",
            label=row["title"],
            channel_id=row["channel_id"],
            channel_name=row["channel_name"],
            category=row["category"],
            view_count=int(row["view_count"]),
            n_comentarios=int(row["n_comentarios"]),
            n_autores_unicos=int(row["n_autores_unicos"]),
        )

    author_stats = comments.groupby("author_channel_id").agg(
        author_name=("author_name", "first"),
        n_comentarios_totales=("comment_id", "count"),
        n_videos_distintos=("video_id", "nunique"),
        likes_recibidos=("like_count", "sum"),
    )
    for author_id, row in author_stats.iterrows():
        G.add_node(
            author_id,
            bipartite="author",
            label=row["author_name"],
            n_comentarios_totales=int(row["n_comentarios_totales"]),
            n_videos_distintos=int(row["n_videos_distintos"]),
            likes_recibidos=int(row["likes_recibidos"]),
        )

    edge_weights = comments.groupby(["author_channel_id", "video_id"]).size().reset_index(name="weight")
    for _, row in edge_weights.iterrows():
        G.add_edge(row["author_channel_id"], row["video_id"], weight=int(row["weight"]))

    return G


def export_tables(G: nx.Graph):
    node_rows = []
    for n, d in G.nodes(data=True):
        row = {"node_id": n}
        row.update(d)
        node_rows.append(row)
    nodes_df = pd.DataFrame(node_rows)
    nodes_df.to_csv(NETWORK_DIR / "bipartite_nodes.csv", index=False)

    edge_rows = [{"source": u, "target": v, "weight": d["weight"]} for u, v, d in G.edges(data=True)]
    edges_df = pd.DataFrame(edge_rows)
    edges_df.to_csv(NETWORK_DIR / "bipartite_edges.csv", index=False)

    return nodes_df, edges_df


def plot_bipartite(G: nx.Graph):
    videos = [n for n, d in G.nodes(data=True) if d["bipartite"] == "video"]
    authors = [n for n, d in G.nodes(data=True) if d["bipartite"] == "author"]

    pos = nx.spring_layout(G, k=0.35, iterations=150, seed=42, weight="weight")

    fig, ax = plt.subplots(figsize=(13, 11))
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.6, ax=ax)

    author_sizes = [15 + 8 * G.nodes[n]["n_videos_distintos"] for n in authors]
    nx.draw_networkx_nodes(G, pos, nodelist=authors, node_color="#7fb3d5", node_size=author_sizes, label="Autores", ax=ax)

    video_sizes = [80 + 4 * G.nodes[n]["n_comentarios"] for n in videos]
    nx.draw_networkx_nodes(G, pos, nodelist=videos, node_color="#e67e22", node_size=video_sizes, label="Videos", ax=ax)

    video_labels = {n: G.nodes[n]["label"][:28] for n in videos}
    nx.draw_networkx_labels(G, pos, labels=video_labels, font_size=6, ax=ax)

    ax.set_title("Red bipartita autor-video (351 nodos, 343 aristas)")
    ax.legend(scatterpoints=1)
    ax.axis("off")
    fig.savefig(FIGURES_DIR / "04_red_bipartita.png", bbox_inches="tight")
    plt.close(fig)
    print("figura guardada: 04_red_bipartita.png")


def main():
    videos = pd.read_csv(VIDEOS_CLEAN_CSV)
    comments = pd.read_csv(COMMENTS_CLEAN_CSV)

    G = build_bipartite(videos, comments)
    nodes_df, edges_df = export_tables(G)

    n_videos = (nodes_df["bipartite"] == "video").sum()
    n_authors = (nodes_df["bipartite"] == "author").sum()

    print(f"Nodos totales: {G.number_of_nodes()} (videos={n_videos}, autores={n_authors})")
    print(f"Aristas totales: {G.number_of_edges()}")
    print(f"Es bipartita (nx check): {nx.is_bipartite(G)}")
    print(f"Peso promedio de arista: {edges_df['weight'].mean():.3f}")
    print(f"Aristas con peso > 1 (mismo autor comento varias veces el mismo video): {(edges_df['weight'] > 1).sum()}")

    plot_bipartite(G)
    nx.write_graphml(G, NETWORK_DIR / "bipartite_author_video.graphml")
    print(f"\nTablas de nodos/aristas guardadas en {NETWORK_DIR}")


if __name__ == "__main__":
    main()
