import argparse

import matplotlib.pyplot as plt
import networkx as nx

from grafo_distancia import grafo


def build_graph(grafo_data):
    g = nx.Graph()
    for origem, vizinhos in grafo_data.items():
        for destino, peso in vizinhos:
            g.add_edge(origem, destino, weight=float(peso))
    return g


def main():
    parser = argparse.ArgumentParser(
        description="Gera imagem do grafo com labels de nós e pesos."
    )
    parser.add_argument(
        "--out",
        default="grafo_distancia.png",
        help="Arquivo de saída PNG (default: grafo_distancia.png)",
    )
    parser.add_argument(
        "--layout",
        choices=["spring", "kamada"],
        default="spring",
        help="Algoritmo de layout (default: spring)",
    )
    args = parser.parse_args()

    g = build_graph(grafo)

    if args.layout == "spring":
        pos = nx.spring_layout(g, seed=42, k=0.85)
    else:
        try:
            pos = nx.kamada_kawai_layout(g)
        except ModuleNotFoundError:
            # Kamada-Kawai can require scipy; fallback keeps script dependency-light.
            pos = nx.spring_layout(g, seed=42, k=0.85)

    plt.figure(figsize=(18, 14))
    nx.draw_networkx_nodes(g, pos, node_size=950, node_color="#9ecae1")
    nx.draw_networkx_edges(g, pos, width=1.5, alpha=0.8)
    nx.draw_networkx_labels(g, pos, font_size=9, font_weight="bold")

    edge_labels = nx.get_edge_attributes(g, "weight")
    edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )

    plt.title("Grafo de Distancias")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Imagem gerada: {args.out}")


if __name__ == "__main__":
    main()
