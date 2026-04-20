import argparse
import csv
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def build_grafo(csv_path, weight_col, undirected=True):
    """Build graph as node -> [[neighbor, weight], ...]."""
    adj = defaultdict(dict)
    nodes = set()

    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            origem = row["source"].strip()
            destino = row["target"].strip()
            peso = float(row[weight_col])

            nodes.add(origem)
            nodes.add(destino)
            adj[origem][destino] = peso
            if undirected:
                adj[destino][origem] = peso

    return {
        node: [[neighbor, adj[node][neighbor]] for neighbor in sorted(adj[node].keys())]
        for node in sorted(nodes)
    }


def build_graph(grafo_data):
    graph = nx.Graph()
    for origem, vizinhos in grafo_data.items():
        for destino, peso in vizinhos:
            graph.add_edge(origem, destino, weight=float(peso))
    return graph


def build_layout_graph(graph):
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(graph.nodes())
    layout_graph.add_edges_from(graph.edges())
    return layout_graph


def _parse_grau(texto):
    valor = texto.strip().replace("−", "-")
    for simbolo in ("∘", "°", "º"):
        valor = valor.replace(simbolo, "")
    return float(valor)


def carregar_posicoes_cidades(path="cidades.csv"):
    posicoes = {}
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            cidade = row["Cidade"].strip()
            latitude = _parse_grau(row["Latitude"])
            longitude = _parse_grau(row["Longitude"])
            posicoes[cidade] = (longitude, latitude)
    return posicoes


def _xy(pos, node):
    point = pos[node]
    return float(point[0]), float(point[1])


def _segments_proper_cross(ax, ay, bx, by, cx, cy, dx, dy):
    """True se os segmentos AB e CD se intersectam no interior (não só no vértice)."""

    def orient(px, py, qx, qy, rx, ry):
        return (qy - py) * (rx - qx) - (qx - px) * (ry - qy)

    def on_seg(px, py, qx, qy, rx, ry):
        return (
            min(px, rx) - 1e-12 <= qx <= max(px, rx) + 1e-12
            and min(py, ry) - 1e-12 <= qy <= max(py, ry) + 1e-12
        )

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (
        o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0
    ):
        return True
    if o1 == 0 and on_seg(ax, ay, cx, cy, bx, by):
        return True
    if o2 == 0 and on_seg(ax, ay, dx, dy, bx, by):
        return True
    if o3 == 0 and on_seg(cx, cy, ax, ay, dx, dy):
        return True
    if o4 == 0 and on_seg(cx, cy, bx, by, dx, dy):
        return True
    return False


def count_edge_crossings(graph, pos):
    """Número de pares de arestas cujas linhas se cruzam (arestas que partilham nó não contam)."""
    edges = list(graph.edges())
    total_edges = len(edges)
    crossings = 0
    for i in range(total_edges):
        u1, v1 = edges[i]
        x1, y1 = _xy(pos, u1)
        x2, y2 = _xy(pos, v1)
        for j in range(i + 1, total_edges):
            u2, v2 = edges[j]
            if len({u1, v1, u2, v2}) < 4:
                continue
            x3, y3 = _xy(pos, u2)
            x4, y4 = _xy(pos, v2)
            if _segments_proper_cross(x1, y1, x2, y2, x3, y3, x4, y4):
                crossings += 1
    return crossings


def _pos_copy(pos):
    return {node: (float(point[0]), float(point[1])) for node, point in pos.items()}


def _total_edge_length(graph, pos):
    total_length = 0.0
    for origem, destino in graph.edges():
        x1, y1 = _xy(pos, origem)
        x2, y2 = _xy(pos, destino)
        total_length += math.hypot(x2 - x1, y2 - y1)
    return total_length


def layout_min_crossings(graph, base_seed=42, spring_samples=72):
    """
    Gera vários layouts e escolhe o com menos cruzamentos de arestas.
    Empate: menor comprimento total das arestas (desenho mais compacto).
    """
    best_pos = None
    best_cross = None
    best_len = None

    def consider(pos):
        nonlocal best_pos, best_cross, best_len
        if pos is None:
            return
        crossings = count_edge_crossings(graph, pos)
        total_length = _total_edge_length(graph, pos)
        if best_cross is None or crossings < best_cross or (
            crossings == best_cross and best_len is not None and total_length < best_len
        ):
            best_cross = crossings
            best_len = total_length
            best_pos = _pos_copy(pos)

    try:
        consider(nx.kamada_kawai_layout(graph, weight="weight", scale=1.0))
    except Exception:
        pass

    try:
        consider(nx.kamada_kawai_layout(graph, scale=1.0))
    except Exception:
        pass

    try:
        consider(nx.spectral_layout(graph, weight="weight", scale=1.0))
    except Exception:
        pass

    ks = (0.85, 1.0, 1.15, 1.35, 1.55, 1.85, 2.1)
    for i in range(spring_samples):
        seed = base_seed + i * 9973
        for k_value in ks:
            consider(
                nx.spring_layout(
                    graph,
                    weight="weight",
                    seed=seed,
                    k=k_value,
                    iterations=220,
                )
            )

    try:
        for prog in ("sfdp", "neato", "fdp"):
            try:
                consider(nx.nx_agraph.graphviz_layout(graph, prog=prog))
            except Exception:
                continue
    except Exception:
        pass

    if best_pos is None:
        best_pos = _pos_copy(
            nx.spring_layout(graph, weight="weight", seed=base_seed, k=1.35, iterations=200)
        )
        best_cross = count_edge_crossings(graph, best_pos)

    return best_pos, best_cross


def main():
    parser = argparse.ArgumentParser(
        description="Gera imagem do grafo de tempo com labels de nós e pesos."
    )
    parser.add_argument(
        "--csv",
        default="lista_edges.csv",
        help="Caminho do CSV de arestas (default: lista_edges.csv)",
    )
    parser.add_argument(
        "--weight-col",
        default="time(min)",
        help='Nome da coluna de peso (default: "time(min)")',
    )
    parser.add_argument(
        "--out",
        default="grafo_tempo.png",
        help="Arquivo de saída PNG (default: grafo_tempo.png)",
    )
    parser.add_argument(
        "--layout",
        choices=["geo", "min_cross", "spring", "kamada"],
        default="min_cross",
        help="Layout: min_cross minimiza cruzamentos num layout comum; geo usa as coordenadas reais das cidades.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Não imprimir número de cruzamentos (modo min_cross).",
    )
    parser.add_argument(
        "--layout-samples",
        type=int,
        default=72,
        metavar="N",
        help=(
            "Só com --layout min_cross: quantas seeds do spring_layout testar "
            "(valores maiores podem reduzir mais cruzamentos, com mais tempo)."
        ),
    )
    args = parser.parse_args()

    grafo_tempo = build_grafo(args.csv, args.weight_col, undirected=True)
    graph = build_graph(grafo_tempo)
    layout_graph = build_layout_graph(graph)

    if args.layout == "geo":
        pos = carregar_posicoes_cidades()
    elif args.layout == "spring":
        pos = nx.spring_layout(graph, weight="weight", seed=42, k=1.35, iterations=200)
    elif args.layout == "kamada":
        try:
            pos = nx.kamada_kawai_layout(graph, weight="weight", scale=1.0)
        except Exception:
            pos = nx.spring_layout(graph, weight="weight", seed=42, k=1.35, iterations=200)
    else:
        pos, n_cross = layout_min_crossings(
            layout_graph, spring_samples=max(8, args.layout_samples)
        )
        if not args.quiet:
            print(f"Cruzamentos de arestas (layout escolhido): {n_cross}")

    plt.figure(figsize=(20, 16))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=4200,
        node_color="#9ecae1",
        linewidths=1.2,
        edgecolors="#3182bd",
    )
    nx.draw_networkx_edges(graph, pos, width=2.0, alpha=0.78, edge_color="#555555")
    node_label_bbox = {
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "#b0c4de",
        "linewidth": 0.6,
        "alpha": 0.94,
    }
    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=11,
        font_weight="bold",
        bbox=node_label_bbox,
    )

    edge_labels = nx.get_edge_attributes(graph, "weight")
    edge_labels = {edge: f"{weight:.1f}" for edge, weight in edge_labels.items()}
    edge_label_bbox = {
        "boxstyle": "round,pad=0.22",
        "facecolor": "white",
        "edgecolor": "#cccccc",
        "linewidth": 0.35,
        "alpha": 0.92,
    }
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8.5,
        bbox=edge_label_bbox,
    )

    plt.title("Grafo de Tempo")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Imagem gerada: {args.out}")


if __name__ == "__main__":
    main()