import argparse
import math

import matplotlib.pyplot as plt
import networkx as nx

from grafo_distancia import grafo


def build_graph(grafo_data):
    g = nx.Graph()
    for origem, vizinhos in grafo_data.items():
        for destino, peso in vizinhos:
            g.add_edge(origem, destino, weight=float(peso))
    return g


def _xy(pos, node):
    p = pos[node]
    return float(p[0]), float(p[1])


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


def count_edge_crossings(g, pos):
    """Número de pares de arestas cujas linhas se cruzam (arestas que partilham nó não contam)."""
    edges = list(g.edges())
    n = len(edges)
    c = 0
    for i in range(n):
        u1, v1 = edges[i]
        x1, y1 = _xy(pos, u1)
        x2, y2 = _xy(pos, v1)
        for j in range(i + 1, n):
            u2, v2 = edges[j]
            if len({u1, v1, u2, v2}) < 4:
                continue
            x3, y3 = _xy(pos, u2)
            x4, y4 = _xy(pos, v2)
            if _segments_proper_cross(x1, y1, x2, y2, x3, y3, x4, y4):
                c += 1
    return c


def _pos_copy(pos):
    return {n: (float(p[0]), float(p[1])) for n, p in pos.items()}


def _total_edge_length(g, pos):
    s = 0.0
    for u, v in g.edges():
        x1, y1 = _xy(pos, u)
        x2, y2 = _xy(pos, v)
        s += math.hypot(x2 - x1, y2 - y1)
    return s


def layout_min_crossings(g, base_seed=42, spring_samples=72):
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
        cr = count_edge_crossings(g, pos)
        ln = _total_edge_length(g, pos)
        if best_cross is None or cr < best_cross or (
            cr == best_cross and best_len is not None and ln < best_len
        ):
            best_cross = cr
            best_len = ln
            best_pos = _pos_copy(pos)

    try:
        consider(
            nx.kamada_kawai_layout(g, weight="weight", scale=1.0)
        )
    except Exception:
        pass

    try:
        consider(nx.kamada_kawai_layout(g, scale=1.0))
    except Exception:
        pass

    try:
        consider(nx.spectral_layout(g, weight="weight", scale=1.0))
    except Exception:
        pass

    ks = (0.85, 1.0, 1.15, 1.35, 1.55, 1.85, 2.1)
    for i in range(spring_samples):
        seed = base_seed + i * 9973
        for k in ks:
            consider(
                nx.spring_layout(
                    g,
                    weight="weight",
                    seed=seed,
                    k=k,
                    iterations=220,
                )
            )

    try:
        for prog in ("sfdp", "neato", "fdp"):
            try:
                consider(nx.nx_agraph.graphviz_layout(g, prog=prog))
            except Exception:
                continue
    except Exception:
        pass

    if best_pos is None:
        best_pos = _pos_copy(
            nx.spring_layout(g, weight="weight", seed=base_seed, k=1.35, iterations=200)
        )
        best_cross = count_edge_crossings(g, best_pos)

    return best_pos, best_cross


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
        choices=["min_cross", "spring", "kamada"],
        default="min_cross",
        help="Layout: min_cross tenta vários algoritmos e minimiza cruzamentos (default).",
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

    g = build_graph(grafo)

    if args.layout == "spring":
        pos = nx.spring_layout(
            g, weight="weight", seed=42, k=1.35, iterations=200
        )
    elif args.layout == "kamada":
        try:
            pos = nx.kamada_kawai_layout(g, weight="weight", scale=1.0)
        except Exception:
            pos = nx.spring_layout(
                g, weight="weight", seed=42, k=1.35, iterations=200
            )
    else:
        pos, n_cross = layout_min_crossings(g, spring_samples=max(8, args.layout_samples))
        if not args.quiet:
            print(f"Cruzamentos de arestas (layout escolhido): {n_cross}")

    plt.figure(figsize=(20, 16))
    # node_size em pontos²: valores ~3500+ acomodam melhor o texto das cidades.
    nx.draw_networkx_nodes(
        g,
        pos,
        node_size=4200,
        node_color="#9ecae1",
        linewidths=1.2,
        edgecolors="#3182bd",
    )
    nx.draw_networkx_edges(g, pos, width=2.0, alpha=0.78, edge_color="#555555")
    node_label_bbox = {
        "boxstyle": "round,pad=0.35",
        "facecolor": "white",
        "edgecolor": "#b0c4de",
        "linewidth": 0.6,
        "alpha": 0.94,
    }
    nx.draw_networkx_labels(
        g,
        pos,
        font_size=11,
        font_weight="bold",
        bbox=node_label_bbox,
    )

    edge_labels = nx.get_edge_attributes(g, "weight")
    edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
    edge_label_bbox = {
        "boxstyle": "round,pad=0.22",
        "facecolor": "white",
        "edgecolor": "#cccccc",
        "linewidth": 0.35,
        "alpha": 0.92,
    }
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        font_size=8.5,
        bbox=edge_label_bbox,
    )

    plt.title("Grafo de Distancias")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Imagem gerada: {args.out}")


if __name__ == "__main__":
    main()
