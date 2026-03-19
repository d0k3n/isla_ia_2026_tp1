import argparse
import csv
from collections import defaultdict
from pprint import pformat


def build_grafo(csv_path, weight_col, undirected=True):
    """Build graph as node -> [[neighbor, weight], ...]."""
    adj = defaultdict(dict)
    nodes = set()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row["source"].strip()
            v = row["target"].strip()
            w = float(row[weight_col])

            nodes.add(u)
            nodes.add(v)
            adj[u][v] = w
            if undirected:
                adj[v][u] = w

    return {
        n: [[m, adj[n][m]] for m in sorted(adj[n].keys())]
        for n in sorted(nodes)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Converte CSV de arestas para variável Python legível: grafo = {...}"
    )
    parser.add_argument(
        "--csv",
        default="ia_tp1 - edges.csv",
        help="Caminho do CSV (default: ia_tp1 - edges.csv)",
    )
    parser.add_argument(
        "--weight-col",
        default="distance(km)",
        help="Nome da coluna de peso (default: distance(km))",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Se informado, não duplica arestas (grafo direcionado).",
    )
    parser.add_argument(
        "--out",
        default="grafo_distancia.py",
        help="Arquivo de saída .py (default: grafo_distancia.py)",
    )
    args = parser.parse_args()

    grafo = build_grafo(
        csv_path=args.csv,
        weight_col=args.weight_col,
        undirected=not args.directed,
    )

    rendered = "grafo = " + pformat(grafo, width=100, sort_dicts=True) + "\n"

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(rendered)

    print(f"Arquivo gerado: {args.out}")
    print("Primeiras linhas:")
    print(rendered[:500] + ("..." if len(rendered) > 500 else ""))


if __name__ == "__main__":
    main()
