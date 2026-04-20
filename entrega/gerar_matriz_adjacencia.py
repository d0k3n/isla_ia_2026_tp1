"""Gera matriz de adjacencia (0/1) a partir de um CSV de arestas."""

import argparse
import csv


def carregar_arestas(csv_path):
    """Le source/target do CSV e retorna lista de pares (u, v)."""
    arestas = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = row["source"].strip()
            v = row["target"].strip()
            arestas.append((u, v))
    return arestas


def construir_matriz(arestas, directed=False):
    """Constroi cidades ordenadas e matriz de adjacencia 0/1."""
    cidades = sorted({n for u, v in arestas for n in (u, v)})
    idx = {cidade: i for i, cidade in enumerate(cidades)}

    n = len(cidades)
    matriz = [[0] * n for _ in range(n)]

    for u, v in arestas:
        i = idx[u]
        j = idx[v]
        matriz[i][j] = 1
        if not directed:
            matriz[j][i] = 1

    return cidades, matriz


def guardar_matriz(csv_out, cidades, matriz):
    """Escreve CSV no formato: cabecalho com cidades e linhas 0/1."""
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + cidades)
        for cidade, linha in zip(cidades, matriz):
            writer.writerow([cidade] + linha)


def main():
    parser = argparse.ArgumentParser(
        description="Gera matriz_adjacencia.csv a partir de lista_edges.csv"
    )
    parser.add_argument(
        "--in",
        dest="csv_in",
        default="lista_edges.csv",
        help="CSV de arestas com colunas source,target (default: lista_edges.csv)",
    )
    parser.add_argument(
        "--out",
        dest="csv_out",
        default="matriz_adjacencia.csv",
        help="CSV de saida (default: matriz_adjacencia.csv)",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Se informado, gera matriz direcionada. Por defeito, nao direcionada.",
    )
    args = parser.parse_args()

    arestas = carregar_arestas(args.csv_in)
    cidades, matriz = construir_matriz(arestas, directed=args.directed)
    guardar_matriz(args.csv_out, cidades, matriz)

    tipo = "direcionada" if args.directed else "nao direcionada"
    print(f"Matriz de adjacencia ({tipo}) gerada em: {args.csv_out}")
    print(f"Total de cidades: {len(cidades)}")
    print(f"Total de arestas lidas: {len(arestas)}")


if __name__ == "__main__":
    main()
