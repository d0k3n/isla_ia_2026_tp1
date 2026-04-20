"""
Comparar a performance dos algoritmos de procura num grafo (custo em tempo).
Ficheiro autónomo — inclui todas as dependências necessárias.
"""

import csv
import math
import time
import heapq
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Configuração de dados
# ---------------------------------------------------------------------------
CSV_EDGES_PATH = "lista_edges.csv"
WEIGHT_COL = "time(min)"
DIST_COL = "distance(km)"
TIME_COL = "time(min)"
R_TERRA_KM = 6371.0


# ---------------------------------------------------------------------------
# Construção do grafo por tempo (min)
# ---------------------------------------------------------------------------
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


def calcular_velocidade_media_kmh(csv_path, dist_col=DIST_COL, time_col=TIME_COL):
    """
    Calcula velocidade média global (km/h) a partir das arestas do dataset.

    Usa média ponderada por tempo: soma(distâncias) / soma(tempos em horas).
    """
    total_dist_km = 0.0
    total_horas = 0.0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = float(row[dist_col])
            t_min = float(row[time_col])
            if t_min <= 0:
                continue
            total_dist_km += d
            total_horas += t_min / 60.0

    if total_horas <= 0:
        return None
    return total_dist_km / total_horas


grafo_tempo = build_grafo(CSV_EDGES_PATH, WEIGHT_COL, undirected=True)
VELOCIDADE_MEDIA_KMH = calcular_velocidade_media_kmh(CSV_EDGES_PATH)


# ---------------------------------------------------------------------------
# Heurística temporal baseada em Haversine e velocidade média
# ---------------------------------------------------------------------------
def _parse_grau(s):
    t = s.strip().replace("−", "-")
    for ch in ("∘", "°", "º"):
        t = t.replace(ch, "")
    return float(t)


def carregar_coordenadas_cidades(path="cidades.csv"):
    coords = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cidade = row["Cidade"].strip()
            lat = _parse_grau(row["Latitude"])
            lon = _parse_grau(row["Longitude"])
            coords[cidade] = (lat, lon)
    return coords


COORDS = carregar_coordenadas_cidades()


def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlamb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlamb / 2) ** 2
    a = min(1.0, max(0.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R_TERRA_KM * c


def heuristica_tempo_estimada(no, destino):
    """
    Estima tempo mínimo em minutos pela distância em linha reta e velocidade média.

    Nota: esta heurística pode não ser estritamente admissível em todos os casos,
    porque depende da calibração da velocidade média global.
    """
    if no not in COORDS or destino not in COORDS:
        return 0.0

    if VELOCIDADE_MEDIA_KMH is None or VELOCIDADE_MEDIA_KMH <= 0:
        return 0.0

    lat1, lon1 = COORDS[no]
    lat2, lon2 = COORDS[destino]
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    return (dist_km / VELOCIDADE_MEDIA_KMH) * 60.0


_falta = sorted(n for n in grafo_tempo if n not in COORDS)
if _falta:
    print("Aviso: cidades no grafo sem coordenadas em cidades.csv:", ", ".join(_falta))


# ---------------------------------------------------------------------------
# Algoritmos de procura
# ---------------------------------------------------------------------------
def custo_percorrido_min(grafo, caminho):
    """Soma dos pesos das arestas ao longo da sequência de cidades no caminho."""
    if not caminho or len(caminho) < 2:
        return None
    total = 0.0
    for i in range(len(caminho) - 1):
        a, b = caminho[i], caminho[i + 1]
        peso = None
        for viz in grafo.get(a, []):
            if viz[0] == b:
                peso = viz[1]
                break
        if peso is None:
            return float("nan")
        total += peso
    return total


def medir_tempo_e_nos_expandidos(procura_funcao, *args):
    start_time = time.time()
    caminho, nos_expandidos = procura_funcao(*args)
    end_time = time.time()
    tempo_execucao = end_time - start_time
    return caminho, nos_expandidos, tempo_execucao, len(nos_expandidos)


def procura_profundidade(grafo, inicial, destino):
    visitados = set()
    caminho = []
    nos_expandidos = []

    def dfs(no):
        if no in visitados:
            return False
        visitados.add(no)
        caminho.append(no)
        nos_expandidos.append(no)
        if no == destino:
            return True
        for vizinho in grafo.get(no, []):
            if dfs(vizinho[0]):
                return True
        caminho.pop()
        return False

    dfs(inicial)
    return caminho, nos_expandidos


def procura_extensao(grafo, inicial, destino):
    fila = deque([inicial])
    visitados = set()
    pai = {inicial: None}
    nos_expandidos = []

    while fila:
        no_atual = fila.popleft()
        if no_atual in visitados:
            continue
        visitados.add(no_atual)
        nos_expandidos.append(no_atual)

        if no_atual == destino:
            seq = []
            n = destino
            while n is not None:
                seq.append(n)
                n = pai[n]
            seq.reverse()
            return seq, nos_expandidos

        for vizinho in grafo.get(no_atual, []):
            v = vizinho[0]
            if v not in pai:
                pai[v] = no_atual
                fila.append(v)

    return [], nos_expandidos


def procura_custo_uniforme(grafo, inicial, destino):
    fila = []
    heapq.heappush(fila, (0, inicial, [inicial]))
    visitados = set()
    nos_expandidos = []

    while fila:
        custo_atual, no_atual, caminho_atual = heapq.heappop(fila)
        if no_atual in visitados:
            continue
        visitados.add(no_atual)
        nos_expandidos.append(no_atual)

        if no_atual == destino:
            return caminho_atual, nos_expandidos

        for vizinho in grafo.get(no_atual, []):
            custo_vizinho = custo_atual + vizinho[1]
            heapq.heappush(fila, (custo_vizinho, vizinho[0], caminho_atual + [vizinho[0]]))

    return [], nos_expandidos


def procura_sofrega(grafo, inicial, destino, heuristica):
    fila = []
    heapq.heappush(fila, (heuristica(inicial, destino), inicial, [inicial]))
    visitados = set()
    nos_expandidos = []

    while fila:
        _, no_atual, caminho_atual = heapq.heappop(fila)
        if no_atual in visitados:
            continue
        visitados.add(no_atual)
        nos_expandidos.append(no_atual)

        if no_atual == destino:
            return caminho_atual, nos_expandidos

        for vizinho in grafo.get(no_atual, []):
            heapq.heappush(fila, (heuristica(vizinho[0], destino), vizinho[0], caminho_atual + [vizinho[0]]))

    return [], nos_expandidos


def procura_a_star(grafo, inicial, destino, heuristica):
    fila = []
    heapq.heappush(fila, (heuristica(inicial, destino), 0, inicial, [inicial]))
    visitados = set()
    nos_expandidos = []

    while fila:
        _, g_atual, no_atual, caminho_atual = heapq.heappop(fila)
        if no_atual in visitados:
            continue
        visitados.add(no_atual)
        nos_expandidos.append(no_atual)

        if no_atual == destino:
            return caminho_atual, nos_expandidos

        for vizinho in grafo.get(no_atual, []):
            g_vizinho = g_atual + vizinho[1]
            f_vizinho = g_vizinho + heuristica(vizinho[0], destino)
            heapq.heappush(fila, (f_vizinho, g_vizinho, vizinho[0], caminho_atual + [vizinho[0]]))

    return [], nos_expandidos


# ---------------------------------------------------------------------------
# Comparação interativa
# ---------------------------------------------------------------------------
def comparar(inicial, destino, grafo=None):
    if grafo is None:
        grafo = grafo_tempo

    algoritmos = [
        ("Resultado de Procura em extensão (BFS)", procura_extensao, False),
        ("Resultado de Procura em profundidade (DFS)", procura_profundidade, False),
        ("Resultado de Procura de custo uniforme (UCS)", procura_custo_uniforme, False),
        ("Resultado de Procura sôfrega / greedy (best-first com heurística)", procura_sofrega, True),
        ("Resultado de Procura A*", procura_a_star, True),
    ]

    for nome, algoritmo, com_heuristica in algoritmos:
        if com_heuristica:
            caminho, nos_expandidos, tempo_execucao, num_nos_expandidos = medir_tempo_e_nos_expandidos(
                algoritmo, grafo, inicial, destino, heuristica_tempo_estimada
            )
        else:
            caminho, nos_expandidos, tempo_execucao, num_nos_expandidos = medir_tempo_e_nos_expandidos(
                algoritmo, grafo, inicial, destino
            )

        custo_min = custo_percorrido_min(grafo, caminho)
        if custo_min is None:
            linha_custo = "Tempo percorrido (min): —"
        elif math.isnan(custo_min):
            linha_custo = "Tempo percorrido (min): — (caminho não coincide com arestas do grafo)"
        else:
            linha_custo = f"Tempo percorrido (min): {custo_min:.2f}"

        print(f"\n{nome}")
        print(f"Caminho: {caminho}")
        print(linha_custo)
        print(f"Nos Expandidos: {nos_expandidos}")
        print(f"Numero de Nos Expandidos: {num_nos_expandidos}")
        print(f"Tempo de Execução: {tempo_execucao:.6f} segundos")


def main():
    grafo = grafo_tempo
    cidades = sorted(grafo.keys())

    print("Comparar a performance dos algoritmos de procura num grafo (tempo em minutos)")
    print("Cidades disponíveis:", ", ".join(cidades))

    if VELOCIDADE_MEDIA_KMH is None:
        print("Velocidade média indisponível; heurística temporal vai usar fallback h=0.")
    else:
        print(f"Velocidade média estimada (km/h): {VELOCIDADE_MEDIA_KMH:.2f}")

    # Modo standalone sem Jupyter
    inicial = input(f"Cidade inicial [{cidades[0]}]: ").strip() or cidades[0]
    destino = input(f"Cidade destino [{cidades[-1]}]: ").strip() or cidades[-1]

    if inicial not in grafo:
        raise ValueError(f"Cidade inicial inválida: {inicial}")
    if destino not in grafo:
        raise ValueError(f"Cidade destino inválida: {destino}")

    comparar(inicial, destino, grafo)


if __name__ == "__main__":
    main()
