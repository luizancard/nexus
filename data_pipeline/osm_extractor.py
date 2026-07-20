# pyrefly: ignore [missing-import]
import osmnx as ox
import networkx as nx
import rasterio
import numpy as np
import math
from pathlib import Path

# Tags de acessibilidade que a OSMnx descarta por padrão (default é um conjunto
# minimo focado em roteamento veicular). Sem isso, surface/smoothness/kerb/etc.
# nunca chegam aos atributos da aresta, mesmo quando existem no OSM bruto.
# 'width' e 'highway' ja vem no default da OSMnx -- mantidos aqui so por clareza.
TAGS_ACESSIBILIDADE_WAY = [
    "surface", "smoothness", "width", "kerb", "tactile_paving", "incline",
    "lit", "handrail", "ramp", "step_count", "wheelchair", "barrier",
    "footway", "highway",
]
TAGS_ACESSIBILIDADE_NODE = [
    "crossing", "tactile_paving", "kerb", "barrier", "wheelchair", "highway",
]

# Sanity bound for raw DEM elevation samples -- same defensive pattern
# geometric_attribute_extractor.MAX_PLAUSIBLE_SLOPE_PCT cites this line as
# following. Anything outside this range gets treated as NoData rather than
# a real elevation. Note this bound would NOT have caught the all-zero-DEM
# bug found and fixed this session (0.0 is inside the range) -- it guards
# against a different failure mode (wild out-of-range values), not silent
# uniform-zero data; that class of bug is now caught by validate_dem.py's
# mean-altitude sanity check instead.
ELEVATION_MIN_PLAUSIBLE_M = 0
ELEVATION_MAX_PLAUSIBLE_M = 3000


def highway_values(raw: str | list[str] | None) -> list[str]:
    """Normalize an edge's `highway` tag into a list of values.

    OSMnx's `simplify=True` (used in `extrair_malha_pedestres`) merges
    consecutive original OSM way segments into one edge; when the segments
    disagree on a tag, OSMnx stores it as a *list* instead of a scalar
    string. Comparing a list to a string with `==`/`in` silently and always
    fails. This is not theoretical: 8 real edges in the Lourdes graph carry
    `highway=['steps', 'footway']`-style lists, and a naive
    `data.get('highway') == 'steps'` check misses every one of them --
    exactly the failure mode that made this project's own methodology doc
    wrongly claim "OSM has no signal on stairs here" when OSM actually did.
    """
    if raw is None:
        return []
    return raw if isinstance(raw, list) else [raw]


def extrair_malha_pedestres(lugar: str) -> nx.MultiDiGraph:
    """
    Baixa o grafo de ruas e calçadas do OpenStreetMap, retendo as tags de
    acessibilidade (surface, smoothness, kerb, tactile_paving, etc.) que a
    OSMnx descartaria com sua configuracao padrao.
    """
    print(f"[1/4] Extraindo grafo de pedestres para: {lugar}...")

    # ox.settings e global e mutavel -- configurar aqui, dentro da funcao,
    # garante que toda chamada a graph_from_place nesta funcao usa o conjunto
    # certo de tags, independente da ordem de import/chamada de quem usa este modulo.
    ox.settings.useful_tags_way = list(
        set(ox.settings.useful_tags_way) | set(TAGS_ACESSIBILIDADE_WAY)
    )
    ox.settings.useful_tags_node = list(
        set(ox.settings.useful_tags_node) | set(TAGS_ACESSIBILIDADE_NODE)
    )

    # network_type='walk' garante que ignoramos vias expressas proibidas para pedestres
    # simplify=True corrige a topologia, removendo nós inúteis no meio de retas
    G = ox.graph_from_place(lugar, network_type='walk', simplify=True)

    print(f"      Grafo extraído: {len(G.nodes)} nós e {len(G.edges)} arestas.")
    return G


def diagnosticar_cobertura_tags(G: nx.MultiDiGraph) -> dict[str, float]:
    """
    Mede, para cada tag de acessibilidade, a porcentagem de arestas que
    realmente tem essa tag presente no OSM. Nao assume esparsidade -- mede.

    Args:
        G: Grafo com tags de acessibilidade extraidas (ver TAGS_ACESSIBILIDADE_WAY).

    Returns:
        Dicionario tag -> porcentagem (0-100) de arestas com essa tag presente.
    """
    total_arestas = G.number_of_edges()
    if total_arestas == 0:
        return {}

    cobertura: dict[str, float] = {}
    for tag in TAGS_ACESSIBILIDADE_WAY:
        presentes = sum(1 for _, _, data in G.edges(data=True) if data.get(tag) is not None)
        cobertura[tag] = round(100.0 * presentes / total_arestas, 2)

    tem_steps = sum(
        1 for _, _, data in G.edges(data=True) if "steps" in highway_values(data.get("highway"))
    )
    cobertura["highway=steps (contagem)"] = tem_steps

    print("[diagnostico] Cobertura de tags de acessibilidade no grafo:")
    for tag, pct in cobertura.items():
        print(f"      {tag}: {pct}")
    return cobertura


def salvar_grafo(G: nx.MultiDiGraph, caminho: Path) -> None:
    """Persiste o grafo em GraphML. Hoje o pipeline nunca salva nada em disco
    -- so existe dentro do bloco __main__ de teste -- o que forca reprocessar
    tudo a cada execucao."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    ox.io.save_graphml(G, filepath=caminho)
    print(f"      Grafo salvo em {caminho}")


def carregar_grafo(caminho: Path) -> nx.MultiDiGraph:
    """Carrega um grafo previamente salvo por `salvar_grafo`."""
    return ox.io.load_graphml(filepath=caminho)


def injetar_topografia_e_calcular_esforco(G: nx.MultiDiGraph, caminho_dem: str) -> nx.MultiDiGraph:
    """
    Cruza o grafo do OSM com o Modelo Digital de Elevação (DEM) e calcula a inclinação.
    """
    print(f"[2/4] Abrindo arquivo DEM para injeção topográfica...")
    
    with rasterio.open(caminho_dem) as src:
        crs_raster = src.crs
        nodata = src.nodata
        
        print(f"[3/4] Projetando o grafo para o CRS do Raster ({crs_raster})...")
        # PULO DO GATO: O OSMnx baixa em Lat/Lon (EPSG:4326). O Raster está em Metros (UTM).
        # Precisamos converter o grafo para o exato mesmo sistema do raster antes de cruzar os dados.
        G_proj = ox.project_graph(G, to_crs=crs_raster)
        
        print("[4/4] Amostrando elevação pixel a pixel para cada nó...")
        # Cria uma lista de coordenadas (X, Y) de todos os cruzamentos (nós) do bairro
        coordenadas = [(data['x'], data['y']) for node, data in G_proj.nodes(data=True)]
        
        # A função src.sample() extrai o valor do pixel exatamente nas coordenadas fornecidas
        amostras = list(src.sample(coordenadas))
        
        # Injetamos a elevação de volta no grafo
        nos_fora_do_raster = 0
        for (node, data), elev in zip(G_proj.nodes(data=True), amostras):
            valor_z = float(elev[0])
            
            # Tratamento de segurança: se o ponto cair fora do mapa ou for NoData
            if nodata is not None and math.isclose(valor_z, nodata, rel_tol=1e-5):
                data['elevation'] = np.nan
                nos_fora_do_raster += 1
            elif valor_z < ELEVATION_MIN_PLAUSIBLE_M or valor_z > ELEVATION_MAX_PLAUSIBLE_M: # Heurística de segurança para anomalias
                data['elevation'] = np.nan
                nos_fora_do_raster += 1
            else:
                data['elevation'] = valor_z

        if nos_fora_do_raster > 0:
            print(f"      [AVISO] {nos_fora_do_raster} nós caíram fora da área do Raster (NoData).")

    # Calcula a inclinação (grade) da rua usando a diferença de elevação entre os nós
    print("      Calculando inclinação matemática (Grade) de todas as arestas...")
    G_proj = ox.elevation.add_edge_grades(G_proj, add_absolute=True)
    
    print("\n PIPELINE CONCLUÍDO COM SUCESSO. Grafo pronto para o modelo de impedância. \n")    
    return G_proj

# BLOCO DE TESTE
if __name__ == "__main__":
    CAMINHO_TIF = str(Path(__file__).resolve().parent.parent / "data_files" / "lourdes_dem_1m.tif")
    CAMINHO_GRAFO = Path(__file__).resolve().parent.parent / "data_files" / "lourdes_graph_latest.graphml"
    BAIRRO = "Lourdes, Belo Horizonte, Minas Gerais, Brazil"

    grafo_bruto = extrair_malha_pedestres(BAIRRO)
    grafo_3d = injetar_topografia_e_calcular_esforco(grafo_bruto, CAMINHO_TIF)
    diagnosticar_cobertura_tags(grafo_3d)
    salvar_grafo(grafo_3d, CAMINHO_GRAFO)

    # Validação rápida: imprime os dados de 1 aresta para provar que funcionou
    aresta_exemplo = list(grafo_3d.edges(data=True))[0]
    print("\nExemplo de Aresta Processada:")
    print(f"Comprimento: {aresta_exemplo[2].get('length', 'N/A')} metros")
    print(f"Inclinação (Grade): {aresta_exemplo[2].get('grade_abs', 'N/A') * 100:.2f}%")