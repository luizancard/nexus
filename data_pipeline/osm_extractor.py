# pyrefly: ignore [missing-import]
import osmnx as ox
import networkx as nx
import rasterio
import numpy as np
import math

def extrair_malha_pedestres(lugar: str) -> nx.MultiDiGraph:
    """
    Baixa o grafo de ruas e calçadas do OpenStreetMap.
    """
    print(f"[1/4] Extraindo grafo de pedestres para: {lugar}...")
    
    # network_type='walk' garante que ignoramos vias expressas proibidas para pedestres
    # simplify=True corrige a topologia, removendo nós inúteis no meio de retas
    G = ox.graph_from_place(lugar, network_type='walk', simplify=True)
    
    print(f"      Grafo extraído: {len(G.nodes)} nós e {len(G.edges)} arestas.")
    return G


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
            elif valor_z < 0 or valor_z > 3000: # Heurística de segurança para anomalias
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
    CAMINHO_TIF = "/Users/luizancard/Developer/lourdes_dem_1m.tif"
    BAIRRO = "Lourdes, Belo Horizonte, Minas Gerais, Brazil"
    
    grafo_bruto = extrair_malha_pedestres(BAIRRO)
    grafo_3d = injetar_topografia_e_calcular_esforco(grafo_bruto, CAMINHO_TIF)
    
    # Validação rápida: imprime os dados de 1 aresta para provar que funcionou
    aresta_exemplo = list(grafo_3d.edges(data=True))[0]
    print("\nExemplo de Aresta Processada:")
    print(f"Comprimento: {aresta_exemplo[2].get('length', 'N/A')} metros")
    print(f"Inclinação (Grade): {aresta_exemplo[2].get('grade_abs', 'N/A') * 100:.2f}%")