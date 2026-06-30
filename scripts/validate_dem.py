import rasterio
import numpy as np
import sys

CAMINHO_DEM = "/Users/luizancard/Developer/nexus_project/data_files/lourdes_dem_1m.tif"

def validar_dem(caminho):
    print(f"Auditando arquivo: {caminho}\n")
    
    try:
        with rasterio.open(caminho) as src:
            # 1. Checagem de Metadados
            print(f"CRS: {src.crs}")
            print(f"Resolução: {src.res}")
            print(f"Dimensões (linhas x colunas): {src.height} x {src.width}")
            print(f"Dtype do raster: {src.dtypes[0]}")
            print(f"Valor NoData declarado: {src.nodata}")
            
            # 2. Ler os dados (np.reshape evita DeprecationWarning do NumPy 2.5)
            dados_raw = src.read(1)
            dados = np.reshape(dados_raw.copy(), dados_raw.shape)
            nodata = src.nodata
            
            # 3. Filtrar dados válidos (ignorando o NoData)
            if nodata is not None:
                # Tolerância para float: comparação aproximada
                if np.issubdtype(dados.dtype, np.floating):
                    mask = ~np.isclose(dados, nodata, rtol=0, atol=1e-3)
                else:
                    mask = (dados != nodata)
            else:
                mask = ~np.isnan(dados)
                
            dados_validos = dados[mask]
            
            # 4. Diagnóstico de pixels
            total_pixels = dados.size
            pixels_nodata = total_pixels - dados_validos.size
            pct_nodata = (pixels_nodata / total_pixels) * 100
            print(f"\nTotal de pixels: {total_pixels:,}")
            print(f"Pixels NoData: {pixels_nodata:,} ({pct_nodata:.1f}%)")
            print(f"Pixels válidos: {dados_validos.size:,} ({100-pct_nodata:.1f}%)")
            
            # Amostra dos valores brutos para diagnóstico
            sample = dados.flatten()[:10]
            print(f"Amostra de valores brutos (primeiros 10): {sample}")
            
            # 5. Estatísticas
            if dados_validos.size == 0:
                print("\nERRO: 100% dos pixels são NoData.")
                print("Possíveis causas:")
                print("  - A interpolação/exportação do DEM falhou.")
                print("  - O valor NoData declarado não corresponde ao sentinel real nos dados.")
                print(f"  - Verifique se o valor '{nodata}' é realmente o fill value usado.")
            else:
                print(f"\nAltitude Mínima: {np.min(dados_validos):.2f}m")
                print(f"Altitude Média: {np.mean(dados_validos):.2f}m")
                print(f"Altitude Máxima: {np.max(dados_validos):.2f}m")
                
                if 700 < np.mean(dados_validos) < 1100:
                    print("\nSUCESSO: O DEM parece estar correto e dentro do intervalo de altitude de BH.")
                else:
                    print("\nALERTA: A altitude média está fora do esperado. Revise os dados.")
                    
    except Exception as e:
        print(f"Erro ao abrir arquivo: {e}")

if __name__ == "__main__":
    validar_dem(CAMINHO_DEM)