import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

INPUT_FILE = "data/dataset_train_completo.csv"
OUTPUT_FILE = "data/dataset_pca.csv"
N_COMPONENTS = None # Mantener todo los componentes

def main():
    # 1. Cargar datos
    if not os.path.exists(INPUT_FILE):
        return
        
    df = pd.read_csv(INPUT_FILE, index_col="Time", parse_dates=True)

    # 2. Guardar OHLCV originales (necesarios para trading_env_v2)
    ohlcv_orig = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    
    # Eliminamos filas con infinitos o nulos que hayan quedado
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 3. Aplicar PCA SOLO a los indicadores (sin OHLCV)
    indicadores = df_clean.drop(
        columns=["Open", "High", "Low", "Close", "Volume"], 
        errors='ignore'
    ).copy()
    
    # Normalización
    scaler = StandardScaler()
    indicadores_scaled = scaler.fit_transform(indicadores)
    
    # Aplicar PCA
    pca = PCA(n_components=N_COMPONENTS)
    principal_components = pca.fit_transform(indicadores_scaled)
    
    # OHLCV + PCA
    column_names = [f'PC{i+1}' for i in range(principal_components.shape[1])]
    df_pca = pd.DataFrame(data=principal_components, columns=column_names, index=df_clean.index)
    ohlcv_aligned = ohlcv_orig.loc[df_clean.index].copy()
    
    df_final = pd.concat([ohlcv_aligned, df_pca], axis=1)
    df_final.to_csv(OUTPUT_FILE)   

if __name__ == "__main__":
    main()