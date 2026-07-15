import json
import os
from datetime import datetime
import numpy as np

class ResultadosManager:
    
    RESULTADOS_FILE = "resultados_entrenamientos.json"
    BALANCE_INICIAL = 10000.0
    
    @classmethod
    def guardar(cls, modelo_nombre, roi_pct, reward_acumulado, dataset, 
                balance_final=None, notas="", metricas_adicionales=None, archivo_destino=None):
        
        # Si no se pasa archivo_destino, usar el resultados por defecto
        if archivo_destino is None:
            archivo_destino = cls.RESULTADOS_FILE

        # Si no se pasa balance_final, calcularlo desde ROI
        if balance_final is None:
            balance_final = cls.BALANCE_INICIAL * (1 + roi_pct / 100)
        
        # Calcular ganancia absoluta
        ganancia_absoluta = balance_final - cls.BALANCE_INICIAL
        
        # Cargar resultados existentes
        if os.path.exists(archivo_destino):
            with open(archivo_destino, 'r') as f:
                resultados = json.load(f)
        else:
            resultados = {}
        
        # Actualizar/añadir entrada
        nueva_entrada = {
            # Métricas principales
            'roi_pct': float(roi_pct),
            'balance_inicial': float(cls.BALANCE_INICIAL),
            'balance_final': float(balance_final),
            'ganancia_absoluta': float(ganancia_absoluta),
            'ganancia_porcentual': float(roi_pct),  # Alias de roi_pct
            
            # Métricas adicionales
            'reward_acumulado': float(reward_acumulado),
            
            # Metadata
            'dataset': dataset,
            'notas': notas,
            'timestamp': datetime.now().isoformat(),
            'fecha_legible': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if(metricas_adicionales is not None):
            for key,value in metricas_adicionales.items():
                if key not in nueva_entrada:
                    if isinstance(value, (np.integer, int)):
                        nueva_entrada[key] = int(value)
                    elif isinstance(value, (np.floating, float)):
                        nueva_entrada[key] = float(value)
                    else:
                        nueva_entrada[key] = value
                
        resultados[modelo_nombre] = nueva_entrada
        # Guardar
        with open(archivo_destino, 'w') as f:
            json.dump(resultados, f, indent=2)
        