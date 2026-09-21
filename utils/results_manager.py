import json
import os
from datetime import datetime
import numpy as np


class ResultadosManager:

    RESULTADOS_FILE = "resultados_entrenamientos.json"
    BALANCE_INICIAL = 10000.0

    # ------------------------------------------------------------------
    # Guardado de UN entrenamiento individual (una arquitectura, una semilla)
    # ------------------------------------------------------------------
    @classmethod
    def guardar(cls, modelo_nombre, roi_pct, reward_acumulado, dataset,
                balance_final=None, notas="", metricas_adicionales=None,
                archivo_destino=None, tiempo_entrenamiento_seg=None,
                semilla=None, arquitectura=None):

        if archivo_destino is None:
            archivo_destino = cls.RESULTADOS_FILE

        if balance_final is None:
            balance_final = cls.BALANCE_INICIAL * (1 + roi_pct / 100)

        ganancia_absoluta = balance_final - cls.BALANCE_INICIAL

        resultados = cls._cargar(archivo_destino)

        nueva_entrada = {
            'roi_pct': float(roi_pct),
            'balance_inicial': float(cls.BALANCE_INICIAL),
            'balance_final': float(balance_final),
            'ganancia_absoluta': float(ganancia_absoluta),
            'ganancia_porcentual': float(roi_pct),

            'reward_acumulado': float(reward_acumulado),

            'dataset': dataset,
            'notas': notas,
            'timestamp': datetime.now().isoformat(),
            'fecha_legible': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Variables de control del experimento (para que quede trazado en el
        # propio JSON qué se mantuvo fijo y qué se cambió entre runs)
        if arquitectura is not None:
            nueva_entrada['arquitectura'] = arquitectura
        if semilla is not None:
            nueva_entrada['semilla'] = int(semilla)
        if tiempo_entrenamiento_seg is not None:
            nueva_entrada['tiempo_entrenamiento_seg'] = float(tiempo_entrenamiento_seg)
            nueva_entrada['tiempo_entrenamiento_min'] = float(tiempo_entrenamiento_seg / 60)

        if metricas_adicionales is not None:
            for key, value in metricas_adicionales.items():
                if key not in nueva_entrada:
                    nueva_entrada[key] = cls._a_tipo_serializable(value)

        resultados[modelo_nombre] = nueva_entrada
        cls._guardar_json(archivo_destino, resultados)

    # ------------------------------------------------------------------
    # Guardado de una COMPARATIVA multi-semilla (lo que pide el tutor:
    # media, desviación, mejor y peor resultado por arquitectura)
    # ------------------------------------------------------------------
    @classmethod
    def guardar_comparativa_semillas(cls, modelo_base_nombre, resultados_por_semilla,
                                      dataset, arquitectura=None, notas="",
                                      tiempos_entrenamiento_seg=None,
                                      archivo_destino=None):
        """
        resultados_por_semilla: dict {semilla: metricas_dict}
            metricas_dict es la salida de evaluar_modelo() (metricas_comunes.py),
            o cualquier dict que incluya al menos 'roi_pct' (o 'profit_pct').
        tiempos_entrenamiento_seg: dict opcional {semilla: segundos}
        """
        if archivo_destino is None:
            archivo_destino = cls.RESULTADOS_FILE

        resultados = cls._cargar(archivo_destino)
        rois = []

        for semilla, metricas in resultados_por_semilla.items():
            roi = metricas.get('roi_pct', metricas.get('profit_pct'))
            rois.append(roi)

            nombre_run = f"{modelo_base_nombre}_seed{semilla}"
            entrada = {cls._a_tipo_serializable_key(k): cls._a_tipo_serializable(v)
                       for k, v in metricas.items()}
            entrada.update({
                'semilla': int(semilla),
                'dataset': dataset,
                'notas': notas,
                'timestamp': datetime.now().isoformat(),
                'fecha_legible': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            if arquitectura is not None:
                entrada['arquitectura'] = arquitectura
            if tiempos_entrenamiento_seg and semilla in tiempos_entrenamiento_seg:
                entrada['tiempo_entrenamiento_seg'] = float(tiempos_entrenamiento_seg[semilla])

            resultados[nombre_run] = entrada

        resumen = {
            'arquitectura': arquitectura,
            'semillas': [int(s) for s in resultados_por_semilla.keys()],
            'roi_mean_pct': float(np.mean(rois)),
            'roi_std_pct': float(np.std(rois)),
            'roi_best_pct': float(np.max(rois)),
            'roi_worst_pct': float(np.min(rois)),
            'dataset': dataset,
            'notas': notas,
            'timestamp': datetime.now().isoformat(),
        }
        resultados[f"{modelo_base_nombre}_RESUMEN"] = resumen

        cls._guardar_json(archivo_destino, resultados)
        return resumen

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def _cargar(cls, archivo_destino):
        if os.path.exists(archivo_destino):
            with open(archivo_destino, 'r') as f:
                return json.load(f)
        return {}

    @classmethod
    def _guardar_json(cls, archivo_destino, resultados):
        with open(archivo_destino, 'w') as f:
            json.dump(resultados, f, indent=2)

    @staticmethod
    def _a_tipo_serializable(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    @staticmethod
    def _a_tipo_serializable_key(k):
        return k