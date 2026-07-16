# Reinforcement-Learning-for-Binance-Trading
Trabajo de Fin de Grado de un agente automático para trading algorítmico con Binance para el par USDT/BTC

Este repositorio contiene un sistema completo de Trading Algorítmico automatizado utilizando algoritmos de aprendizaje por refuerzo profundo (PPO, RecurrentPPO con LSTM, CNN, SAC). El agente evalúa las condiciones del mercado, calcula indicadores técnicos y se conecta a la Testnet de Binance para operar en tiempo real gestionando el riesgo de forma eficiente.

El sistema evalúa las condiciones del mercado, calcula indicadores técnicos, y se conecta a la Testnet de Binance para operar en tiempo real gestionando el riesgo mediante un Circuit Breaker. El modelo de producción es un `RecurrentPPO` (política LSTM), seleccionado tras comparar varias arquitecturas (PPO/MLP, CNN 1D, CNN-LSTM, SAC).

---

## Arquitectura del proyecto

```
.
├── requirements.txt
├── resultados_entrenamientos.json
├── resultados_comparativa_tfg.json
├── resultados_viewer.html
├── data/
│   ├── dataset_puntuaciones_picos_valles_atr_15minutos2021_24.csv
│   └── evaluaciones_tfg/
│       ├── dataset_1_Marzo_2020_Covid.csv
│       ├── dataset_2_Febrero_2021_Bull.csv
│       ├── dataset_3_Julio_2021_Lateral.csv
│       ├── dataset_4_Junio_2022_Bear.csv
│       ├── dataset_5_Noviembre_2022_Cambio.csv
│       ├── dataset_6_Agosto_2023_Volatilidad.csv
│       └── dataset_7_Marzo_2024_BullReciente.csv
├── datasets/
│   ├── generar_dataset_completo.py
│   ├── generar_puntuaciones_picos_valles_atr.py
│   ├── generar_puntuaciones_picos_valles.py
│   ├── generar_puntuaciones.py
│   ├── crear_dataset.py
│   └── aplicar_pca.py
├── envs/
│   ├── trading_env.py
│   └── trading_env_v2.py
├── train/
│   ├── entrenar.py
│   ├── entrenarCNN.py
│   ├── entrenarCNN-LSTM.py
│   ├── entrenarSAC.py
│   ├── entrenar_PPOMLP.py
│   └── experimento_boom_BTC.py
├── models/
│   └── checkpoints/
├── scalers/
│   ├── scaler.py
│   └── scaler_15m.pkl
├── logs/
│   └── tensorboard/
├── producción/
│   ├── ui.py
│   ├── lstm_controller.py
│   ├── compraVenta.py
│   ├── circuit_breaker.py
│   ├── evaluador.py
│   ├── generar_evaluaciones.py
│   ├── operaciones.py
│   ├── operaciones_por_agente.csv
│   └── binance.log
└── utils/
    ├── config.ini
    ├── config_params.py
    ├── indicators_final.py
    ├── indicators_new.py
    ├── klines_example.py
    └── results_manager.py
```

### Descripción por carpeta

**`datasets/`** — Generación de los datos de entrenamiento.
- `generar_dataset_completo.py`: descarga el histórico de velas de Binance y calcula todos los indicadores técnicos.
- `generar_puntuaciones_picos_valles_atr.py`: etiqueta el dataset con el sistema de puntuaciones basado en ATR (picos y valles), usado como guía del agente durante el entrenamiento.
- `crear_dataset.py`, `generar_puntuaciones.py`, `generar_puntuaciones_picos_valles.py`, `aplicar_pca.py`: versiones previas o auxiliares del pipeline de datos.

**`envs/`** — Entornos de simulación (Gymnasium) donde opera el agente.
- `trading_env_v2.py`: entorno actual, con acciones continuas, comisiones, spread, Circuit Breaker y sistema de puntuaciones ATR.
- `trading_env.py`: versión inicial, conservada como referencia histórica.

**`train/`** — Scripts de entrenamiento, uno por arquitectura.
- `entrenar.py`: `RecurrentPPO` (LSTM), con decaimiento de learning rate y `CheckpointCallback`. Es el que produce el modelo de producción.
- `entrenarCNN.py` / `entrenarCNN-LSTM.py`: variantes convolucional pura e híbrida.
- `entrenarSAC.py`, `entrenar_PPOMLP.py`: arquitecturas comparativas descartadas.
- `experimento_boom_BTC.py`: prueba de estrés forzando al agente a memorizar una tendencia alcista concreta.

**`models/`** — Modelos entrenados (`.zip` de Stable-Baselines3 / sb3-contrib).
- `models/checkpoints/`: snapshots periódicos guardados durante el entrenamiento (cada 25.000 steps), usados para elegir el punto de corte con mejor generalización en lugar del modelo final por defecto.

**`scalers/`** — Normalización de los indicadores.
- `scaler.py`: ajusta el `StandardScaler` sobre el conjunto de entrenamiento y lo serializa.
- `scaler_15m.pkl`: scaler ya ajustado, usado tanto en entrenamiento (val/test) como en producción. Debe ser siempre el mismo objeto en ambos lados; no se reajusta en producción.

**`logs/`** — Salida de entrenamiento: monitor CSVs y logs de TensorBoard (`logs/tensorboard/`).

**`producción/`** — Ejecución en vivo contra Binance Testnet.
- `ui.py`: interfaz de consola, punto de entrada del sistema en producción.
- `lstm_controller.py`: descarga velas en vivo, calcula indicadores, aplica el scaler y consulta al modelo.
- `compraVenta.py`: ejecuta las órdenes reales (BUY/SELL) contra la API de Binance.
- `circuit_breaker.py`: bloquea la operativa si el drawdown supera el límite configurado o si una orden compromete más del porcentaje de capital permitido.
- `evaluador.py`: evalúa uno o varios modelos candidatos contra los 7 periodos históricos de `data/evaluaciones_tfg/`.
- `generar_evaluaciones.py`: descarga y prepara esos 7 datasets de evaluación.
- `operaciones.py` / `operaciones_por_agente.csv`: registro persistente de toda operación ejecutada por el agente.
- `binance.log`: log de conexión y eventos de red con Binance.

**`utils/`** — Utilidades compartidas por el resto del proyecto.
- `config.ini` / `config_params.py`: credenciales y endpoints de la API (ver sección siguiente).
- `indicators_final.py`: motor de cálculo de indicadores técnicos actual (versiones relativas al precio de cierre, estacionarias). Es el que debe usarse en cualquier script nuevo.
- `indicators_new.py`: versión anterior (indicadores en precio absoluto), conservada solo como referencia histórica; no usar para nuevos entrenamientos ni en producción.
- `klines_example.py`: funciones de conexión a Binance (cliente, descarga de velas, saldos, comisiones).
- `results_manager.py`: persistencia de métricas en los JSON de resultados.

---

## Configuración de Binance Testnet (API Keys)

Las credenciales viven en `utils/config.ini`, bajo la sección `# --- TESTNET cartera producción ---`:

```ini
api_key_test=PONER AQUI API-KEY
api_secret_test=PONER AQUI API-SECRET
url_base_api_test=https://testnet.binance.vision
```

Pasos:

1. Entra en [https://testnet.binance.vision](https://testnet.binance.vision) e inicia sesión con tu cuenta de GitHub.
2. Genera una API Key/Secret desde el panel de la Testnet (esto crea también la cartera simulada con 10.000 USDT y 1 BTC).
3. Abre `utils/config.ini` y sustituye `api_key_test` y `api_secret_test` por tus valores.
4. Guarda el archivo. `utils/config_params.py` lo lee automáticamente en cada ejecución; no hace falta tocar ningún otro fichero.

### Importante: no subas tus claves a GitHub

`config.ini` contiene credenciales reales una vez rellenado, y este repositorio está sincronizado con GitHub. Para evitar publicar tus claves:

1. Añade esta línea a un archivo `.gitignore` en la raíz del proyecto:
   ```
   utils/config.ini
   ```
2. Mantén en el repositorio una copia de referencia sin credenciales reales, por ejemplo `utils/config.ini.example`, con los mismos campos vacíos o con el texto `PONER AQUI API-KEY`.
3. Si en algún momento ya subiste tus claves reales a un commit anterior, revócalas desde el panel de la Testnet y genera unas nuevas — quitarlas de `.gitignore` a partir de ahora no las borra del historial de git.

Si quieres, puedo generar ese `.gitignore` y el `config.ini.example` directamente.

---

## Instalación

```bash
pip install -r requirements.txt
```

El `requirements.txt` incluido cubre todas las dependencias detectadas en el código (conexión a Binance, procesamiento de datos, entornos Gymnasium, Stable-Baselines3/sb3-contrib, PyTorch, TensorBoard, e interfaz de consola con `rich`). No requiere instalar nada adicional a mano para arrancar el sistema.

---

## Flujo de trabajo completo

El ciclo de vida del proyecto sigue este orden: generación de datos, entrenamiento, evaluación y despliegue.

### 1. Generar el dataset de entrenamiento

Descarga el histórico de velas y calcula los indicadores técnicos:

```bash
python datasets/generar_dataset_completo.py
```

Etiqueta el dataset con el sistema de puntuaciones ATR (picos y valles):

```bash
python datasets/generar_puntuaciones_picos_valles_atr.py
```

El resultado final se guarda en `data/`.

### 2. Generar los datasets de evaluación

Genera los 7 periodos históricos independientes (Covid 2020, bull 2021, lateral 2021, bear 2022, cambio de tendencia 2022, volatilidad 2023, bull 2024) usados exclusivamente para validar la generalización del modelo, nunca para entrenar:

```bash
python producción/generar_evaluaciones.py
```

Se almacenan en `data/evaluaciones_tfg/`.

### 3. Entrenar el modelo

```bash
python train/entrenar.py USD
```

(El argumento `USD` o `BTC` define el objetivo de recompensa; ver `envs/trading_env_v2.py`.)

Durante el entrenamiento se realiza automáticamente:
- Carga del dataset y ajuste del `StandardScaler` sobre el conjunto de train.
- Entrenamiento del modelo `RecurrentPPO` (LSTM) con decaimiento de learning rate.
- Guardado de checkpoints cada 25.000 steps.
- Registro de métricas en `resultados_entrenamientos.json`.

Archivos generados:
- Modelo final y checkpoints: `models/` y `models/checkpoints/`.
- Scaler ajustado: `scalers/scaler_15m.pkl`.
- Logs de entrenamiento: `logs/`.

Para entrenar las arquitecturas alternativas: `python train/entrenarCNN.py`, `python train/entrenarCNN-LSTM.py`, `python train/entrenarSAC.py`, `python train/entrenar_PPOMLP.py`.

### 4. Monitorizar el entrenamiento con TensorBoard

En una segunda terminal, con el entrenamiento en marcha o ya finalizado:

```bash
tensorboard --logdir logs/tensorboard
```

Abre la URL que indica la consola (por defecto `http://localhost:6006`) para ver recompensa acumulada, pérdidas, entropía y demás métricas de entrenamiento en tiempo real.

### 5. Evaluar los checkpoints contra los 7 periodos históricos

```bash
python producción/evaluador.py
```

Compara el o los modelos candidatos (editables en la lista `CANDIDATOS` dentro de `evaluador.py`) contra los 7 periodos de `data/evaluaciones_tfg/`, usando el `scaler_15m.pkl` fijo — nunca uno reajustado por periodo. Los resultados se consolidan en `resultados_comparativa_tfg.json`.

### 6. Ejecutar el sistema

```bash
python producción/ui.py
```

Menú con tres modos:

- **Evaluación con 7 periodos**: lanza `evaluador.py` desde la propia interfaz y muestra un resumen de métricas.
- **Paper Trading en Vivo**: conecta con Binance Testnet, muestra el saldo real disponible, pide cuánto capital asignar al bot, y entra en un bucle de 15 minutos por vela: descarga mercado, el modelo LSTM decide Comprar/Vender/Esperar, el Circuit Breaker valida el riesgo, y `compraVenta.py` ejecuta la orden real en la Testnet. Cada operación queda registrada en `producción/operaciones_por_agente.csv`. Se puede detener en cualquier momento con Ctrl+C.
- **Veredicto instantáneo (oráculo)**: consulta el estado actual del mercado y devuelve una recomendación puntual (Comprar/Vender/Esperar) sin ejecutar ninguna orden real. Útil para consultar al modelo sin exponer capital.

---

## Dónde se almacenan los resultados

| Tipo de resultado | Ubicación |
|---|---|
| Métricas de cada entrenamiento | `resultados_entrenamientos.json` |
| Métricas de evaluación contra los 7 periodos históricos | `resultados_comparativa_tfg.json` |
| Modelos entrenados (final y checkpoints) | `models/` y `models/checkpoints/` |
| Scaler de producción | `scalers/scaler_15m.pkl` |
| Logs de entrenamiento (TensorBoard, monitor CSV) | `logs/` |
| Operaciones ejecutadas en producción (Testnet) | `producción/operaciones_por_agente.csv` |
| Log de conexión con Binance | `producción/binance.log` |

---

## Visualización y análisis de resultados

El proyecto incluye un dashboard interactivo que no requiere servidor ni instalación:

1. Abre `resultados_viewer.html` en cualquier navegador moderno.
2. Pulsa "Cargar Resultados".
3. Selecciona uno de los dos JSON: `resultados_entrenamientos.json` o `resultados_comparativa_tfg.json`.

El visor genera automáticamente:
- Heatmaps comparativos de rendimiento entre modelos.
- Ranking global de modelos.
- Gráficos de radar para comparar dos modelos entre sí.
- Análisis de riesgo (Drawdown vs Win Rate).
- Tablas detalladas de todas las métricas por evaluación.

Este visor es la forma más rápida de comparar candidatos antes de decidir cuál desplegar en producción (ver `RUTA_MODELO` en `producción/lstm_controller.py`).
