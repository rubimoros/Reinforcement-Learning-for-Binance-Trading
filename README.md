# Reinforcement-Learning-for-Binance-Trading
Trabajo de Fin de Grado de un agente automático para trading algorítmico con Binance para el par USDT/BTC

Este repositorio contiene un sistema completo de Trading Algorítmico automatizado utilizando algoritmos de aprendizaje por refuerzo profundo (PPO, RecurrentPPO con LSTM, CNN, SAC). El agente evalúa las condiciones del mercado, calcula indicadores técnicos y se conecta a la Testnet de Binance para operar en tiempo real gestionando el riesgo de forma eficiente.

---

## Arquitectura del Proyecto

La estructura del código está dividida exactamente en las siguientes carpetas y archivos:

*   **`requirements.txt`**: Listado de dependencias necesarias para correr el proyecto.
*   **`resultados_entrenamientos.json`**: Registro donde se almacenan las métricas completas de cada modelo entrenado.
*   **`resultados_comparativa_tfg.json`**: Registro de métricas tras someter los modelos a diferentes periodos de evaluación históricos.
*   **`resultados_viewer.html`**: Dashboard interactivo y visor profesional para analizar visualmente los dos archivos JSON anteriores.
*   **`ui.py`**: Interfaz interactiva por consola que permite al usuario final lanzar evaluaciones, paper trading en vivo o consultar el oráculo.

**Directorios principales:**

### 1. `utils/` (Herramientas y Configuración)
*   **`config.ini`**: **[IMPORTANTE]** Archivo donde se configuran las URLs de los endpoints y las credenciales de Binance.
*   **`config_params.py`**: Lector de configuración para el archivo `.ini`.
*   **`klines_example.py`**: Funciones para conexión a Binance, solicitud de velas, saldos y comisiones.
*   **`indicators_new.py` / `indicators_final.py`**: Motores de cálculo de indicadores técnicos (SMA, MACD, RSI, Bandas de Bollinger, ATR, etc.).
*   **`results_manager.py`**: Gestor de persistencia que vuelca las métricas de los entrenamientos en los ficheros JSON.

### 2. `train/` (Entrenamiento de Modelos)
Contiene los scripts que construyen y entrenan a los agentes usando distintos algoritmos. Los modelos `.zip` y logs se crearán en tiempo de ejecución.
*   **`entrenar.py`**: Entrena la arquitectura `RecurrentPPO` (LSTM) con decaimiento de learning rate y guardado de checkpoints.
*   **`entrenar_PPOMLP.py`**: Entrenamiento básico con la arquitectura clásica `PPO` (MLP).
*   **`entrenarCNN.py`**: Entrenamiento utilizando redes convolucionales 1D.
*   **`entrenarCNN-LSTM.py`**: Entrenamiento híbrido extrayendo características espaciales y temporales.
*   **`entrenarSAC.py`**: Entrenamiento bajo el algoritmo Soft Actor-Critic.
*   **`experimento_boom_BTC.py`**: Script de prueba de estrés obligando al modelo a memorizar una tendencia alcista.

### 3. `scalers/` (Preprocesamiento)
*   **`scaler.py`**: Módulo que ajusta y genera el objeto `StandardScaler`
*   **`scaler_15m.pkl`**: Archivo serializado con las medias y desviaciones del dataset, utilizado para normalizar los datos en vivo del entorno de producción de forma coherente

### 4. `producción/` (Ejecución en Vivo y Evaluación)
*   **`binance.log`**: Archivo donde se registran los eventos locales de red y conexión a Binance
*   **`circuit_breaker.py`**: Sistema de gestión de riesgos que bloquea compras por encima de un % del balance o si el *Drawdown* supera el límite (-25%)
*   **`compraVenta.py`**: Módulo ejecutor. Envía peticiones (BUY, SELL) a Binance Testnet, maneja modos *Maker/Taker* y verifica los estados de las órdenes
*   **`evaluador.py`**: Carga el modelo candidato (ej. los checkpoints) y los enfrenta a los 7 periodos CSV para generar el JSON de comparativas
*   **`generar_evaluaciones.py`**: Descarga automáticamente de Binance el histórico para los 7 periodos de evaluación y calcula los indicadores
*   **`lstm_controller.py`**: Controlador principal de producción. Descarga las últimas 200 velas en vivo, prepara la secuencia para la red LSTM, realiza la predicción (`Comprar/Vender/Esperar`) y arroja el veredicto por consola
*   **`operaciones.py`**: Generador del log CSV
*   **`operaciones_por_agente.csv`**: Registro persistente de todas las acciones que ejecuta el agente en la Testnet (Fecha, Acción, Precio, Cantidad, Estado, Razón)

---

## Configuración de Binance Testnet (API Keys)

Para que el sistema de producción se conecte a tu cuenta de simulación de Binance, debes insertar tus credenciales en el archivo `utils/config.ini`

1. Entra en [Binance Testnet](https://testnet.binance.vision/) y genera una clave API
2. Abre el archivo `utils/config.ini`
3. Baja hasta la sección **`# --- TESTNET cartera producción ---`**
5. Sustituye el texto por tus claves
   api_key_test=TU_API_KEY
   api_secret_test=TU_API_SECRET
   url_base_api_test=[https://testnet.binance.vision](https://testnet.binance.vision)

---

# Flujo de Trabajo

El ciclo de vida del proyecto sigue una secuencia bien definida, desde la generación de los datos de entrenamiento hasta la ejecución del agente en un entorno de producción mediante la interfaz de consola.

## 1. Generación del Dataset de Entrenamiento

El primer paso consiste en obtener los datos históricos del mercado y preparar el conjunto de datos que servirá para entrenar al agente.

En primer lugar, se ejecuta el script encargado de descargar el histórico de velas de Binance y calcular todos los indicadores técnicos necesarios:

```bash
python producción/generar_dataset_completo.py
```

Una vez generado el histórico, es necesario etiquetar los datos mediante el algoritmo de puntuación basado en ATR, que identifica automáticamente los puntos óptimos de compra y venta.

```bash
python producción/generar_puntuaciones_picos_valles_atr.py
```

El dataset final se almacenará dentro del directorio:

```
data/
```

---

## 2. Generación de los Datasets de Evaluación

Para evaluar la capacidad de generalización del modelo se generan varios conjuntos de datos independientes que representan diferentes condiciones del mercado (mercados alcistas, bajistas, alta volatilidad, crash de COVID-19, etc.).

```bash
python producción/generar_evaluaciones.py
```

Los siete históricos generados se almacenarán en:

```
data/evaluaciones_tfg/
```

Estos conjuntos de datos se utilizan exclusivamente para evaluación y no intervienen durante el entrenamiento del modelo.

---

## 3. Entrenamiento del Modelo

Una vez preparados los datos, puede iniciarse el proceso de entrenamiento del agente.

```bash
python train/entrenar.py USD
```

Durante esta fase se realizan automáticamente las siguientes operaciones:

- Carga del dataset de entrenamiento.
- Ajuste del `StandardScaler`.
- Entrenamiento del modelo LSTM basado en Reinforcement Learning.
- Guardado periódico de checkpoints.
- Registro de métricas para su posterior análisis.

### Archivos generados

Los modelos entrenados se almacenan en:

```
models/
```

Los checkpoints intermedios se guardan en:

```
models/checkpoints/
```

El `StandardScaler` utilizado durante el entrenamiento se almacena en:

```
scalers/
```

### Monitorización con TensorBoard

El progreso del entrenamiento puede visualizarse en tiempo real ejecutando en otra terminal:

```bash
tensorboard --logdir logs/tensorboard
```

TensorBoard permite analizar métricas como la recompensa acumulada, pérdidas, evolución del aprendizaje y otros indicadores relevantes durante el entrenamiento.

---

## 4. Evaluación de Modelos

Tras finalizar el entrenamiento, los modelos pueden evaluarse automáticamente sobre los siete periodos históricos generados anteriormente.

```bash
python producción/evaluador.py
```

El evaluador compara el rendimiento de todos los checkpoints disponibles y genera un informe consolidado en:

```
resultados_comparativa_tfg.json
```

Este archivo recoge las métricas de rendimiento obtenidas por cada modelo durante las distintas evaluaciones.

---

## 5. Ejecución del Sistema

La ejecución del sistema se realiza mediante la interfaz de consola incluida en el proyecto.

```bash
python ui.py
```

La aplicación ofrece tres modos de funcionamiento:

### Evaluación del Agente

Ejecuta una evaluación rápida del modelo sobre los siete escenarios históricos disponibles y muestra un resumen de las métricas obtenidas.

### Paper Trading

Inicia el entorno de producción conectado a Binance Testnet.

Durante su ejecución:

- Se descargan datos del mercado en tiempo real.
- El modelo realiza inferencias continuamente.
- El Circuit Breaker supervisa el riesgo de la cartera.
- Las órdenes de compra y venta se envían a Binance Testnet.
- Todas las operaciones quedan registradas automáticamente en:

```
producción/operaciones_por_agente.csv
```

### Oráculo

Permite consultar el estado actual del mercado sin ejecutar ninguna operación.

El agente analiza los datos más recientes y devuelve una recomendación:

- **COMPRAR**
- **VENDER**
- **ESPERAR**

Este modo resulta especialmente útil para validar las predicciones del modelo sin realizar operaciones sobre la cuenta de simulación.

---

# Visualización y Análisis de Resultados

El proyecto incorpora un visor web que permite analizar de forma interactiva los resultados obtenidos durante el entrenamiento y la evaluación de los modelos.

Para utilizarlo:

1. Abrir el archivo `resultados_viewer.html` en cualquier navegador moderno.
2. Pulsar **"Cargar Resultados"**.
3. Seleccionar uno de los archivos JSON generados por el sistema:
   - `resultados_entrenamientos.json`
   - `resultados_comparativa_tfg.json`

El visor genera automáticamente diferentes herramientas de análisis, entre las que se incluyen:

- Heatmaps comparativos de rendimiento.
- Rankings globales de modelos.
- Gráficos Radar para comparar dos modelos.
- Análisis de riesgo mediante Drawdown y Win Rate.
- Comparativas detalladas de las métricas obtenidas en cada evaluación.

Este dashboard facilita la selección del modelo más adecuado antes de su despliegue en producción.
