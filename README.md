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
4. Sustituye el texto por tus claves
   api_key_test=TU_API_KEY
   api_secret_test=TU_API_SECRET
   url_base_api_test=[https://testnet.binance.vision](https://testnet.binance.vision)
