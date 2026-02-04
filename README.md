# 📈 Predicción del Precio de las Acciones de Oracle

Este proyecto realiza un análisis completo de series temporales y modelado predictivo sobre los precios históricos de las acciones de Oracle Corporation, integrando técnicas de análisis estadístico, econometría y Machine Learning.

🎯 Objetivos del proyecto
- Analizar la evolución histórica del precio y volumen de las acciones de Oracle.
- Descomponer la serie temporal en tendencia, estacionalidad y residuo.
- Evaluar la estacionariedad mediante la prueba ADF.
- Modelar el comportamiento del precio con ARIMA.
- Analizar la volatilidad utilizando modelos GARCH.
- Construir un modelo de regresión supervisada para predicción del precio.
- Evaluar el desempeño de los modelos con métricas cuantitativas.

📁 Descripción del dataset
El dataset contiene información bursátil diaria de Oracle, incluyendo:
- Date: fecha de negociación
- Open: precio de apertura
- High: precio máximo
- Low: precio mínimo
- Close: precio de cierre
- Volume: volumen negociado

Se generan variables adicionales como:
- Año, mes y día
- Retornos diarios

📊 Análisis exploratorio y estadístico
- Visualizaciones
  -- Evolución del precio de cierre.
  -- Evolución del volumen de operaciones.
  -- Media móvil y desviación estándar.
  -- Análisis precio–volumen.
  -- Descomposición de series temporales
  -- Modelo multiplicativo
- Identificación de:
  -- Tendencia
  -- Estacionalidad
  -- Residuo

  ⏱️ Análisis de series temporales
- Pruebas estadísticas
- ADF (Augmented Dickey-Fuller) para evaluar estacionariedad.
- Funciones de autocorrelación (ACF) y autocorrelación parcial (PACF).
- Modelos implementados
- ARIMA (5,1,0) para predicción del precio.
- Evaluación visual del pronóstico sobre conjunto de prueba.
- GARCH (1,1) para modelar la volatilidad de los retornos.

🤖 Modelado predictivo
Se implementa un modelo de Regresión Lineal utilizando variables temporales:
- Features
- Año
- Mes
- Día
- Métricas de evaluación
- Mean Squared Error (MSE)
- R² Score
- Se comparan valores reales vs. predichos mediante visualización temporal.

🛠️ Tecnologías utilizadas
- Python
- Pandas / numpy
- Matplotlib / Seaborn
- statsmodels
- scikit-learn
- arch (GARCH)

📂 Estructura del proyecto
├── Predicción de precios de acciones de Oracle.py
├── Oracle Dataset.csv
└── README.md

📌 Resultados principales
- Identificación clara de componentes de tendencia y estacionalidad.
- La serie original no es estacionaria, requiriendo diferenciación.
- El modelo ARIMA captura adecuadamente la dinámica temporal del precio.
- El modelo GARCH permite analizar la volatilidad de los retornos.
- La regresión lineal temporal ofrece una aproximación base para predicción.

⚠️ Disclaimer

Este proyecto tiene fines educativos y analíticos.
No constituye asesoramiento financiero ni recomendaciones de inversión.

👤 Autor

Flavia Hepp
Data Science · Econometría · Series Temporales
