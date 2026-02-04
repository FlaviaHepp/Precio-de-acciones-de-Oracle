# Precio-de-acciones-de-Oracle
Predicción de precios de acciones de Oracle

En este proyecto se buscó analizar y predecir los precios de las acciones de Oracle utilizando métodos estadísticos, análisis de series temporales y técnicas de aprendizaje automático. 

*Herramientas utilizadas:* Python, pandas, numpy, matplotlib, statsmodels y scikit-learn.
**Desarrollo:**
Las fechas se convierten en un índice para facilitar el análisis temporal.
Se verifican valores faltantes y se resumen las estadísticas descriptivas.
**Visualización de los datos:**
Gráficos para entender cómo han evolucionado los precios de cierre y el volumen a lo largo del tiempo.
Se calculan medios y desviaciones móviles estándar para identificar tendencias y fluctuaciones.
**Análisis de series temporales:**
*Descomposición:* Se separan los datos en componentes de tendencia, estacionalidad y ruido.
*Prueba Dickey-Fuller:* Determina si la serie es estacionaria (requisito para ciertos modelos).
*Autocorrelaciones (ACF y PACF):* Ayudan a entender relaciones entre datos pasados ​​y presentes.
**Modelado predictivo:**
*ARIMA:* Para capturar los patrones de dependencia temporal en los precios y dividir los datos en conjuntos de entrenamiento y prueba, ajusta el modelo y realiza pronósticos.
*Garch:* Para analizar la volatilidad diaria de los rendimientos.
*Regresión lineal:* Para predecir precios de cierre.
*Análisis de volatilidad y volumen:* Los rendimientos diarios destacan fluctuaciones significativas, una característica típica del mercado bursátil. Se analiza la relación entre el volumen de operaciones y los precios, revelando una valoración positiva.

**Resultados clave:**
Se observa un crecimiento significativo en los precios de las acciones a lo largo de los años.
El precio de cierre de la acción muestra una tendencia alcista a lo largo del tiempo.
El volumen de operaciones muestra una alta volatilidad, con picos significativos en ciertos períodos.
La descomposición de las series temporales revela tendencias subyacentes, patrones estacionales y componentes de ruido.
La prueba aumentada de Dickey-Fuller indica que la serie temporal no es estacionaria, por lo que es necesario diferenciarla para modelarla.
El modelo ARIMA proporciona pronósticos razonables, capturando la dirección general de los movimientos del precio de las acciones.
Sin embargo, es posible que no capte todas las complejidades y cambios repentinos en los precios de las acciones.
El modelo GARCH captura efectivamente la volatilidad en los rendimientos de las acciones, destacando períodos de mayor y menor volatilidad.
Los pronósticos de volatilidad pueden ser valiosos para la gestión de riesgos y las estrategias comerciales.
El modelo de regresión lineal, aunque simplista, ofrece una predicción básica de los precios de las acciones en función de las características de la fecha.
El rendimiento del modelo se puede mejorar incorporando más funciones y utilizando técnicas avanzadas.
Los rendimientos diarios exhiben una volatilidad significativa, característica del comportamiento del mercado de valores.
Existe una relación positiva entre el volumen de operaciones y los precios de las acciones, lo que sugiere que los aumentos de precios suelen ir acompañados de mayores volúmenes.

📈 Predicción del Precio de las Acciones de Oracle

Este proyecto realiza un análisis completo de series temporales y modelado predictivo sobre los precios históricos de las acciones de Oracle Corporation, integrando técnicas de análisis estadístico, econometría y Machine Learning.

🎯 Objetivos del proyecto

Analizar la evolución histórica del precio y volumen de las acciones de Oracle.

Descomponer la serie temporal en tendencia, estacionalidad y residuo.

Evaluar la estacionariedad mediante la prueba ADF.

Modelar el comportamiento del precio con ARIMA.

Analizar la volatilidad utilizando modelos GARCH.

Construir un modelo de regresión supervisada para predicción del precio.

Evaluar el desempeño de los modelos con métricas cuantitativas.

📁 Descripción del dataset

El dataset contiene información bursátil diaria de Oracle, incluyendo:

Date: fecha de negociación

Open: precio de apertura

High: precio máximo

Low: precio mínimo

Close: precio de cierre

Volume: volumen negociado

Se generan variables adicionales como:

Año, mes y día

Retornos diarios

📊 Análisis exploratorio y estadístico
Visualizaciones

Evolución del precio de cierre.

Evolución del volumen de operaciones.

Media móvil y desviación estándar.

Análisis precio–volumen.

Descomposición de series temporales

Modelo multiplicativo

Identificación de:

Tendencia

Estacionalidad

Residuo

⏱️ Análisis de series temporales
Pruebas estadísticas

ADF (Augmented Dickey-Fuller) para evaluar estacionariedad.

Funciones de autocorrelación (ACF) y autocorrelación parcial (PACF).

Modelos implementados

ARIMA (5,1,0) para predicción del precio.

Evaluación visual del pronóstico sobre conjunto de prueba.

GARCH (1,1) para modelar la volatilidad de los retornos.

🤖 Modelado predictivo

Se implementa un modelo de Regresión Lineal utilizando variables temporales:

Features

Año

Mes

Día

Métricas de evaluación

Mean Squared Error (MSE)

R² Score

Se comparan valores reales vs. predichos mediante visualización temporal.

🛠️ Tecnologías utilizadas

Python

pandas / numpy

Matplotlib / Seaborn

statsmodels

scikit-learn

arch (GARCH)

📂 Estructura del proyecto
├── Predicción de precios de acciones de Oracle.py
├── Oracle Dataset.csv
└── README.md

▶️ Cómo ejecutar el proyecto

Clonar el repositorio

git clone https://github.com/tu_usuario/nombre_del_repo.git


Instalar dependencias

pip install pandas numpy matplotlib seaborn scikit-learn statsmodels arch


Ejecutar el script

python "Predicción de precios de acciones de Oracle.py"

📌 Resultados principales

Identificación clara de componentes de tendencia y estacionalidad.

La serie original no es estacionaria, requiriendo diferenciación.

El modelo ARIMA captura adecuadamente la dinámica temporal del precio.

El modelo GARCH permite analizar la volatilidad de los retornos.

La regresión lineal temporal ofrece una aproximación base para predicción.

⚠️ Disclaimer

Este proyecto tiene fines educativos y analíticos.
No constituye asesoramiento financiero ni recomendaciones de inversión.

👤 Autor

Flavia Hepp
Data Science · Econometría · Series Temporales
