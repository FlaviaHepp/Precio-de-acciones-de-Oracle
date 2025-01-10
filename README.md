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

