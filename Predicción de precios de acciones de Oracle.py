"""
Oracle Corporation es una empresa multinacional de tecnología conocida por su conjunto completo y totalmente integrado de aplicaciones en 
la nube y servicios de plataforma . Fundada en 1977 por Larry Ellison, Bob Miner y Ed Oates, Oracle ha crecido hasta convertirse en una 
de las empresas de sistemas de software y hardware más grandes del mundo, particularmente reconocida por sus sistemas de gestión de bases 
de datos. A lo largo de los años, Oracle ha ampliado su oferta de productos para incluir soluciones en la nube , software empresarial y 
productos de hardware , manteniendo una fuerte presencia en el mercado tecnológico global. La empresa salió a bolsa en 1986, lo que 
impulsó significativamente su crecimiento y alcance de mercado.

Este conjunto de datos proporciona un registro completo de los cambios en el precio de las acciones de Oracle desde su oferta pública 
inicial en 1986. Incluye columnas esenciales como la fecha , el precio de apertura , el precio más alto del día, el precio más bajo del 
día, el precio de cierre , el precio de cierre ajustado y el volumen de operaciones .

Estos datos extensos son invaluables para realizar análisis históricos, pronosticar el desempeño futuro de las acciones y comprender las 
tendencias del mercado a largo plazo relacionadas con las acciones de Oracle."""

#Análisis completo del conjunto de datos de precios de acciones de Oracle
#Este cuaderno proporciona un análisis completo del conjunto de datos de precios de acciones de Oracle. El análisis incluye preprocesamiento 
# de datos, descriptivo # estadísticas, visualizaciones, análisis de series temporales, modelos predictivos, análisis de volatilidad y 
# análisis de volumen.

#Importar bibliotecas y conjuntos de datos necesarios
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
#Prueba de estacionariedad
from statsmodels.tsa.stattools import adfuller
#Autocorrelación y autocorrelación parcial
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
#Modelado ARIMA
from statsmodels.tsa.arima.model import ARIMA
#GARCH Modelado
from arch import arch_model


df = pd.read_csv('Oracle Dataset.csv')

df['Date'] = pd.to_datetime(df['Date'])

df.set_index('Date', inplace=True)

#Preprocesamiento de datos
print("Valores faltantes en el conjunto de datos:", df.isnull().sum())

#Descriptive Statistics
print("Estadísticas descriptivas del conjunto de datos:", df.describe())

#Visualizaciones
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Cerrar Precio')
plt.title('Precios de cierre de las acciones de Oracle a lo largo del tiempo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Cerrar Precio\n')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(df['Volume'], label='Volumen de comercio', color='orange')
plt.title('Volumen de negociación de acciones de Oracle a lo largo del tiempo\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Volumen\n')
plt.legend()
plt.show()

#Descomposición de series temporales
decomposition = seasonal_decompose(df['Close'], model='multiplicative', period=365)
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.show()

#Estadísticas rodantes
rolling_mean = df['Close'].rolling(window=12).mean()
rolling_std = df['Close'].rolling(window=12).std()

plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Cerrar Precio')
plt.plot(rolling_mean, label='Media rodante', color='fuchsia')
plt.plot(rolling_std, label='Estándar rodante', color='white')
plt.title('Media móvil y desviación estándar\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Cerrar Precio\n')
plt.legend()
plt.show()

result = adfuller(df['Close'])
print('Estadística del ADF:', result[0])
print('valor p:', result[1])
print('Valores criticos:', result[4])

lag_acf = acf(df['Close'], nlags=20)
lag_pacf = pacf(df['Close'], nlags=20)

plt.figure(figsize=(14, 7))
plt.stem(lag_acf)
plt.title('Función de autocorrelación\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Retrasos\n')
plt.ylabel('ACF\n')
plt.show()

plt.figure(figsize=(14, 7))
plt.stem(lag_pacf)
plt.title('Función de autocorrelación parcial\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Retrasos\n')
plt.ylabel('PACF')
plt.show()

train_size = int(len(df) * 0.8)
train, test = df['Close'][:train_size], df['Close'][train_size:]

model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))
plt.figure(figsize=(14, 7))
plt.plot(train, label='Tren')
plt.plot(test, label='Prueba')
plt.plot(forecast, label='Pronóstico', color='c')
plt.title('Pronóstico del modelo ARIMA\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Cerrar Precio\n')
plt.legend()
plt.show()

returns = df['Close'].pct_change().dropna()
model_garch = arch_model(returns, vol='Garch', p=1, q=1)
model_garch_fit = model_garch.fit()

forecast_garch = model_garch_fit.forecast(horizon=5)

#Modelado predictivo e ingeniería de características
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.day

X = df[['Year', 'Month', 'Day']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Error medio cuadrado: {mse}')
print(f'Puntuación R^2: {r2}')

plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicho', linestyle='--')
plt.title('Predicción del precio de las acciones de Oracle\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Cerrar Precio\n')
plt.legend()
plt.show()

#Análisis de volatilidad
df['Returns'] = df['Close'].pct_change()
plt.figure(figsize=(14, 7))
plt.plot(df['Returns'], label='Devoluciones diarias', color='lime')
plt.title('Devoluciones diarias de acciones de Oracle\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Fecha\n')
plt.ylabel('Devoluciones\n')
plt.legend()
plt.show()

#Análisis de volumen
plt.figure(figsize=(14, 7))
sns.regplot(x='Volume', y='Close', data=df, scatter_kws={'alpha':0.3})
plt.title('Precio de las acciones de Oracle frente al volumen de operaciones\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Volumen\n')
plt.ylabel('Cerrar Precio\n')
plt.show()
"""
Conclusiones
Estadísticas descriptivas:

El conjunto de datos proporciona una visión completa de los movimientos del precio de las acciones de Oracle desde su salida a bolsa en 
1986.
Se observa un crecimiento significativo en los precios de las acciones a lo largo de los años.
Visualizaciones:

El precio de cierre de la acción muestra una tendencia alcista a lo largo del tiempo.
El volumen de operaciones muestra una alta volatilidad, con picos significativos en ciertos períodos.
Análisis de series temporales:

La descomposición de las series temporales revela tendencias subyacentes, patrones estacionales y componentes de ruido.
La prueba aumentada de Dickey-Fuller indica que la serie temporal no es estacionaria, por lo que es necesario diferenciarla para modelarla.
Modelado ARIMA:

El modelo ARIMA proporciona pronósticos razonables, capturando la dirección general de los movimientos del precio de las acciones.
Sin embargo, es posible que no capte todas las complejidades y cambios repentinos en los precios de las acciones.
Modelado GARCH:

El modelo GARCH captura efectivamente la volatilidad en los rendimientos de las acciones, destacando períodos de mayor y menor volatilidad.
Los pronósticos de volatilidad pueden ser valiosos para la gestión de riesgos y las estrategias comerciales.
Modelado predictivo:

El modelo de regresión lineal, aunque simplista, ofrece una predicción básica de los precios de las acciones en función de las características de la fecha.
El rendimiento del modelo se puede mejorar incorporando más funciones y utilizando técnicas avanzadas.
Análisis de volatilidad y volumen:

Los rendimientos diarios exhiben una volatilidad significativa, característica del comportamiento del mercado de valores.
Existe una relación positiva entre el volumen de operaciones y los precios de las acciones, lo que sugiere que los aumentos de precios suelen 
ir acompañados de mayores volúmenes.
En general, el análisis proporciona una comprensión integral del comportamiento del precio de las acciones de Oracle a lo largo del tiempo, 
destacando tendencias, volatilidad y posibles enfoques de modelado predictivo. Los conocimientos pueden ayudar a los inversores y analistas 
a tomar decisiones informadas basadas en datos históricos y pronósticos.
"""


