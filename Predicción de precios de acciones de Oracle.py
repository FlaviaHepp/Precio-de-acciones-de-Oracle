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



