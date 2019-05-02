#Modulos
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import scipy.stats as sc
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
#lectura de los datos
datos = pd.read_csv('C:/Users/lenovo/Downloads/monthly-milk-production-pounds-p.csv')
datos.columns = ['Mes','Produccion de leche']#nombramos la columna
datos.drop(168,axis=0,inplace=True)#quitamos los valores vacios
datos['Mes']=pd.to_datetime(datos['Mes'])#creamos un indice
datos.set_index('Mes',inplace=True)#le indicamos el indice al dataframe
datos.index = pd.DatetimeIndex(datos.index.values,freq=datos.index.inferred_freq)
##prueba de estacionariedad
est01 =  adfuller(datos['Produccion de leche'])
#grafico serie original
fig,ax = plt.subplots()
ax.plot(datos['Produccion de leche'],label='Produccion de leche')
plt.legend(loc='upper left')
plt.title('Produccion mensual de leche por vaca 1962-1975')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de leche en libras')
plt.show()
#p.value 0.6274267086030331, no es estacionaria
##transformacion de box-cox
sert = sc.boxcox(datos['Produccion de leche'])
#grafico serie transformada
fig,ax = plt.subplots()
ax.plot(sert[0],label='Produccion de leche transformada')
plt.legend(loc='upper left')
plt.title('Produccion mensual de leche por vaca 1962-1975')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de leche en libras')
plt.show()
#lamda de 0.8382967311348767
sertt= sert[0]
est02 = adfuller(sertt)#p.value de 0.5969113535585175, tampoco es estacionaria
#prueba de diferenciacion
serdif = datos.diff()
serdif = serdif.dropna()#serie diferenciada sin datos faltantes
est03 = adfuller(serdif['Produccion de leche'])
#grafico serie diferenciada
fig,ax = plt.subplots()
ax.plot(serdif,label='Produccion de leche diferenciada')
plt.legend(loc='upper left')
plt.title('Produccion mensual de leche por vaca 1962-1975')
plt.xlabel('Fecha')
plt.ylabel('Cantidad de leche en libras')
plt.show()
#p.value 0.030068004001784693, ya es estacionaria al 5%
#funcion de autocorrelacion
plt.figure(figsize=(8,8))
plt.subplot(211)
plot_acf(serdif,lags=30,ax=plt.gca(),title='ACF')
plt.subplot(212)
plot_pacf(serdif,lags=30,ax=plt.gca(),title='PACF')
plt.show()
#dado que la serie esta diferenciada, tendremos un ARIMA(p,1,q)
#pronostico con ARIMA
ser_d_t = datos['Produccion de leche']
lent = int(len(ser_d_t)*0.8)
entre = ser_d_t[:lent]
prueb = ser_d_t[lent:]
mod = ARIMA(entre,order=(1,1,4))#seleccionado con auto arima
mod_ajus = mod.fit()
predic = mod_ajus.forecast(steps=len(prueb))
pron = pd.DataFrame(predic[0]).set_index(prueb.index)
#grafico del pronostico
fig, ax=plt.subplots(figsize=(8,4))
ax.plot(entre,label='Entrenamiento')
ax.plot(prueb,label='Prueba')
ax.plot(pron, label='Pronostico ARIMA(1,1,4)')
plt.legend(loc='upper left')
plt.title('ARIMA')
plt.ylabel('Produccion de leche en libras')
plt.xlabel('Fecha')
plt.savefig('C:/Users/lenovo/Documents/Python Scripts/arima.png')
plt.show()
#dado que la serie presenta estacionalidad, es necesario usar el arima estacional
mod02 = SARIMAX(entre,order=(0,1,1),seasonal_order=(0,1,1,12))#seleccionado con autoarima
mod02_ajus = mod02.fit()
predic02 = mod02_ajus.forecast(steps=len(prueb))
pron02 = pd.DataFrame(predic02).set_index(prueb.index)
#grafico del arima estacional
fig, ax=plt.subplots(figsize=(8,4))
ax.plot(entre,label='Entrenamiento')
ax.plot(prueb,label='Prueba')
ax.plot(pron02, label='Pronostico SARIMA(0,1,1),(0,1,1)[12]')
plt.legend(loc='upper left')
plt.title('SARIMA')
plt.ylabel('Produccion de leche en libras')
plt.xlabel('Fecha')
plt.savefig('C:/Users/lenovo/Documents/Python Scripts/sarima.png')
plt.show()




















