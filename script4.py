#importar los modulos
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib.pyplot as plt
#lectura de los datos
datos = pd.read_csv('C:/Users/lenovo/Downloads/televisa.csv')
datos.columns = ['Fecha','Cierre']#nombramos la columna
datos['Fecha']=pd.to_datetime(datos['Fecha'],format='%d/%m/%Y')#creamos un indice
datos['Fecha']=pd.to_datetime(datos['Fecha'])#creamos un indice
datos.set_index('Fecha',inplace=True)
serie = datos.asfreq(freq='B').interpolate()   
entre = serie[0:int(len(serie)*.8)]#datos de entrenamiento
prueb = serie[int(len(serie)*.8)+1:]#datos de prueba
#suavizado Holt Winters con tendencia y estacionalidad aditivas
hholtwintaa = ExponentialSmoothing(entre,seasonal_periods=8,
   trend='add',seasonal='add').fit()
#pronostico 
proholtwintaa = hholtwintaa.forecast(steps=len(prueb))
pholtwintaa = pd.DataFrame(proholtwintaa).set_index(prueb.index)
#grafico a-a
fig, ax=plt.subplots(figsize=(8,4))
ax.plot(entre,label='Entrenamiento')
ax.plot(hholtwintaa.fittedvalues,label='Valores ajustados')
ax.plot(prueb,label='Prueba')
ax.plot(pholtwintaa, label='Pronostico HW ad-ad')
plt.legend(loc='upper right')
plt.title('Suavizado Holt Winters')
plt.ylabel('Precio de cierre')
plt.xlabel('Fecha')
plt.show()

