#importar los modulos
import pandas as pd
from statsmodels.tsa.api import Holt
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
#suavizado Holt
hholt = Holt(entre).fit(smoothing_level=0.8,smoothing_slope=.05)
#pronostico
proholt = hholt.forecast(steps=len(prueb))
pholt = pd.DataFrame(proholt).set_index(prueb.index)
#grafico
fig, ax=plt.subplots(figsize=(8,4))
ax.plot(entre,label='Entrenamiento')
ax.plot(hholt.fittedvalues,label='Valores ajustados')
ax.plot(prueb,label='Prueba')
ax.plot(pholt, label='Pronostico Holt')
plt.legend(loc='upper right')
plt.title('Suavizado Holt')
plt.ylabel('Precio de cierre')
plt.xlabel('Fecha')
plt.show()


