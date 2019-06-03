#importar los modulos necesarios
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
#leer la base de datos que contiene la serie
datos = pd.read_csv('C:/Users/lenovo/Downloads/televisa.csv')
datos.columns = ['Fecha','Cierre']#nombramos la columna
datos['Fecha']=pd.to_datetime(datos['Fecha'],format='%d/%m/%Y')#creamos un indice
datos['Fecha']=pd.to_datetime(datos['Fecha'])#creamos un indice
datos.set_index('Fecha',inplace=True)
serie = datos.asfreq(freq='B').interpolate()   
entre = serie[0:int(len(serie)*.8)]#datos de entrenamiento
prueb = serie[int(len(serie)*.8)+1:]#datos de prueba
#suavizado exponencial simple a la serie con un alpha de 0.2
ses = SimpleExpSmoothing(entre).fit(smoothing_level=0.5)
#pronostico de12 periodos co el suavizado simple
proses = ses.forecast(steps=len(prueb))
pses = pd.DataFrame(proses).set_index(prueb.index)
#grafico
fig, ax=plt.subplots(figsize=(8,4))
ax.plot(entre,label='Entrenamiento')
ax.plot(prueb,label='Prueba')
ax.plot(pses, label='Pronostico SES')
ax.plot(ses.fittedvalues,label='Valores ajustados')
plt.legend(loc='upper right')
plt.title('Suavizado exponencial simple')
plt.ylabel('Precio de cierre')
plt.xlabel('Fecha')
plt.show()





