
#importar los modulos necesarios
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
#leer la base de datos que contiene la serie
datos = pd.read_csv('C:/Users/lenovo/Downloads/televisa.csv')
dat = np.array(datos)#convertir los datos en vector para ordenarlos
dat2 = np.array(datos)
for i in range (len(datos)):
    dat[i]=dat2[(len(datos)-1)-i]
dat3 = np.array(dat)    #vector de datos ya ordenado
serie = pd.DataFrame(dat3)   
serie.columns=['Fecha','Cierre'] #nombrar las columnas
serie =serie.convert_objects(convert_numeric=True)#parsear los numeros
serie['Fecha'] = pd.to_datetime(serie['Fecha'],format='%d/%m/%Y')#parsear las fechas
serie = serie.set_index('Fecha') #fijar las fechas como indice
prop = int(len(serie)*.08)
entre = serie[0:prop]#datos de entrenamiento
prueb = serie[(prop+1):]#datos de prueba
#suavizado exponencial simple a la serie con un alpha de 0.2
ses = SimpleExpSmoothing(entre).fit(smoothing_level=0.8)
#pronostico de12 periodos co el suavizado simple
proses = ses.forecast(steps=len(prueb))
pses = pd.DataFrame(proses).set_index(prueb.index)
#grafico
fig, ax=plt.subplots(figsize=(8,4))
ax.plot(entre,label='Entrenamiento')
ax.plot(prueb,label='Prueba')
ax.plot(pses, label='Pronostico SES')
plt.legend(loc='upper right')
plt.title('Suavizado exponencial simple')
plt.ylabel('Precio de cierre')
plt.xlabel('Fecha')
plt.show()





