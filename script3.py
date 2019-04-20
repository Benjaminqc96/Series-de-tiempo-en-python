#importar los modulos
import numpy as np
import pandas as pd
from statsmodels.tsa.api import Holt
import matplotlib.pyplot as plt
#lectura de los datos
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
entre = serie[0:402]#datos de entrenamiento
prueb = serie[403:] #datos de prueba
#suavizado Holt
hholt = Holt(entre).fit(smoothing_level=0.8)
#pronostico
proholt = hholt.forecast(steps=len(prueb))
pholt = pd.DataFrame(proholt).set_index(prueb.index)
#grafico
fig, ax=plt.subplots(figsize=(8,4))
ax.plot(entre,label='Entrenamiento')
ax.plot(prueb,label='Prueba')
ax.plot(pholt, label='Pronostico Holt')
plt.legend(loc='upper right')
plt.title('Suavizado Holt')
plt.ylabel('Precio de cierre')
plt.xlabel('Fecha')
plt.show()


