#importar los modulos necesarios
import pandas as pd
import statsmodels.api as sm
import numpy as np
#leer la base de datos que contiene la serie
datos = pd.read_csv('C:/Users/lenovo/Downloads/televisa.csv')
#ordenar de mas antiguo a mas reciente
dat = np.array(datos)
dat2 = np.array(datos)
for i in range (len(datos)):
    dat[i]=dat2[(len(datos)-1)-i]
dat3 = np.array(dat)    
serie = pd.DataFrame(dat3)   
serie.columns=['Fecha','Cierre'] #nombrar las columnas
serie =serie.convert_objects(convert_numeric=True)#parsear los numeros
serie['Fecha'] = pd.to_datetime(serie['Fecha'],format='%d/%m/%Y')#parsear las fechas
serie = serie.set_index('Fecha') #fijar las fechas como indice
comp = sm.tsa.seasonal_decompose(serie,freq=251)#descomponer la serie
obser = comp.observed #serie observada
tendencia = comp.trend #tendencia
estacionalidad = comp.seasonal #estacionalidad
resid = comp.resid #residuales
#grafico metodo aditivo
comp.plot()
#metodo multiplicativo
compm = sm.tsa.seasonal_decompose(serie,freq=251,model='multiplicative')
#descomponer la serie
obserm = comp.observed #serie observada
tendenciam = comp.trend #tendencia
estacionalidadm = comp.seasonal #estacionalidad
residm = comp.resid #residuales
#grafico metodo multiplicativo
compm.plot()