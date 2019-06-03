#Modulos
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
#lectura de los datos
serie = pd.read_csv('C:/Users/lenovo/Downloads/televisa.csv')
serie.columns = ['Fecha','Cierre']#nombramos la columna
serie['Fecha']=pd.to_datetime(serie['Fecha'],format='%d/%m/%Y')#creamos un indice
serie['Fecha']=pd.to_datetime(serie['Fecha'])#creamos un indice
serie.set_index('Fecha',inplace=True)#incorporar el indice
ser_tel = serie.asfreq(freq='B').interpolate()#interpolar NA
#metodo aditivo
compo = sm.tsa.seasonal_decompose(ser_tel,model='additive')
obser = compo.observed
tendencia = compo.trend
estacio = compo.seasonal
resid = compo.resid
#grafico metodo aditivo
plt.figure(figsize=(8,8))
plt.subplot(411)
plt.title("Descomposicion por metodo aditivo",loc='center')
plt.plot(obser)
plt.subplot(412)
plt.plot(tendencia)
plt.subplot(413)
plt.plot(estacio)
plt.subplot(414)
plt.plot(resid)
plt.show()
#metodo multiplicativo
compm = sm.tsa.seasonal_decompose(ser_tel,model='multiplicative')
#descomponer la serie
obserm = compm.observed #serie observada
tendenciam = compm.trend #tendencia
estacionalidadm = compm.seasonal #estacionalidad
residm = compm.resid #residuales
#grafico metodo multiplicativo
plt.figure(figsize=(8,8))
plt.subplot(411)
plt.title("Descomposicion por metodo multiplicativo",loc='center')
plt.plot(obserm)
plt.subplot(412)
plt.plot(tendenciam)
plt.subplot(413)
plt.plot(estacionalidadm)
plt.subplot(414)
plt.plot(residm)
plt.show()
