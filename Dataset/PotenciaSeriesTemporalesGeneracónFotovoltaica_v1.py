#!/usr/bin/env python
# coding: utf-8

# Aprovecharemos las bondades de Pandas para cargar y tratar nuestros datos. Comenzamos importando las librerías que utilizaremos y leyendo el archivo csv.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('fast')

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler


# In[2]:


df = pd.read_csv('Potencia_Generación_Fotovoltaica.csv',  parse_dates=[0], header=None,index_col=0, squeeze=True,names=['fecha','unidades'])
df.head()


# Datos estadísticos que nos brinda pandas con describe()

# In[3]:


df.describe()


# Veamos cuantas muestras tenemos

# In[4]:


print(len(df['2017-01']))


# De hecho aprovechemos el tener indice de fechas con pandas y saquemos los promedios semanales:

# In[5]:


semanas =df.resample('W').mean()
semanas


# Veamos la gráfica de generación fotovoltaica del mes de enero 2017

# In[6]:


demanda201808 = df['2017-01-01':'2017-01-31']
plt.plot(demanda201808.values)


# Lo que haremos es alterar nuestro flujo de entrada del archivo csv que contiene una columna con la generación despachada, y lo convertiremos en varias columnas. ¿Y porqué hacer esto? En realidad lo que haremos es tomar nuestra serie temporal y la convertiremos en un «problema de tipo supervisado« para poder alimentar nuestra red neuronal y poder entrenarla con backpropagation («como es habitual»). Para hacerlo, debemos tener unas entradas y unas salidas para entrenar al modelo.

# In[7]:


PASOS=144

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# load dataset
values = df.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(-1, 1))
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.head()


# Usaremos como entradas las columnas encabezadas como var1(t-95) a (t-1) y nuestra salida (lo que sería el valor «Y» de la función) será el var1(t) -la última columna

# Antes de crear la red neuronal, subdividiremos nuestro conjunto de datos en train y en validación. Algo importante de este procedimiento, a diferencia de en otros problemas en los que podemos «mezclar» los datos de entrada, es que en este caso nos importa mantener el orden en el que alimentaremos la red. Por lo tanto, haremos una subdivisión de los primeros 3361 datos consecutivos para entrenamiento de la red y los siguientes 1008 para su validación. Se puede variar esta proporción por ejemplo a 80-20 y comparar resultados )

# In[8]:


# split into train and test sets
values = reframed.values
n_train_days = 4464 - (1008+PASOS)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)


# Hemos transformado la entrada en un arreglo con forma (3361,1,95) esto significa algo así como «3361 entradas con vectores de 1×95

# La arquitectura de la red neuronal será:
# 
# Entrada 95 inputs
# 1 capa oculta con 95 neuronas 
# La salida será 1 sola neurona
# Como función de activación utilizamos tangente hiperbólica puesto que utilizaremos valores entre -1 y 1.
# Utilizaremos como optimizador Adam y métrica de pérdida (loss) Mean Absolute Error
# Como la predicción será un valor continuo y no discreto, para calcular el Accuracy utilizaremos Mean Squared Error y para saber si mejora con el entrenamiento se debería ir reduciendo con las EPOCHS.

# In[9]:


def crear_modeloFF():
    model = Sequential() 
    model.add(Dense(PASOS, input_shape=(1,PASOS),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mean_absolute_error',optimizer='Adam',metrics=["mse"])
    model.summary()
    return model


# En pocos segundos vemos una reducción del valor de pérdida tanto del set de entrenamiento como del de validación.

# In[10]:


EPOCHS=40

model = crear_modeloFF()

history=model.fit(x_train,y_train,epochs=EPOCHS,validation_data=(x_val,y_val),batch_size=PASOS)


# Visualizamos al conjunto de validación (recordemos que eran 1008 datos)

# In[11]:


results=model.predict(x_val)
print( len(results) )
plt.scatter(range(len(y_val)),y_val,c='g')
plt.scatter(range(len(results)),results,c='r')
plt.title('validate')
plt.show()


# Veamos y comparemos también cómo disminuye el LOSS tanto en el conjunto de train como el de Validate, esto es bueno ya que indica que el modelo está aprendiendo. A su vez pareciera no haber overfitting, pues las curvas de train y validate son distintas.

# In[12]:


plt.plot(history.history['loss'])
plt.title('loss')
plt.plot(history.history['val_loss'])
plt.title('validate loss')
plt.show()


# In[13]:


plt.title('Accuracy')
plt.plot(history.history['mean_squared_error'])
plt.show()


# In[14]:


compara = pd.DataFrame(np.array([y_val, [x[0] for x in results]])).transpose()
compara.columns = ['real', 'prediccion']

inverted = scaler.inverse_transform(compara.values)

compara2 = pd.DataFrame(inverted)
compara2.columns = ['real', 'prediccion']
compara2['diferencia'] = compara2['real'] - compara2['prediccion']
compara2.head()


# In[15]:


compara2.describe()


# In[16]:


compara2['real'].plot()
compara2['prediccion'].plot()


# Usaremos los dos últimos días de enero 2017 para calcular la primer día de febrero.

# In[17]:


ultimosDia = df['2017-01-30':'2017-01-31']
ultimosDia


# Y ahora seguiremos el mismo preprocesado de datos que hicimos para el entrenamiento: escalando los valores, llamando a la función series_to_supervised pero esta vez sin incluir la columna de salida «Y» pues es la que queremos hallar. Por eso, verán en el código que hacemos drop() de la última columna.

# In[20]:


values = ultimosDia.values
values = values.astype('float32')
# normalize features
values=values.reshape(-1, 1) # esto lo hacemos porque tenemos 1 sola dimension
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, PASOS, 1)
reframed.drop(reframed.columns[[144]], axis=1, inplace=True)
reframed


# De este conjunto «ultimosDias» tomamos sólo la última fila, y la dejamos en el formato correcto para la red neuronal con reshape:

# In[21]:


values = reframed.values
x_test = values[143:, :]
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(x_test.shape)
x_test


# Ahora crearemos una función para ir «rellenando» el desplazamiento que hacemos por cada predicción. Esto es porque queremos predecir los 144 primeros datos del 1 de febrero . Entonces para las 00H10 del 1 de febrero, ya tenemos el set con los últimos 144 datos del 31 de enero. Pero para pronosticar a las 00H20 del 1 de febrero necesitamos los 144 datos anteriores que INCLUYEN a las 00H10 del 1 de febrero y ese valor, lo obtenemos en nuestra predicción anterior. Y así hasta las 00H00 del 2 de febrero.

# In[22]:


def agregarNuevoValor(x_test,nuevoValor):
    for i in range(x_test.shape[2]-1):
        x_test[0][0][i] = x_test[0][0][i+1]
    x_test[0][0][x_test.shape[2]-1]=nuevoValor
    return x_test


# In[23]:


results=[]
for i in range(144):
    parcial=model.predict(x_test)
    results.append(parcial[0])
    print(x_test)
    x_test=agregarNuevoValor(x_test,parcial[0])


# Ahora las predicciones están en el dominio del -1 al 1 y nosotros lo queremos en nuestra escala «real». Entonces vamos a «re-transformar» los datos con el objeto «scaler» que creamos antes.

# Ya podemos crear un nuevo DataFrame Pandas por si quisiéramos guardar un nuevo csv con el pronóstico. Y lo visualizamos.

# In[25]:


prediccion1 = pd.DataFrame(inverted)
prediccion1.columns = ['pronostico']
prediccion1.plot()
prediccion1.to_csv('pronostico.csv')

