#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 03:08:23 2020

@author: edwinsolis
"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense

url = 'https://raw.githubusercontent.com/esoliss/AMI/master/Dataset/belohorizonte2017.csv'
df = pd.read_csv(url, error_bad_lines=False)
df = df.drop(columns=['Timestamp','kVARh rec int','kWh rec int','kVAR sd rec'])
df = df.drop(df.columns[-1],axis=1)
df = df.rename(columns={"kVAR sd del": "Q", "kVARh del int": "kVARh","kW sd del": "P","kWh del int": "kWh","kVA sd del": "S"})
df['Local Time'] = df['Local Time'].astype('datetime64[ns]')
var = list(df.columns)
va = {'Potencia Activa':var[3], 'Potencia Reactiva':var[1], 'Potencia Aparente': var[8], 'Voltaje An':var[5],
      'Voltaje Bn':var[6], 'Voltaje Cn':var[7], 'Corriente A':var[9], 'Corriente B':var[10], 'Corriente C':var[11],
      'Energía Activa':var[4], 'Energía Reactiva':var[3]}

# SELECCIONAR VARIABLE
select = 'Potencia Activa' 
## Opciones:
#'Potencia Activa', 'Potencia Reactiva', 'Potencia Aparente', 'Voltaje An', 'Voltaje bn', 'Voltaje Cn'
#'Corriente A', 'Corriente B', 'Corriente C', 'Energía Activa', 'Energía Reactiva'

# SELECIONAR FECHAS
mes_inicio = 1
dia_inicio = 2
mes_final = 1
dia_final = 10

dfil = df[['Local Time',va[select]]].copy()
ini=datetime.datetime(2017,mes_inicio,dia_inicio,0,0)
if mes_inicio ==12:
    yf = 2018
else:
    yf =2017
fin=datetime.datetime(yf,mes_final,dia_final,0,0)
mask = (dfil["Local Time"]>=ini) & (dfil["Local Time"]<fin)
seq = dfil.loc[mask] 

## Multi Layer Perceptron univariable one step
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
n_steps = 4
raw_seq = seq[va[select]].to_list()
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = np.array([raw_seq[-4], raw_seq[-3], raw_seq[-2], raw_seq[-1]])
x_input = x_input.reshape((1, n_steps))
yhat = model.predict(x_input, verbose=0)
print(yhat)
medi_i=df[df['Local Time']==datetime.datetime(2017,mes_final,dia_final,0,0)].index.values
medi = df[va[select]][medi_i[0]]
#Plot results
#Histograma
plt.hist(seq[va[select]],bins=20)
plt.title('Histograma: '+select)
plt.show()
#Curva y prediccion
sns.lineplot(x='Local Time', y=va[select], data=seq)
plt.xticks(rotation=15)
plt.title(select)
plt.scatter(datetime.datetime(2017,mes_final,dia_final,0,0),yhat,color='red',label='Predicción = '+str(yhat[0][0]))
plt.scatter(datetime.datetime(2017,mes_final,dia_final,0,0),medi,color='green',label='Medición  = '+str(np.round(medi,4)))
plt.legend()
plt.show()

## Multi Layer Perceptron univariable multistep
# split a univariate sequence into samples
def split_sequencemo(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# define input sequence
#same raw sequence
# choose a number of time steps
n_steps_in, n_steps_out = 30, 30
# split into samples
X, y = split_sequencemo(raw_seq, n_steps_in, n_steps_out)
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps_in))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
testset = raw_seq[-n_steps_in:-1]
testset.append(raw_seq[-1])
x_input = np.array(testset)
x_input = x_input.reshape((1, n_steps_in))
yhat = model.predict(x_input, verbose=0)
print(yhat)

dfpl = df[['Local Time',va[select]]].copy()
ini=datetime.datetime(2017,mes_inicio,dia_inicio,0,0)
if mes_inicio ==12:
    yf = 2018
else:
    yf =2017
fin=datetime.datetime(yf,mes_final,dia_final,0,0)+datetime.timedelta(minutes=n_steps_out*15)
maskp = (dfpl["Local Time"]>=ini) & (dfpl["Local Time"]<fin)
seqp = dfpl.loc[maskp] 
#Curva y prediccion
sns.lineplot(x='Local Time', y=va[select], data=seqp,label='Medición')
plt.xticks(rotation=15)
plt.title(select+'MLP multi step forecast')
dres =[]
rres =[]
for t in range(n_steps_out):
    dres.append(datetime.datetime(2017,mes_final,dia_final,0,0)+datetime.timedelta(minutes=15*t))
    rres.append(yhat[0][t])
sns.lineplot(dres,rres,color='red',label='Predicción')
plt.legend()
plt.show()