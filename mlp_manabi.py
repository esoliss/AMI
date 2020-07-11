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

url1 = 'https://raw.githubusercontent.com/esoliss/AMI/master/Dataset/CALCETA72881.csv'
url2 = 'https://raw.githubusercontent.com/esoliss/AMI/master/Dataset/CHONE1104639.csv'
url3 = 'https://raw.githubusercontent.com/esoliss/AMI/master/Dataset/CHONE1100232.csv'
df1 = pd.read_csv(url1, error_bad_lines=False)
df1['Date'] = df1['Date'].astype('datetime64[ns]')
df1['Time'] = df1['Time'].astype('datetime64[ns]')
for index, row in df1.iterrows():
    df1['Date'][index]=datetime.datetime(df1['Date'][index].year,df1['Date'][index].month,
                                         df1['Date'][index].day,df1['Time'][index].hour,
                                         df1['Time'][index].minute)
df1 = df1.drop(df1.columns[1],axis=1)
df2 = pd.read_csv(url2, error_bad_lines=False)
df2['Date'] = df2['Date'].astype('datetime64[ns]')
df2['Time'] = df2['Time'].astype('datetime64[ns]')
for index, row in df2.iterrows():
    df2['Date'][index]=datetime.datetime(df2['Date'][index].year,df2['Date'][index].month,
                                         df2['Date'][index].day,df2['Time'][index].hour,
                                         df2['Time'][index].minute)
df2 = df2.drop(df2.columns[1],axis=1)
df3 = pd.read_csv(url3, error_bad_lines=False)
df3['Date'] = df3['Date'].astype('datetime64[ns]')
df3['Time'] = df3['Time'].astype('datetime64[ns]')
for index, row in df3.iterrows():
    df3['Date'][index]=datetime.datetime(df3['Date'][index].year,df3['Date'][index].month,
                                         df3['Date'][index].day,df3['Time'][index].hour,
                                         df3['Time'][index].minute)
df3 = df3.drop(df3.columns[1],axis=1)

var = list(df1.columns)
va = {'Frecuencia':var[1], 'Voltaje L1n':var[2], 'Voltaje L2n': var[3], 'Voltaje L12':var[4],
      'THD V1':var[6], 'THD V2':var[7], 'Flicker L1':var[109], 'Flicker L2':var[110], 
      'Voltaje %':var[111],'Corriente L1':var[112], 'Corriente L2':var[113], 'THD I1':var[114],
      'THD I2':var[115], 'THD In':var[116],'Potencia Activa L1':var[117],'Potencia Activa L2':var[119],
      'Potencia Activa Total':var[121],'Potencia Aparente L1':var[123],'Potencia Aparente L2':var[124],
      'Potencia Aparente Total':var[125],'Factor de potencia L1':var[126],
      'Factor de potencia L2':var[128],'Factor de potencia Total':var[130],'Potencia Reactiva L1':var[132],
      'Potencia Reactiva L2':var[133],'Potencia Reactiva Total':var[134],'Energia L1':var[135],
      'Energia L2':var[136], 'Energia Total':var[138]}

## GRAFICA DE VARIABLES

# SELECCIONAR VARIABLE
select = 'Frecuencia' 
## Opciones:
#'Frecuencia', 'Voltaje L1n', 'Voltaje L2n', 'Voltaje L12',
#'THD V1', 'THD V2', 'Flicker L1', 'Flicker L2', 'Voltaje %',
#'Corriente L1', 'Corriente L2', 'THD I1','THD I2', 'THD In',
#'Potencia Activa L1','Potencia Activa L2', 'Potencia Activa Total',
#'Potencia Aparente L1','Potencia Aparente L2', 'Potencia Aparente Total',
#'Factor de potencia L1','Factor de potencia L2','Factor de potencia Total',
#'Potencia Reactiva L1', 'Potencia Reactiva L2', 'Potencia Reactiva Total',
#'Energia L1', 'Energia L2', 'Energia Total'


# SELECIONAR FECHAS
dia_inicio = 15
dia_final = 22
#OPCIONES
##CALCETA72881: ini 15 - 11:30 fin 22 - 11:20
##CHONE1104639: ini 20 - 11:50 fin 27 - 11:40
##CHONE1100232: ini 20 - 14:50 fin 27 - 14:40

# Mayo de 2020 3 datasets de un mes
mes_inicio = 5
mes_final = 5

dfil = df1[['Date',va[select]]].copy()
ini=datetime.datetime(2020,5,dia_inicio,0,0)
fin=datetime.datetime(2020,5,dia_final,0,0)
mask = (dfil["Date"]>=ini) & (dfil["Date"]<fin)
seq = dfil.loc[mask] 

## PERCEPTRON CALCETA72881
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
medi_i=df1[df1['Date']==datetime.datetime(2020,mes_final,dia_final,0,0)].index.values
medi = df1[va[select]][medi_i[0]]
#Plot results
#Histograma
plt.hist(seq[va[select]],bins=20)
plt.title('Histograma: '+select+' CALCETA72881')
plt.show()
#Curva y prediccion
#plt.pyplot.set_xlim(left=datetime.datetime(2020,mes_final,dia_final,0,0)-200*datetime.timedelta(minutes=10))
sns.lineplot(x='Date', y=va[select], data=seq)
plt.xticks(rotation=15)
plt.title(select + ' CALCETA72881')
plt.scatter(datetime.datetime(2020,mes_final,dia_final,0,0),yhat,color='red',label='Predicción = '+str(yhat[0][0]))
plt.scatter(datetime.datetime(2020,mes_final,dia_final,0,0),medi,color='green',label='Medición  = '+str(np.round(medi,4)))
plt.legend()
plt.show()



## PERCEPTRON CHONE1104639
# SELECCIONAR VARIABLE
select = 'Energia Total' 
## Opciones:
#'Frecuencia', 'Voltaje L1n', 'Voltaje L2n', 'Voltaje L12',
#'THD V1', 'THD V2', 'Flicker L1', 'Flicker L2', 'Voltaje %',
#'Corriente L1', 'Corriente L2', 'THD I1','THD I2', 'THD In',
#'Potencia Activa L1','Potencia Activa L2', 'Potencia Activa Total',
#'Potencia Aparente L1','Potencia Aparente L2', 'Potencia Aparente Total',
#'Factor de potencia L1','Factor de potencia L2','Factor de potencia Total',
#'Potencia Reactiva L1', 'Potencia Reactiva L2', 'Potencia Reactiva Total',
#'Energia L1', 'Energia L2', 'Energia Total'



# SELECIONAR FECHAS
dia_inicio = 20
dia_final = 26
#OPCIONES
##CALCETA72881: ini 15 - 11:30 fin 22 - 11:20
##CHONE1104639: ini 20 - 11:50 fin 27 - 11:40
##CHONE1100232: ini 20 - 14:50 fin 27 - 14:40

# Mayo de 2020 3 datasets de un mes
mes_inicio = 5
mes_final = 5

dfil = df2[['Date',va[select]]].copy()
ini=datetime.datetime(2020,5,dia_inicio,0,0)
fin=datetime.datetime(2020,5,dia_final,0,0)
mask = (dfil["Date"]>=ini) & (dfil["Date"]<fin)
seq = dfil.loc[mask] 
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
n_steps_in, n_steps_out = 90, 90
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
raw_seq = seq[va[select]].to_list()
testset = raw_seq[-n_steps_in:-1]
testset.append(raw_seq[-1])
x_input = np.array(testset)
x_input = x_input.reshape((1, n_steps_in))
yhat = model.predict(x_input, verbose=0)
print(yhat)

dfpl = df2[['Date',va[select]]].copy()
ini=datetime.datetime(2020,mes_inicio,dia_inicio,0,0)
fin=datetime.datetime(2020,mes_final,dia_final,0,0)+datetime.timedelta(minutes=n_steps_out*10)
maskp = (dfpl["Date"]>=ini) & (dfpl["Date"]<fin)
seqp = dfpl.loc[maskp] 
#Curva y prediccion
#plt.pyplot.set_xlim(left=datetime.datetime(2020,mes_final,dia_final,0,0)-200*datetime.timedelta(minutes=10))
sns.lineplot(x='Date', y=va[select], data=seqp,label='Medición')
plt.xticks(rotation=15)
plt.title(select+'MLP multi step forecast'+' CHONE1104639')
dres =[]
rres =[]
for t in range(n_steps_out):
    dres.append(datetime.datetime(2020,mes_final,dia_final,0,0)+datetime.timedelta(minutes=10*t))
    rres.append(yhat[0][t])
sns.lineplot(dres,rres,color='red',label='Predicción')
plt.legend()
plt.show()

## PERCEPTRON CHONE1100232
# SELECCIONAR VARIABLE
select1 = 'Voltaje L1n' 
select2 = 'Voltaje L2n' 
select3 = 'Voltaje L12' 
## Opciones:
#'Frecuencia', 'Voltaje L1n', 'Voltaje L2n', 'Voltaje L12',
#'THD V1', 'THD V2', 'Flicker L1', 'Flicker L2', 'Voltaje %',
#'Corriente L1', 'Corriente L2', 'THD I1','THD I2', 'THD In',
#'Potencia Activa L1','Potencia Activa L2', 'Potencia Activa Total',
#'Potencia Aparente L1','Potencia Aparente L2', 'Potencia Aparente Total',
#'Factor de potencia L1','Factor de potencia L2','Factor de potencia Total',
#'Potencia Reactiva L1', 'Potencia Reactiva L2', 'Potencia Reactiva Total',
#'Energia L1', 'Energia L2', 'Energia Total'


# SELECIONAR FECHAS
dia_inicio = 20
dia_final = 26
#OPCIONES
##CALCETA72881: ini 15 - 11:30 fin 22 - 11:20
##CHONE1104639: ini 20 - 11:50 fin 27 - 11:40
##CHONE1100232: ini 20 - 14:50 fin 27 - 14:40

# Mayo de 2020 3 datasets de un mes
mes_inicio = 5
mes_final = 5

dfil1 = df3[['Date',va[select1]]].copy()
ini=datetime.datetime(2020,5,dia_inicio,0,0)
fin=datetime.datetime(2020,5,dia_final,0,0)
mask = (dfil1["Date"]>=ini) & (dfil1["Date"]<fin)
seq1 = dfil1.loc[mask]

dfil2 = df3[['Date',va[select2]]].copy()
ini=datetime.datetime(2020,5,dia_inicio,0,0)
fin=datetime.datetime(2020,5,dia_final,0,0)
mask = (dfil2["Date"]>=ini) & (dfil2["Date"]<fin)
seq2 = dfil2.loc[mask]

dfil3 = df3[['Date',va[select3]]].copy()
ini=datetime.datetime(2020,5,dia_inicio,0,0)
fin=datetime.datetime(2020,5,dia_final,0,0)
mask = (dfil3["Date"]>=ini) & (dfil3["Date"]<fin)
seq3 = dfil3.loc[mask]

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y) 


# define input sequence
raw_seq1 = seq1[va[select1]].to_list()
raw_seq2 = seq2[va[select2]].to_list()
raw_seq3 = seq3[va[select3]].to_list()

in_seq1 = np.array(seq1[va[select1]].to_list())
in_seq2 = np.array(seq2[va[select2]].to_list())
out_seq = np.array(seq3[va[select3]].to_list())
# convert to [rows, columns] structure
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

# horizontally stack columns
dataset = np.hstack((in_seq1, in_seq2, out_seq))
# choose a number of time steps
n_steps_in, n_steps_out = 200, 200

# convert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
# flatten input
n_input = X.shape[1] * X.shape[2]
X = X.reshape((X.shape[0], n_input))
# flatten output
n_output = y.shape[1] * y.shape[2]
y = y.reshape((y.shape[0], n_output))
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_input))
model.add(Dense(n_output))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
testset1 = raw_seq1[-n_steps_in:-1]
testset1.append(raw_seq1[-1])
testset2 = raw_seq2[-n_steps_in:-1]
testset2.append(raw_seq2[-1])
testset3 = raw_seq3[-n_steps_in:-1]
testset3.append(raw_seq3[-1])
x_input = np.array([np.array(testset1),np.array(testset2) ,np.array(testset3)])
x_input = x_input.reshape((1, n_input))
yhat = model.predict(x_input, verbose=0)
print(yhat)
#Curva y prediccion
#plt.pyplot.set_xlim(left=datetime.datetime(2020,mes_final,dia_final,0,0)-200*datetime.timedelta(minutes=10))
sns.lineplot(x='Date', y=va[select1], data=seq1,label=str(select1))
sns.lineplot(x='Date', y=va[select2], data=seq2,label=str(select2))
sns.lineplot(x='Date', y=va[select3], data=seq3,label=str(select3))
plt.xticks(rotation=15)
plt.title(select3+'MLP multi step forecast'+' CHONE1100232')
dres =[]
rres =[]
rres2 =[]
rres3 =[]
m=0
for t in range(n_steps_out):
    dres.append(datetime.datetime(2020,mes_final,dia_final,0,0)+datetime.timedelta(minutes=10*t))
    rres.append(yhat[0][m])
    rres2.append(yhat[0][1+m])
    rres3.append(yhat[0][2+m])
    m+=3
sns.lineplot(dres,rres,label='Predicción '+str(select1))
sns.lineplot(dres,rres2,label='Predicción '+str(select2))
sns.lineplot(dres,rres3,color='red',label='Predicción '+str(select3))
plt.legend()
plt.show()