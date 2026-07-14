# --------------- Importaciones

import pandas as pd
from sklearn.metrics import mean_absolute_error

# --------------- Fin de las Importaciones

#para tripleten
# train = pd.read_csv("/datasets/gold_recovery_train.csv")
# test = pd.read_csv("/datasets/gold_recovery_test.csv")
# full = pd.read_csv("/datasets/gold_recovery_full.csv")

# 1.1. Abre los archivos y examina los datos.

train = pd.read_csv(r"C:\Users\sasor\Desktop\Tripleten\Sprint 13\Proyecto\gold_recovery_train.csv")
test = pd.read_csv(r"C:\Users\sasor\Desktop\Tripleten\Sprint 13\Proyecto\gold_recovery_test.csv")
full = pd.read_csv(r"C:\Users\sasor\Desktop\Tripleten\Sprint 13\Proyecto\gold_recovery_full.csv")

print(train.shape)
print(train.info())
print(train.dtypes)
print(train.isnull().sum())

print(test.shape)
print(test.info())
print(test.dtypes)
print(test.isnull().sum())

print(full.shape)
print(full.info())
print(full.dtypes)
print(full.isnull().sum())

"""se pasa el formato de la columna date al correcto datetime"""
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
full["date"] = pd.to_datetime(full["date"])

# 1.2. Comprueba que el cálculo de la recuperación sea correcto. Calcula la recuperación de la característica rougher.output.recovery mediante el conjunto de entrenamiento. Encuentra el EAM entre tus cálculos y los valores de la característica. Facilita los resultados.

# formula: recovery = c * (f - t) / (f * (c - t)) * 100
# c = oro en el concentrado rougher
# f = oro en la alimentacion
# t = oro en las colas rougher

c = train["rougher.output.concentrate_au"]
f = train["rougher.input.feed_au"]
t = train["rougher.output.tail_au"]

recovery_calculado = c * (f - t) / (f * (c - t)) * 100

# comparo solo las filas donde hay valor real y valor calculado, si no el eam no se puede calcular
comparacion = pd.DataFrame({"calculado": recovery_calculado, "real": train["rougher.output.recovery"]}).dropna()
 
eam = mean_absolute_error(comparacion["real"], comparacion["calculado"])
print("eam entre el recovery calculado y el real:", eam)
