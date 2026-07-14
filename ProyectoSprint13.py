# --------------- Importaciones

import pandas as pd

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