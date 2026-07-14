# --------------- Importaciones

import pandas as pd
from matplotlib import pyplot as plt
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
print("eam entre el recovery calculado y el real:", round(eam,5))

"""como pequeño analisis, la comparacion nos da resultados identicos por lo que la diferencia es pracitcamente 0."""

# 1.3. Analiza las características no disponibles en el conjunto de prueba. ¿Cuáles son estos parámetros? ¿Cuál es su tipo?

columnas_faltantes = sorted(set(train.columns) - set(test.columns))
print("cantidad de columnas que no estan en test:", len(columnas_faltantes))
print(columnas_faltantes)

"""aquellas caracteristicas no disponibles en el conjunto de prueba son aquellos que tienen tipo .calculation o .output. valores que se saben despues, no al momento de iniciar este proceso"""

# 1.4. Realiza el preprocesamiento de datos.

"""se dividen los df por fecha, se busca rellenar los nulos usando los valores ordenados de manera cronologica"""
train = train.sort_values("date")
test = test.sort_values("date")
full = full.sort_values("date")

"""se necesita quitar las filas con na de mis target ya que no sirven para entrenar, posteriormente se rellenan las columnas de features, se decide en agarrar el ultimo valor valido porque si lo hago asi entonces el dato que se rellene tendra sentido cronologico, a diferencia de usar un promedio, mediana, etc."""
train = train.dropna(subset=["rougher.output.recovery", "final.output.recovery"])
train = train.ffill().bfill()
test = test.ffill().bfill()

"""verificacion rapida de nulos"""
print("train:", train.isna().sum().sum())
print("test:", test.isna().sum().sum())

# 2. Analiza los datos

# 2.1. Observa cómo cambia la concentración de metales (Au, Ag, Pb) en función de la etapa de purificación.

etapas = ["rougher.input.feed", "rougher.output.concentrate", "primary_cleaner.output.concentrate", "final.output.concentrate"]
metales = ["au", "ag", "pb"]

for metal in metales:
    medias = [train[f"{etapa}_{metal}"].mean() for etapa in etapas]
    plt.plot(etapas, medias, marker="o", label=metal)

plt.xticks(rotation=45, ha="right")
plt.title("concentracion promedio por etapa")
plt.legend()
plt.tight_layout()
plt.show()

#lo que se ve es que el oro (au) va subiendo de concentracion en cada etapa de purificacion, mientras que la plata (ag) va bajando. el plomo (pb) se mantiene mas o menos estable con una leve subida. tiene sentido porque el proceso esta pensado justamente para ir concentrando el oro