# --------------- Importaciones

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, make_scorer
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

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

"""el au sube de concentracion por etapa de purificacion. el ag baja. el pb sube. el oro es el que sube mas de todos"""

# 2.2. Compara las distribuciones del tamaño de las partículas de la alimentación en el conjunto de entrenamiento y en el conjunto de prueba. Si las distribuciones varían significativamente, la evaluación del modelo no será correcta.

plt.figure()
plt.hist(train["rougher.input.feed_size"], bins=50, alpha=0.5, label="train")
plt.hist(test["rougher.input.feed_size"], bins=50, alpha=0.5, label="test")
plt.title("distribucion de rougher.input.feed_size")
plt.legend()
plt.show()

"""se redondea a 4 decimales para facilitar la lectura"""
print("mean feed_size train:", round(train["rougher.input.feed_size"].mean(),4))
print("mean feed_size test:", round(test["rougher.input.feed_size"].mean(),4))

"""distribuciones similares entre test y train con diferencia del volumen"""

# 2.3. Considera las concentraciones totales de todas las sustancias en las diferentes etapas: materia prima, concentrado rougher y concentrado final. ¿Observas algún valor anormal en la distribución total? Si es así, ¿merece la pena eliminar esos valores de ambas muestras? Describe los resultados y elimina las anomalías.

"""separacion enlistada con las columnas de interes para que se sumen en cada una de las etapas"""
feed_cols = ["rougher.input.feed_au", "rougher.input.feed_ag",
             "rougher.input.feed_pb", "rougher.input.feed_sol"]
rougher_cols = ["rougher.output.concentrate_au", "rougher.output.concentrate_ag",
                "rougher.output.concentrate_pb", "rougher.output.concentrate_sol"]
final_cols = ["final.output.concentrate_au", "final.output.concentrate_ag",
              "final.output.concentrate_pb", "final.output.concentrate_sol"]

"""se especifica el axis=1 para que no se sume columna por columna ya que se buscan las concentraciones totales de las sutancias"""
train_feed_total = train[feed_cols].sum(axis=1)
train_rougher_total = train[rougher_cols].sum(axis=1)
train_final_total = train[final_cols].sum(axis=1)

# histograma - alimentacion rougher
plt.figure()
plt.hist(train_feed_total, bins=50)
plt.title("concentracion alimentacion rougher")
plt.show()

# histograma - concentrado rougher
plt.figure()
plt.hist(train_rougher_total, bins=50)
plt.title("concentracion concentrado rougher")
plt.show()

#histogra,a - concentrado final
plt.figure()
plt.hist(train_final_total, bins=50)
plt.title("concentracion concentrado final")
plt.show()

"""cuando la concentracion es cercana o igual a 0, se descartan, esta situacion se considera comi(anomalias) por lo tanto se tienen que descartar"""

#reviso en antes_train la estructura previa para comparar despues
antes_train = train.shape[0]
train = train[(train_feed_total > 10) & (train_rougher_total > 10) & (train_final_total > 10)]
#impresion para verificar ña cantidad de filas perdidas
print("filas de train antes:", antes_train, "despues de anomalias:", train.shape[0])

test_feed_total = test[feed_cols].sum(axis=1)
antes_test = test.shape[0]
test = test[test_feed_total > 10]
print("filas de test antes:", antes_test, "despues anomalias:", test.shape[0])

# features y targets

target_test = full[full["date"].isin(test["date"])][
    ["date", "rougher.output.recovery", "final.output.recovery"]
]

test = test.merge(target_test, on="date", how="left")
test = test.dropna(subset=["rougher.output.recovery", "final.output.recovery"])

#lista de lo que ira en el modelo
feature_cols = []
for columna in test.columns:
    if columna not in ["date", "rougher.output.recovery", "final.output.recovery"]:
        feature_cols.append(columna)

features_train = train[feature_cols]
target_train_rougher = train["rougher.output.recovery"]
target_train_final = train["final.output.recovery"]

features_test = test[feature_cols]
target_test_rougher = test["rougher.output.recovery"]
target_test_final = test["final.output.recovery"]

"""esto se necesita para la parte 3 pero se me olvido hacerla antes junto al paso 2.3"""

# 3. Construye el modelo

# 3.1. Escribe una función para calcular el valor final de sMAPE.

"""formulas proporcionadas por tp para el proyecto"""
def smape(target, predicciones):
    target = np.array(target)
    predicciones = np.array(predicciones)
    return np.mean(np.abs(target - predicciones) / ((np.abs(target) + np.abs(predicciones)) / 2)) * 100

def smape_final(target_rougher, pred_rougher, target_final, pred_final):
    smape_rougher = smape(target_rougher, pred_rougher)
    smape_f = smape(target_final, pred_final)
    return 0.25 * smape_rougher + 0.75 * smape_f

smape_scorer = make_scorer(smape, greater_is_better=False)

# 3.2. Entrena diferentes modelos. Evalúalos aplicando la validación cruzada. Elige el mejor modelo y pruébalo utilizando la muestra de prueba. Facilita los resultados.

"""se eligieron esa n_estimators=50 y max_depth=8, ya que se considera suficiente, pero principalemnte porque si subo ambos, el entrenamiento dura demasiado y no mejora tanto la calidad del resultado, el cambio es mas drastico para max_depth cuando aumenta"""
modelos = {
    "regresion lineal": LinearRegression(),
    "arbol de decision": DecisionTreeRegressor(random_state=12345, max_depth=8),
    "bosque aleatorio": RandomForestRegressor(random_state=12345, n_estimators=50, max_depth=8),
}

resultados = {}
for nombre, modelo in modelos.items():
    smape_rougher_cv = -cross_val_score(
        modelo, features_train, target_train_rougher, cv=5, scoring=smape_scorer).mean()
    smape_final_cv = -cross_val_score(modelo, features_train, target_train_final, cv=5, scoring=smape_scorer).mean()
    smape_total_cv = 0.25 * smape_rougher_cv + 0.75 * smape_final_cv
    resultados[nombre] = smape_total_cv
    print(nombre)
    print("smape rougher (cv):", smape_rougher_cv)
    print("smape final (cv):", smape_final_cv)
    print("smape total (cv):", smape_total_cv)
    print()

mejor_nombre = min(resultados, key=resultados.get)
print("mejor modelo de los 3s:", mejor_nombre)

#entrenamiento del mejor modelo con todo el train y lo pruebo con el set de prueba
modelo_rougher = modelos[mejor_nombre]
modelo_final = modelos[mejor_nombre]

modelo_rougher.fit(features_train, target_train_rougher)
modelo_final.fit(features_train, target_train_final)

pred_test_rougher = modelo_rougher.predict(features_test)
pred_test_final = modelo_final.predict(features_test)

smape_prueba = smape_final(target_test_rougher, pred_test_rougher, target_test_final, pred_test_final)
print("smape final - set de prueba:", smape_prueba)

"""en resumen: el bosque aleatorio fue el mejor modelo, tiene los resultados mas bajos a comparacion de los demas, siendo el arbol de decision con el smape mas alto de todos los modelos. el bosque aleatorio es el mejor modelo entonces, para predecir la recuperacion del au"""
