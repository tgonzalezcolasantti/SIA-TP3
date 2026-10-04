# Ejercicio 1: destilación de probabilidades de fraude

## Reproducir el estudio

Desde la raíz del proyecto:

```powershell
python -m ej1_fraude.run
```

El comando lee `data/fraud_dataset.csv`. Se puede indicar otro CSV con
`--dataset`.
Guarda el análisis, dos gráficos y `tiny_model.npz` en `results/ej1_fraude/`.
La semilla es 2 y se puede cambiar con `--seed`. Los resultados siguientes
corresponden a esa configuración y al CSV cuyo SHA-256 consta en `results.json`.

## Exploración del conjunto de datos

El CSV tiene 7.500 transacciones, nueve variables de entrada y ninguna celda
vacía ni fila duplicada. La documentación incluida en el ZIP define
`big_model_fraud_probability` como la salida de BigModel y advierte que
`flagged_fraud` **no debe utilizarse para entrenar**. Se usa únicamente para
evaluar decisiones y elegir el umbral. Hay 869 fraudes confirmados (11,59%).

| Variable de entrada | Significado | Mínimo | Máximo |
| --- | --- | ---: | ---: |
| `timestamp` | Momento de compra, Unix segundos | 1.700.001.808 | 1.731.534.222 |
| `amount_usd` | Importe en USD | 1 | 2.000 |
| `quantity_purchased` | Unidades del artículo | 1 | 24 |
| `session_duration_seconds` | Duración de la sesión | 5 | 726,8 |
| `days_since_last_purchase` | Días desde la compra anterior | 0 | 142,32 |
| `account_age_days` | Antigüedad de la cuenta | 1 | 3.649 |
| `device_screen_resolution` | Ancho × alto de pantalla, píxeles | 1.006.733 | 8.310.940 |
| `time_since_last_login_s` | Segundos desde el último ingreso | 10 | 40.160,8 |
| `items_viewed_before_purchase` | Artículos vistos antes de comprar | 1 | 29 |

La probabilidad de BigModel va de 0,000897 a 1, con mediana 0,3573. Antes de
entrenar, cada variable se centra y divide por su desvío estándar. En cada fold
esos valores se calculan **solo con las muestras de entrenamiento**.

## Comparación del aprendizaje

Se usaron las 7.500 muestras, tasa de aprendizaje 0,001, 100 épocas, semilla 2
y el mismo intervalo inicial de pesos `[-0,05, 0,05]`. El no lineal usa
activación logística con β = 1. Se minimiza el error cuadrático respecto de la
probabilidad de BigModel.

| Modelo | MSE inicial | MSE final | MAE final | Predicciones fuera de [0, 1] |
| --- | ---: | ---: | ---: | ---: |
| Lineal | 0,267982 | 0,026160 | 0,129371 | 506 |
| Logístico | 0,092067 | 0,010869 | 0,076147 | 0 |

**Underfitting:** el lineal reduce su error, pero se estabiliza muy por encima
del logístico. Además, 506 predicciones no son probabilidades válidas. El
logístico es el candidato más adecuado para continuar.

**Saturación:** el error lineal deja de mejorar después de las primeras épocas.
El logístico mejora hasta aproximadamente la época 50 y luego se estabiliza en
MSE 0,010869. Esto muestra un límite práctico de estas configuraciones; una
curva plana por sí sola no demuestra la capacidad máxima teórica del modelo.

Ver `results/ej1_fraude/learning_curves.png`.

## Generalización y selección de modelo

Se evaluó el logístico con validación cruzada de cinco folds aleatorios. Cada
muestra se predijo exactamente una vez fuera de entrenamiento; cada fold usa
6.000 transacciones para ajustar pesos y 1.500 para validar. Elegimos MSE y MAE
porque el objetivo principal es aproximar la **probabilidad** de BigModel.

| Fold | Fraudes en validación | MSE |
| --- | ---: | ---: |
| 1 | 178 | 0,010726 |
| 2 | 173 | 0,010914 |
| 3 | 186 | 0,009967 |
| 4 | 167 | 0,011774 |
| 5 | 165 | 0,011181 |

El MSE fuera de muestra fue **0,010912 ± 0,000660** entre folds y el MAE
conjunto fue **0,076293**. La cercanía con el MSE de entrenamiento no muestra
un sobreajuste importante en este estudio. No se elige el fold con mejor
resultado para presentarlo como el modelo final: se evalúa el comportamiento
entre particiones y luego se reentrena el logístico con las 7.500 muestras.

## Umbral recomendado

Para convertir la probabilidad estimada en una alerta, se eligió el umbral que
maximiza **F2** sobre predicciones fuera de muestra. F2 prioriza recall sobre
precision, porque perder un fraude puede ser más costoso que revisar una alerta
adicional. Sin costos reales de ambos errores, esto es un criterio explícito de
trabajo, no un óptimo económico demostrado.

| Umbral | Precision | Recall | F2 | Falsos positivos | Fraudes omitidos |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0,5 | 33,27% | 100,00% | 0,714 | 1.743 | 0 |
| **0,8013** | **75,36%** | **95,40%** | **0,906** | **271** | **40** |

Como control de generalización del **procedimiento de selección del umbral**,
en cada fold se eligió un umbral con datos internos de entrenamiento y se midió
en el fold externo. El resultado conjunto fue precision **73,92%**, recall
**94,94%**, F2 **0,898**, 291 falsos positivos y 44 fraudes omitidos. Esta es
una estimación más prudente que medir el umbral 0,8013 en los mismos datos que
sirvieron para escogerlo.

La recomendación para CompanyX es un TinyModel logístico con nueve entradas,
**10 parámetros aprendidos** (nueve pesos y un bias), normalización guardada y
umbral **0,8013**. `results/ej1_fraude/tiny_model.npz` contiene todo lo
necesario para inferir; `ej1_fraude.predict.predict_rows` lo carga y devuelve
probabilidades y alertas. No conocemos el tamaño ni el costo de BigModel, así
que no podemos cuantificar el ahorro frente a él. Tampoco se evaluó la
calibración de TinyModel respecto del fraude real ni se dispone de un conjunto
externo de producción.
