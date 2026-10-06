# Ejercicio 2: clasificación de dígitos

Ejecutar desde la raíz del proyecto:

```powershell
uv run python -m ej2_digitos.run
```

El estudio usa el perceptrón multicapa compartido en `models/neural_network.py`.
Procesa imágenes de 28 × 28 píxeles con activación tanh, objetivos codificados
como −1/1 y clasificación mediante el índice de la salida más alta. Entrena
por lotes con operaciones matriciales de NumPy; `batch_size=1` actualiza por
muestra, `128` por mini batch y `-1` una vez por época con todos los ejemplos.

## Selección y evaluación

`digits.csv` se divide de forma estratificada en 80% para entrenamiento y 20%
para validación. Con mini batches de 128 se comparan dos arquitecturas
(`[784, 64, 10]` y `[784, 128, 64, 10]`), dos tasas (`0,01` y `0,05`) y dos
optimizadores (descenso de gradiente sin momentum y con momentum `0,9`). Son
ocho configuraciones. Se elige la de mayor accuracy de validación; el MSE
resuelve empates.

Luego se mantienen fijos esos hiperparámetros y se comparan los tres modos de
actualización con la **misma cantidad de épocas**. Esto aísla el efecto del
tamaño de lote, aunque los modos realizan distintos números de actualizaciones
por época y la tasa elegida para mini batch puede no ser ideal para los otros.
La modalidad se selecciona otra vez por accuracy de validación y MSE. Se
reentrena desde cero con todo `digits.csv` y recién entonces se abre
`digits_test.csv`, que no participa en la selección.

Para cada configuración se registran accuracy y MSE de entrenamiento y
validación, así como tiempo de entrenamiento. En test se reportan accuracy,
macro F1, métricas por dígito y matriz de confusión. Los archivos generados son
`results/ej2_digitos/comparison.csv`, `results.json` y `digit_model.model`.
El modelo se recupera con `MultiLayerPerceptron.load(ruta)`.

Se pueden ajustar `--epochs`, `--mini-batch-size`, `--seed`,
`--validation-fraction`, `--train`, `--test` y `--output-dir`.

## Resultado con la configuración predeterminada

Con semilla 2 y 12 épocas, la mejor configuración de mini batch fue
`[784, 128, 64, 10]`, tasa `0,01` y momentum `0,9`, con **94,94%** de accuracy
de validación.

| Actualización | Tamaño de lote | Accuracy de validación | Tiempo de entrenamiento* |
| --- | ---: | ---: | ---: |
| Online | 1 | 12,45% | 75,2 s |
| Mini batch | 128 | 94,94% | 2,3 s |
| Batch completo | 9.960 | 38,65% | 2,7 s |

*Medido en esta máquina; puede variar en otros equipos. Son resultados de la
misma tasa fija y 12 épocas, no una comparación de la mejor configuración
posible de cada modalidad. Tras reentrenar el mini batch con las 12.449
imágenes de aprendizaje, la accuracy de test fue **84,18%** y el macro F1
**0,7944**.

Hay una limitación importante de los datos: `digits.csv` no contiene ningún
ejemplo del dígito 8 y apenas 271 del 5, mientras que `digits_test.csv` contiene
243 ejemplos del 8. El recall del 8 fue **0%** y el del 5 **72,65%**. La
diferencia entre validación y test se debe en buena parte a esa distribución;
el ejercicio 3 permite estudiar el efecto de `more_digits.csv` sin usar test
para entrenar.
