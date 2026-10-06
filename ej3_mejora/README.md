# Ejercicio 3: mejora del clasificador de dígitos

Ejecutar desde la raíz del proyecto:

```powershell
uv run python -m ej3_mejora.run
```

El script utiliza el mismo perceptrón multicapa del ejercicio 2, con activación
Tanh, objetivos −1/1, error cuadrático medio y momentum 0,9. Guarda la búsqueda
en `results/ej3_mejora/comparison.csv`, el análisis completo en `results.json`,
el modelo final en `digit_model.model` y dos gráficos:
`validation_comparison.png` y `test_confusion.png`. Estos archivos generados se ignoran en
Git; se regeneran con el comando anterior. Se pueden ajustar las rutas y los
parámetros con `--old`, `--new`, `--test`, `--output-dir`, `--seed`, `--epochs`,
`--batch-size`, `--balance-count` y `--validation-fraction`.

## Datos y evaluación

`digits.csv` contiene 12.449 imágenes y `more_digits.csv` contiene 15.741.
Comparten 3.689 imágenes idénticas, por lo que su unión tiene **24.501 ejemplos
únicos**. Se eliminan los duplicados *antes* de dividir los datos para evitar que
una misma imagen quede en entrenamiento y validación. En la unión hay 585 ochos
y 785 cincos; en `digits.csv` había cero ochos y 271 cincos.

Se aparta el 20% de la unión mediante una división estratificada con semilla 2.
Todas las configuraciones se comparan en esa misma validación. La variante
«solo datos nuevos» entrena únicamente con las imágenes de `more_digits.csv` que
quedaron del lado de entrenamiento. Se selecciona por accuracy de validación,
desempatando con MSE. Después se crea otra red desde cero, se entrena con todas
las imágenes elegibles de aprendizaje y se evalúa en `digits_test.csv`.

El rebalanceo repite con reemplazo imágenes de las clases 5 y 8 hasta contar con
2.000 presentaciones de cada una en el entrenamiento. La ampliación de datos
añade una o dos copias por imagen, desplazadas un píxel horizontal y/o
verticalmente. Ambas operaciones se aplican **después de dividir** los datos,
solo al entrenamiento; validación y test conservan sus imágenes originales.

## Variantes y resultado

Todas las filas usan 40 épocas, mini batches de 128 y momentum 0,9. `Balance`
indica el objetivo de presentaciones por clase minoritaria; `Copias` indica
cuántas imágenes desplazadas se añaden por cada presentación original.

| Datos | Capas | Tasa | Balance | Copias | Accuracy de validación |
| --- | --- | ---: | ---: | ---: | ---: |
| Nuevos | 784–128–64–10 | 0,01 | 0 | 0 | 96,00% |
| Unión | 784–128–64–10 | 0,01 | 0 | 0 | 96,59% |
| Unión | 784–128–64–10 | 0,01 | 2.000 | 0 | 96,61% |
| Unión | 784–256–128–10 | 0,01 | 0 | 0 | 96,84% |
| Unión | 784–256–128–10 | 0,01 | 2.000 | 0 | 97,02% |
| Unión | 784–256–128–10 | 0,02 | 2.000 | 0 | 97,43% |
| Unión | 784–256–128–10 | 0,02 | 2.000 | 1 | 98,00% |
| **Unión** | **784–256–128–10** | **0,02** | **2.000** | **2** | **98,16%** |

La última variante fue seleccionada. Al reentrenarla con las 24.501 imágenes
únicas de aprendizaje se obtienen 27.131 presentaciones antes de la ampliación
y 81.393 incluyendo las copias desplazadas. Sobre las 2.497 imágenes de test
alcanzó **98,20% de accuracy** y **98,17% de macro F1**. El recall fue **95,52%**
para el 5 y **96,30%** para el 8. El modelo del ejercicio 2 había obtenido
84,18% de accuracy y 0% de recall para el 8.

## Respuestas a la consigna

**(a) Mejor resultado.** El modelo final supera el objetivo de 98% en el test
provisto: 98,20%. Es una medición sobre este conjunto, no una garantía para
todas las imágenes futuras.

**(b) Técnicas utilizadas.** Se aprovecharon los datos nuevos y los antiguos
sin duplicarlos; se probó una arquitectura mayor, una tasa distinta, rebalanceo
de las clases escasas y desplazamientos de entrenamiento. Las comparaciones de
validación muestran el aporte de cada decisión. La mayor mejora adicional de
la última etapa vino de los desplazamientos: de 97,43% sin ellos a 98,16% con
dos copias.

**(c) Otros factores.** La distribución de datos cambió de forma importante:
el primer conjunto no contenía ochos y tenía pocos cincos. El nuevo conjunto
aporta ejemplos de ambas clases. Por eso parte de la mejora respecto del
ejercicio 2 proviene de disponer de ejemplos que antes faltaban, además de los
cambios en el entrenamiento. La unión también aporta más ejemplos únicos de
los demás dígitos. La prueba «solo datos nuevos» frente a la unión, con la
misma arquitectura y tasa, subió de 96,00% a 96,59% en validación.
