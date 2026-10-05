# SIA-TP3

## Organización

- `models/`: perceptrones y funciones de activación compartidos.
- `tests/`: validaciones con problemas simples.
- `ej1_fraude/`: comparación y generalización para fraude.
- `ej2_digitos/`: clasificación de dígitos.
- `ej3_mejora/`: mejora del clasificador con más datos.

Las carpetas de los ejercicios 2 y 3 están preparadas para incorporar sus
datasets y experimentos.

El [Ejercicio 1](ej1_fraude/README.md) ya incluye la exploración del dataset de
fraude, la comparación de modelos, el estudio de generalización y el umbral
recomendado. Con los archivos del enunciado en `data/`, se reproduce con
`python -m ej1_fraude.run`.

## Experimentos en paralelo

El runner de `scripts/batch.py`, adaptado del TP2, ejecuta varias corridas
independientes del ejercicio 1 con distintas semillas. Los hilos coordinan
subprocesos separados para que cada corrida tenga su propio estado de NumPy y
sus propios archivos de salida:

```powershell
uv run python -m scripts.batch --seeds 2,3,4 --tasks 2
```

`--tasks` limita las corridas simultáneas. Cada semilla guarda su `results.json`,
gráficos, modelo y `run.log` en `results/batch_ej1/seed_N/`; el resumen queda en
`results/batch_ej1/summary.csv`. Si alguna corrida falla o supera `--timeout`,
el comando termina con código distinto de cero.

## Validación de los perceptrones

Desde la raíz del proyecto, ejecutar:

```powershell
python -m tests.validate_models
```

El programa prueba AND con escalón, el ajuste de `y = x` con activación lineal,
el ajuste de `y = tanh(x)` con activación no lineal y XOR con un perceptrón
multicapa de topología `[2, 3, 2, 1]`. Usa la semilla 2 y guarda parámetros,
métricas por época, pesos finales y predicciones en `results/validation.json`.
Se puede cambiar la semilla con `--seed` y el archivo de salida con `--output`.
El comando devuelve un código distinto de cero si alguna validación falla.
Estas pruebas verifican que los algoritmos aprenden ejemplos simples; no miden
generalización sobre datos nuevos.
