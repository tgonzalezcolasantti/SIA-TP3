# SIA-TP3

## Validación de los perceptrones

Desde la raíz del proyecto, ejecutar:

```powershell
python validate_models.py
```

El programa prueba AND con escalón, el ajuste de `y = x` con activación lineal,
el ajuste de `y = tanh(x)` con activación no lineal y XOR con un perceptrón
multicapa de topología `[2, 3, 2, 1]`. Usa la semilla 2 y guarda parámetros,
métricas por época, pesos finales y predicciones en `results/validation.json`.
Se puede cambiar la semilla con `--seed` y el archivo de salida con `--output`.
El comando devuelve un código distinto de cero si alguna validación falla.
Estas pruebas verifican que los algoritmos aprenden ejemplos simples; no miden
generalización sobre datos nuevos.
