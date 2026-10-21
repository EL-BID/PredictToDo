# Pesos GConvGRU regionales · 15 de mayo de 2026

Última entrega regional identificada en los artefactos disponibles del proyecto:
modelos central y no central, seleccionados en el manifiesto del 18 de mayo de
2026 como candidatos finales para despliegue y pruebas. Los tensores de pesos se conservan
sin modificaciones; se exporta únicamente el `state_dict` del checkpoint para
evitar dependencias de serialización y rutas internas de entrenamiento. Cada región tiene su propio `model.pth`, configuración JSON,
procedencia y métricas de validación. No son dos versiones intercambiables de un
modelo único para toda Bogotá.

## Compatibilidad de las columnas

**Estos pesos solo deben utilizarse con las columnas, el orden y las
transformaciones de esta versión. Tener el mismo número de columnas no basta.**

- [input_columns.json](input_columns.json) registra el orden efectivo de entrada,
  los grupos de datos y el orden de salidas.
- [COLUMNS.md](COLUMNS.md) permite consultar cada columna y su posición.
- [feature_order.original.json](feature_order.original.json) conserva el registro
  original del entrenamiento. No usar sus listas planas directamente: incluyen
  `road_id` y `timeframe`, que no forman parte de las dimensiones dinámicas
  efectivas registradas en `data_meta`. El registro efectivo usa esas dimensiones
  por archivo y añade calendario y clima en el orden del procesamiento.

| Tensor al entrar en `GNN_Temporal_Model` | Forma | Contenido |
|---|---|---|
| `static_inputs` | `(N, 117)` | Atributos de cada vía |
| `dynamic_seq_inputs` | `(B, 150, N, 130)` | 120 variables históricas + 6 de calendario + 4 de clima |
| `future_known_inputs` | `(B, 50, N, 11)` | 5 variables programadas + 6 de calendario |
| `edge_index` | `(2, E)` | Conexiones, con índices alineados a las filas de las vías |
| Salida | `(B, N, 50, 2)` | Congestión y velocidad, en ese orden |

`B` es el número de ventanas, `N` el número de vías y `E` el de conexiones.
La resolución es de 15 minutos. Se conservan 150 pasos históricos y 50 futuros.
El calendario del contrato es Colombia y su zona horaria declarada es
`America/Bogota`. El clima se concatena **después** de las variables de calendario,
solo en el historial: `wind_direction`, `humidity`, `wind_speed`, `precipitation`.

No ordenar alfabéticamente las columnas ni los archivos de entrada. No eliminar
categorías sin observaciones en una muestra, intercambiar nombres parecidos,
rellenar fuentes enteras faltantes con ceros ni sustituir las variables agregadas
de `jams_variables_dataset.h5` por las del ejemplo sin reconstruir su significado.
Las variables futuras deben estar disponibles al emitir el pronóstico.

## Alcance y elementos necesarios para inferencia real

El orden dinámico y futuro se reconstruyó con los metadatos del propio paquete,
las dimensiones efectivas registradas y el código de preparación de entradas.
El paquete original **no incluye el orden estático**: las 117 columnas estáticas
se recuperaron de los metadatos procesados locales y coinciden en dimensión.
Su huella y esta procedencia quedan registradas en `input_columns.json`; no se
ha establecido de forma independiente su identidad con el archivo histórico
exacto usado en el entrenamiento.

El paquete tampoco contiene las constantes de normalización del entrenamiento.
**No se suministra un sustituto estimado con datos nuevos.** Para inferencia real
se deben recuperar y verificar las constantes originales, la preparación e
imputación de variables, el grafo y la selección regional compatibles. El
cargador divide por constantes por variable y limita a `[-1, 1]`; calendario se
añade después. Las salidas requieren la transformación inversa correspondiente
para interpretarse en niveles de congestión y velocidad, no son necesariamente
unidades originales al salir de la red.

El notebook `notebooks/local_training_example.ipynb` entrena un modelo didáctico
con otra configuración y normalización. **No se pueden sustituir sus pesos por
estos archivos directamente.** Su muestra omite clima y usa fuentes Waze
separadas. Esta publicación conserva pesos y documenta sus entradas; no constituye
un paquete autónomo de inferencia ni acredita reproducir el desempeño operativo.

Los modelos se entrenaron con artefactos parciales de semillas y ventanas
regionales. Las métricas de cada `provenance.json` pertenecen a su validación
original; no representan una nueva evaluación ni una garantía para otras ciudades.

## Cargar pesos con la implementación del repositorio

Desde la raíz del repositorio, con sus dependencias instaladas:

```python
import json
from pathlib import Path
import torch
from src.models.gnn_temporal import GNN_Temporal_Model

folder = Path("models/gconvgru_regional_2026_05_15/central")
config = json.loads((folder / "model_config.json").read_text())
model = GNN_Temporal_Model(edge_index=None, **config)
state = torch.load(folder / "model.pth", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
# Antes de invocar model(...), preparar los tensores según input_columns.json
# y verificar las transformaciones originales descritas arriba.
```

Cambiar `central` por `noncentral` selecciona el otro modelo regional.
`checksums.json` registra SHA-256 de los pesos y archivos de contrato.
La comprobación de publicación carga ambos estados estrictamente y ejecuta
una pasada con tensores sintéticos para verificar las formas y valores finitos;
no valida la preparación de datos reales ni la precisión de las predicciones.
