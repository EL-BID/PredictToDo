# PredictToDo

PredictToDo pronostica la congestión vehicular por segmento de vía. Combina el
historial de tráfico con las conexiones de la red vial y las características de
su entorno para aprender cómo cambia la congestión en el espacio y en el tiempo.
El resultado es una secuencia de niveles de congestión previstos para cada vía.

La red se representa como un grafo: los nodos son segmentos viales y las aristas
son conexiones entre ellos. Los modelos gráficos temporales utilizan esa
estructura junto con secuencias de observaciones para generar sus predicciones.
El desarrollo se realizó con datos de Bogotá; el mismo flujo puede adaptarse a
otras ciudades preparando datos equivalentes y evaluando los modelos en ese contexto.

El uso básico consiste en **preparar los datos, construir ventanas temporales,
entrenar un modelo y evaluar sus predicciones sobre periodos posteriores**.
Los componentes de Python se pueden utilizar desde un notebook o una aplicación,
en un equipo local o en cualquier infraestructura con las dependencias necesarias.

- [Ejemplo de uso](#ejemplo-de-uso)
- [Datos de entrada](#datos-de-entrada)
- [Instalación](#instalación)
- [Preparar los datos](#preparar-los-datos)
- [Entrenar un modelo](#entrenar-un-modelo)
- [Evaluar y generar predicciones](#evaluar-y-generar-predicciones)
- [Adaptar el ejemplo](#adaptar-el-ejemplo)
- [Pruebas y contribuciones](#pruebas-y-contribuciones)
- [Licencia](#licencia)

## Ejemplo de uso

El notebook [Entrenamiento y evaluación con datos locales](notebooks/local_training_example.ipynb)
muestra el procedimiento completo utilizando directamente el cargador y los
modelos de `src`. Permite configurar los archivos de entrada, inspeccionar una
muestra, entrenar una red GConvGRU, comparar sus predicciones con una referencia
sencilla y guardar los pesos junto con su configuración.

La estructura del ejemplo está preparada para una muestra de datos preprocesados
que se incorporará posteriormente. Sus nombres de archivo son ilustrativos y se
ajustan en una única celda de configuración. El notebook explica el formato
esperado y señala los archivos faltantes antes de iniciar el entrenamiento.

## Datos de entrada

Para el caso de Bogotá se combinaron datos de tráfico, infraestructura vial,
entorno urbano y condiciones meteorológicas. La información se relaciona con
los segmentos de la red vial; las observaciones de tráfico y las afectaciones
se agregan en intervalos de 15 minutos.

| Dato utilizado | Qué representa | Cómo se utiliza |
|---|---|---|
| **Malla vial integral (MVI)** | Geometría de los segmentos de vía, sus identificadores, longitud y sentido de circulación. | Construye el grafo vial y permite asignar las demás observaciones a cada segmento. |
| **Congestión y velocidades de Waze** | Registros de congestión con ubicación, nivel de congestión, velocidad y demora. | Genera el historial de tráfico y los valores objetivo para entrenar y evaluar las predicciones. Se calculan estadísticas por segmento e intervalo. |
| **Irregularidades de Waze** | Reportes de condiciones anómalas del tráfico y alertas asociadas, como accidentes, peligros y cierres. | Aporta señales sobre alteraciones del tráfico, agregadas según su coincidencia espacial y temporal con cada vía. |
| **Eventos** | Aglomeraciones, manifestaciones y emergencias, con localización y duración. | Representa afectaciones temporales sobre las vías próximas y conectadas al lugar del evento. |
| **Incidentes de tránsito** | Registros localizados de accidentes y otras incidencias, clasificados por tipo y detalle del caso. | Cuenta y caracteriza los incidentes asociados a cada segmento e intervalo temporal. |
| **Planes de manejo de tránsito (PMT)** | Intervenciones sobre la red vial representadas mediante puntos y tramos, con horarios de vigencia. | Indica la presencia de intervenciones durante la ventana observada y, cuando se conocen sus horarios, durante el horizonte de predicción. |
| **Clima** | Mediciones de estaciones meteorológicas: precipitación, humedad, velocidad del viento y dirección del viento. | Asigna las mediciones a las vías por proximidad y las alinea con las ventanas temporales del tráfico. |
| **Edificaciones de OpenStreetMap** | Huellas de edificios y sus categorías, como residencial, comercial, industrial o de uso público. | Cuenta edificaciones por tipo en un entorno de 100 metros de cada segmento para caracterizar su contexto urbano. |
| **Paraderos zonales del SITP** | Localización y características de los puntos de parada del transporte público. | Asocia los paraderos a los segmentos cercanos y calcula conteos y distancias por vía. |
| **Zonificación y uso del suelo** | Delimitación de las Zonas de Análisis de Transporte (ZAT) y superficies destinadas a actividades residenciales, comerciales, industriales, educativas y otras. | Relaciona cada vía con su zona y añade características de las actividades urbanas de su entorno. |
| **Población por ZAT** | Población total y por grupos de edad de cada Zona de Análisis de Transporte. | Incorpora atributos demográficos al grafo mediante la asociación espacial entre zonas y vías. |
| **Estado de la malla vial** | Condición física y calificaciones del estado de los segmentos viales. | Añade características de la infraestructura que contextualizan las condiciones de circulación. |

Además, se derivan variables de calendario a partir de las fechas: hora, día de
la semana, fin de semana y festivos de Colombia. Estas variables describen los
patrones recurrentes del tráfico y pueden calcularse para el horizonte futuro.

Para reproducir un modelo se debe conservar la versión de los datos procesados,
el orden de variables y las constantes de normalización utilizados en su
entrenamiento. El [detalle de las fuentes y su procesamiento](docs/data-sources.md)
relaciona cada conjunto con los componentes que lo transforman.

Los datos de OpenStreetMap se atribuyen a **© OpenStreetMap contributors**.

## Instalación

Utilizar Python **3.12** y las dependencias declaradas en
[pyproject.toml](pyproject.toml). Desde la carpeta del proyecto, instalar el entorno
con `uv`:

```bash
uv sync --all-groups
uv run jupyter lab
```

Seleccionar el notebook de ejemplo y un kernel asociado a ese entorno. PyTorch,
PyTorch Geometric y PyTorch Geometric Temporal implementan los modelos; NumPy,
Pandas, SciPy y HDF5 permiten manipular sus entradas. Las extensiones de PyTorch
Geometric deben ser compatibles con la versión de PyTorch y el dispositivo usado.
El notebook selecciona GPU si está disponible y CPU en caso contrario.

## Preparar los datos

El modelo necesita un grafo vial, atributos estáticos por segmento, observaciones
históricas y valores objetivo para aprender. Los identificadores y el orden de
las vías deben coincidir en todos los conjuntos; las observaciones deben
compartir una resolución temporal regular.

Los módulos de [preprocesamiento](src/preprocessing/) permiten asociar las fuentes
a las vías, agregar observaciones y construir representaciones dispersas. Si ya
se dispone de datos preprocesados compatibles, se puede comenzar directamente
con [TrafficDataset](src/sparse_dataloader.py).

El cargador forma muestras alrededor de una vía de referencia y selecciona una
ventana histórica seguida del periodo que se desea pronosticar:

| Elemento de la muestra | Significado |
|---|---|
| Atributos estáticos | Características de cada vía y su entorno |
| Secuencia histórica | Variables observadas antes de emitir el pronóstico |
| Variables futuras conocidas | Información disponible de antemano, como calendario o intervenciones programadas |
| Conexiones del grafo | Relaciones entre las vías seleccionadas |
| Valores objetivo | Congestión observada durante el horizonte que se aprende a predecir |

El formato de los archivos y las dimensiones de los tensores se explican en el
notebook. Antes de entrenar, definir qué significan los valores ausentes y cómo
se tratarán. Un registro faltante no equivale necesariamente a ausencia de congestión.

## Entrenar un modelo

Separar los datos cronológicamente en entrenamiento, validación y prueba. Las
ventanas de diferentes particiones no deben compartir observaciones. Calcular
las transformaciones de normalización con entrenamiento y conservarlas sin
reajustar para los demás periodos.

[GNN_Temporal_Model](src/models/gnn_temporal.py) implementa las arquitecturas
GConvGRU, GConvLSTM y TGCN, entre otras. Para crear un modelo se indican las
dimensiones de sus variables, la longitud de la secuencia histórica y el horizonte
de predicción. Estas dimensiones deben corresponder con las muestras del cargador.

El ciclo de aprendizaje sigue cuatro pasos:

1. Obtener una muestra con las vías, sus conexiones y las secuencias temporales.
2. Ejecutar el modelo para predecir los niveles de congestión futuros.
3. Comparar las predicciones con los objetivos mediante una función de pérdida.
4. Propagar los gradientes y actualizar los pesos con un optimizador.

El notebook hace explícito ese ciclo con PyTorch, sin depender de lanzadores ni
servicios externos. Utiliza error cuadrático medio para enseñar el procedimiento.
Las [funciones de pérdida](src/training/losses.py) permiten explorar criterios
más específicos para los niveles de congestión.

La validación sirve para elegir arquitectura, parámetros y versión de los pesos.
El periodo de prueba se reserva para evaluar el modelo una vez terminada esa
selección. Un entrenamiento corto sobre una muestra pequeña demuestra el flujo;
la calidad predictiva requiere una evaluación representativa.

## Evaluar y generar predicciones

Comparar las predicciones con observaciones posteriores a las utilizadas para
entrenar. El error absoluto medio (MAE) resume cuánto difieren los niveles previstos
de los observados. Revisar también el comportamiento por horizonte, nivel de
congestión y zona, y compararlo con una referencia sencilla.

El notebook incluye una referencia constante calculada con entrenamiento y una
tabla de valores observados y predichos. En una evaluación más amplia se puede
comparar también con persistencia: mantener el último nivel observado como
predicción futura.

Para generar predicciones nuevas, cargar los pesos, preparar una ventana reciente
y aplicar exactamente el mismo orden de variables y normalización del
entrenamiento. Las entradas futuras solo pueden contener información conocida al
emitir el pronóstico. Las observaciones futuras de tráfico se reservan para evaluar.

Guardar junto a los pesos la arquitectura, las transformaciones, la definición
de variables y la versión de los datos permite repetir la inferencia. El ejemplo
incluye una comprobación de que guardar y recargar el modelo conserva la predicción.

## Adaptar el ejemplo

Comenzar con una muestra pequeña y coherente de vías y periodos facilita comprobar
el flujo antes de ampliar el entrenamiento. Ajustar las ventanas y el horizonte a
la resolución temporal y al uso previsto; añadir variables exige actualizar sus
metadatos y las dimensiones del modelo.

Para cubrir una red mayor, recorrer distintos subgrafos manteniendo alineados los
nodos, las aristas y las variables de cada muestra. Para trabajar con otra ciudad,
adaptar las fuentes y validar con observaciones locales antes de utilizar los
resultados en decisiones operativas.

## Pruebas y contribuciones

Las [pruebas](tests/) documentan el comportamiento esperado del cargador, las
transformaciones y los componentes de entrenamiento. Para ejecutar la suite:

```bash
uv run pytest tests
```

Las pruebas de software comprueban contratos y comportamiento; la evaluación
predictiva requiere datos reservados y métricas. Al modificar un componente,
comprobar tanto sus entradas y salidas como su efecto sobre el flujo completo.

Para contribuir, describir el problema, el cambio y su validación en el pull
request. Mantener un trato respetuoso en issues y revisiones y aportar ejemplos
reproducibles. La descripción metodológica está disponible en el
[white paper](white_paper/traffic_forecasting.tex).

## Licencia

La licencia del proyecto está incluida en [LICENSE.md](LICENSE.md). Consulta ese archivo para conocer los permisos, condiciones y restricciones de uso, modificación y distribución de la herramienta.

## Limitación de responsabilidades

El BID no será responsable, bajo circunstancia alguna, de daño ni indemnización, moral o patrimonial; directo o indirecto; accesorio o especial; o por vía de consecuencia, previsto o imprevisto, que pudiese surgir:

i. Bajo cualquier teoría de responsabilidad, ya sea por contrato, infracción de derechos de propiedad intelectual, negligencia o bajo cualquier otra teoría; y/o

ii. A raíz del uso de la Herramienta Digital, incluyendo, pero sin limitación de potenciales defectos en la Herramienta Digital, o la pérdida o inexactitud de los datos de cualquier tipo. Lo anterior incluye los gastos o daños asociados a fallas de comunicación y/o fallas de funcionamiento de computadoras, vinculados con la utilización de la Herramienta Digital.

## Acknowledgments / Reconocimientos

**Copyright © [2025]. Inter-American Development Bank ("IDB"). Authorized Use.**
The procedures and results obtained based on the execution of this software are those programmed by the developers and do not necessarily reflect the views of the IDB, its Board of Executive Directors or the countries it represents.

**Copyright © [2025]. Banco Interamericano de Desarrollo ("BID"). Uso Autorizado.**
Los procedimientos y resultados obtenidos con la ejecución de este software son los programados por los desarrolladores y no reflejan necesariamente las opiniones del BID, su Directorio Ejecutivo ni los países que representa.

### Support and Usage Documentation / Documentación de Soporte y Uso

**Copyright © [2025]. Inter-American Development Bank ("IDB").** The Support and Usage Documentation is licensed under the Creative Commons License CC-BY 4.0 license. The opinions expressed in the Support and Usage Documentation are those of its authors and do not necessarily reflect the opinions of the IDB, its Board of Executive Directors, or the countries it represents.

**Copyright © [2025]. Banco Interamericano de Desarrollo (BID).** La Documentación de Soporte y Uso está licenciada bajo la licencia Creative Commons CC-BY 4.0. Las opiniones expresadas en la Documentación de Soporte y Uso son las de sus autores y no reflejan necesariamente las opiniones del BID, su Directorio Ejecutivo ni los países que representa.
