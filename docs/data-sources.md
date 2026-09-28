# Datos utilizados en PredictToDo

El caso de Bogotá combina observaciones de tráfico con información de la red
vial, el entorno urbano, las afectaciones a la movilidad y el clima. La tabla de
[datos de entrada](../README.md#datos-de-entrada) describe cada conjunto y su uso.

## Información espacial y urbana

La malla vial permite identificar los segmentos y sus conexiones según la
geometría y el sentido de circulación. Sobre esa estructura se incorporan las
características del entorno:

- Las edificaciones se cuentan por categoría dentro de un entorno de 100 metros
  de cada segmento. Las categorías incluyen usos residenciales, comerciales,
  industriales y públicos. La fuente es OpenStreetMap, con atribución
  **© OpenStreetMap contributors**.
- Los paraderos del SITP se relacionan con los segmentos próximos para obtener
  conteos y distancias que describen la presencia del transporte público.
- Las Zonas de Análisis de Transporte (ZAT) aportan la delimitación territorial
  para asociar población y usos del suelo a las vías.
- La población incluye el total y los grupos de 5–24, 25–64 y mayores de 65 años
  definidos en el procesamiento.
- El uso del suelo describe superficies destinadas a actividades residenciales,
  comerciales, industriales, educativas y otras.
- El estado de la malla vial incorpora clasificaciones de condición física y
  atributos de la infraestructura.

Estos datos enriquecen los nodos del grafo. La selección final de variables debe
corresponder con la utilizada durante el entrenamiento de cada modelo.

## Tráfico y afectaciones temporales

Los registros de congestión de Waze contienen nivel de congestión, velocidad y
demora. Se asignan a segmentos viales y se agregan en intervalos de 15 minutos
para construir el historial y las variables objetivo.

Las irregularidades de Waze aportan alertas y condiciones anómalas del tráfico.
Su asociación con las vías considera la coincidencia espacial y temporal de
cada registro.

Los eventos, los incidentes y los planes de manejo de tránsito representan
fuentes distintas de afectación. Los eventos describen aglomeraciones,
manifestaciones y emergencias; los incidentes registran accidentes y otras
incidencias clasificadas por tipo. Los planes de manejo de tránsito describen
intervenciones puntuales o sobre tramos y sus horarios de vigencia. Los horarios
conocidos permiten construir variables para el horizonte futuro.

## Clima y calendario

Se utilizan cuatro variables meteorológicas: humedad, precipitación, dirección
del viento y velocidad del viento. Las mediciones de las estaciones se asocian
con las vías por proximidad y se alinean temporalmente con el tráfico.

La hora, el día de la semana, la condición de fin de semana y los festivos de
Colombia se calculan a partir de las fechas. Permiten representar patrones
recurrentes y están disponibles para los periodos futuros.

## Alcance de la descripción

La descripción se contrastó con los procesadores de ambos proyectos, la
documentación metodológica y los registros de entrenamiento. Aunque el documento
metodológico menciona temperatura, las entradas verificadas contemplan las cuatro
variables meteorológicas indicadas arriba.

Los volúmenes semafóricos forman parte de los insumos recopilados durante el
proyecto, pero no se confirmó su integración en el flujo de entrenamiento
revisado. Por ello no se presentan como entrada utilizada por el modelo.

Las fechas de corte, los enlaces originales y las condiciones particulares de
cada fuente requieren completar su trazabilidad. Esta descripción explica el
contenido y el uso de los datos; no establece permisos de redistribución.
