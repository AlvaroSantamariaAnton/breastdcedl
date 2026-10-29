# BreastDCEDL · Predicción de respuesta patológica completa

Trabajo académico individual de **Álvaro Santamaría Antón** sobre predicción de
respuesta patológica completa (**pCR**) mediante una CNN 2D propia entrenada desde
cero con imágenes de resonancia magnética dinámica con contraste (**DCE-MRI**).

El repositorio reúne el código, la arquitectura utilizada, una selección verificable
de los resultados del experimento final y una aplicación web que ejecuta la
inferencia en el navegador. El objetivo es estudiar el proceso completo: preparación
de entradas, separación por paciente, entrenamiento, selección en validación,
evaluación final y presentación de resultados.

> **Uso académico.** El rendimiento obtenido es limitado y no demuestra utilidad
> clínica. La aplicación no debe utilizarse para diagnóstico ni decisiones terapéuticas.

**[Abrir la aplicación web](https://breastdcedl-web.vercel.app/)** ·
[Arquitectura y diagrama](docs/ARQUITECTURA.md) ·
[Manifiesto del modelo final](experiments/modelo_final.json) ·
[Resultados conservados](experiments/experimento-7-wd001-folds/20261025T114348_995603Z/) ·
[Modelos ONNX en Hugging Face](https://huggingface.co/AvaroSantamaria/breastdcedl/tree/main)

## Contenido

- [Cómo usar la web](#cómo-usar-la-web)
- [Problema y datos](#problema-y-datos)
- [Modelo final](#modelo-final)
- [Resultados y limitaciones](#resultados-y-limitaciones)
- [Funcionamiento de la aplicación](#funcionamiento-de-la-aplicación)
- [Contenido del repositorio](#contenido-del-repositorio)
- [Referencias y condiciones de uso](#referencias-y-condiciones-de-uso)

## Cómo usar la web

La aplicación está publicada en **[breastdcedl-web.vercel.app](https://breastdcedl-web.vercel.app/)**.
Se utiliza directamente desde el navegador, sin instalar programas ni crear una
cuenta. Hace falta conexión a Internet para cargar la página y descargar los modelos.

### Explorar el proyecto

El menú permite recorrer **Datos**, **Modelo**, **Experimentos y decisiones**,
**Resultados y límites**, **Predicción** y **Referencias**. Las secciones explican
la entrada de la red, la arquitectura, las comparaciones y las limitaciones.
Puedes pulsar las figuras para ampliarlas.

### Analizar un corte

1. Entra en **Predicción** desde el menú o pulsa **Probar una predicción**.
2. Selecciona los tres archivos del mismo corte en sus casillas correspondientes:
   **PRE**, **EARLY** y **LATE**. Respeta el orden de las fases.
3. Pulsa **Analizar corte**. La primera carga puede tardar mientras se descarga
   el modelo y se prepara la inferencia en el navegador.
4. Revisa las imágenes, la visualización de realce y el resultado. La app muestra
   el score de pCR en porcentaje y la clase asignada con umbral **0,35**:
   pCR si el score es igual o superior al 35 %, y no-pCR si es inferior.
5. Si quieres conservar el resultado, utiliza el botón de descarga: obtendrás un
   archivo JSON con la predicción y datos técnicos, sin incluir las imágenes.
6. Para analizar otro corte, cambia los archivos o utiliza el botón de limpieza.
   Al cambiar una fase se descarta el resultado anterior; pulsa de nuevo
   **Analizar corte** para obtener una nueva predicción.

Cada análisis corresponde a **un corte**, representado por sus tres fases.
La interfaz no agrupa automáticamente varios análisis en una paciente. Las
métricas publicadas del test, en cambio, se calcularon agregando los cortes por
paciente; no deben atribuirse sin más a una predicción individual de la interfaz.

### Consultar la explicación visual

Después de obtener una predicción, puedes calcular el mapa **Grad-CAM** desde
su apartado. La primera consulta descarga también el modelo de explicación.
Puedes seleccionar la clase que quieres explicar (**pCR** o **no-pCR**), cambiar
la fase usada como fondo y ajustar la opacidad.

Cambiar la fase de fondo no recalcula una explicación independiente para esa
fase: el mapa corresponde a la entrada conjunta de tres canales. Elegir la clase
alternativa tampoco cambia la predicción original. Si el mapa no tiene suficiente
contraste espacial, se muestra la imagen sin colorear.

El color expresa intensidad relativa de contribución a la salida del modelo.
**No es una delimitación del tumor, una medida de certeza ni una explicación causal.**

### Si aparece un error

Comprueba que has seleccionado tres fases distintas y compatibles, que los
archivos no están dañados y que sus dimensiones coinciden. Si falla la descarga
del modelo, revisa la conexión a Internet y vuelve a intentar el análisis.
No es necesario cambiar el formato de las imágenes ni convertirlas silenciosamente
para forzar su aceptación: deben cumplir los requisitos siguientes.

### Formato de las imágenes

| Propiedad | Requisito |
|---|---|
| Archivos | Tres PNG: PRE, EARLY y LATE |
| Codificación | Escala de grises de 8 bits |
| Correspondencia | Misma paciente, mismo corte y fases alineadas |
| Dimensiones | Iguales entre las tres fases; cada lado de al menos 16 píxeles |
| Tamaño máximo | 16 MiB por archivo y 1.048.576 píxeles por fase |
| Preparación | No convertir a RGB ni aplicar normalización ImageNet |

El lector rechaza formatos incompatibles, archivos corruptos, fases idénticas por
píxeles y dimensiones discordantes. No convierte automáticamente DICOM ni PNG de
16 bits. Tampoco puede comprobar por sí solo la identidad de la paciente o la
correcta alineación de las fases.

El entrenamiento utilizó **256 × 256 píxeles**. Admitir otros tamaños no implica
que estén validados: pueden cambiar la escala anatómica y el campo de visión.

## Problema y datos

La tarea consiste en anticipar si una paciente alcanzará pCR después del tratamiento
a partir de imágenes anteriores a este. La clase positiva es **pCR**; no se trata
de detectar directamente enfermedad residual ni de segmentar el tumor.

Se utiliza el material docente derivado de BreastDCEDL, con las cohortes Duke,
I-SPY1 e I-SPY2 y tres fases temporales por corte:

- **PRE:** imagen previa al contraste.
- **EARLY:** fase temprana tras el contraste.
- **LATE:** fase tardía tras el contraste.

Son canales temporales, no los canales de color de una fotografía RGB.

| Partición utilizada | Pacientes | Cortes |
|---|---:|---:|
| Desarrollo | 1.097 | 10.945 |
| Test final | 176 | 1.758 |

El desarrollo emplea los cinco folds docentes. **La separación se hace por
paciente:** sus cortes permanecen juntos para evitar que imágenes correlacionadas
aparezcan a ambos lados de la partición. Las métricas finales se calculan por
paciente, no tratando cada corte como una observación independiente.

Las imágenes del dataset y los metadatos docentes no se distribuyen en este
repositorio. La web permite consultar las métricas y figuras del trabajo sin
acceder a esos materiales.

## Modelo final

El experimento seleccionado es **`experimento-7-wd001-folds`**. Cada red tiene
**117.329 parámetros**, seis convoluciones y la siguiente secuencia:

```text
PRE / EARLY / LATE · 3 × 256 × 256
  → Bloque 1: Conv 3→16 + ReLU → Conv 16→16 + ReLU → MaxPool
  → Bloque 2: Conv 16→32 + ReLU → Conv 32→32 + ReLU → MaxPool
  → Bloque 3: Conv 32→64 + ReLU → MaxPool
  → Bloque 4: Conv 64→128 + ReLU → MaxPool
  → Promedio global adaptativo (GAP): 128 características
  → Densa 128→64 + ReLU → Dropout 0,2
  → Densa 64→1 → logit
```

Todas las convoluciones usan kernel 3 × 3, stride 1 y padding 1. Los MaxPool
son 2 × 2 con stride 2. No se utilizan pesos preentrenados, BatchNorm, atención
ni variables clínicas en el modelo seleccionado. El [documento de arquitectura](docs/ARQUITECTURA.md)
explica el diagrama, las dimensiones y el número de parámetros capa por capa.

### Entrenamiento y selección

| Ajuste | Valor |
|---|---|
| Pérdida | BCE normal por corte |
| Optimizador | Adam |
| Learning rate | 0,001, constante |
| Weight decay | 0,001 |
| Batch size | 64 cortes |
| Semilla | 42 |
| Máximo de épocas | 50 |
| Parada por paciencia | 20 épocas sin mejora estricta, desde la época 10 |
| Checkpoint elegido | Primer máximo de AUC de validación por paciente |
| Aumento | Volteo horizontal conjunto de las fases, probabilidad 0,5, solo en train |

Se comparó BCE normal con BCE ponderada. La variante ponderada se conserva como
comparación, pero **no aporta los pesos del predictor final**.

Los píxeles se convierten a `float32` y se dividen entre 255. Después se aplica
una media y desviación comunes a las tres fases, ajustadas exclusivamente con
train de cada fold y guardadas con su checkpoint.

### Conjunto de cinco redes y umbral

El predictor conserva los cinco `best.pt`, seleccionados en las épocas
**25, 42, 15, 28 y 19** para los folds 0 a 4. No hay un reentrenamiento adicional
sobre todo desarrollo.

1. Cada red normaliza el corte con sus propias estadísticas y produce un logit.
2. Se aplica sigmoid a cada logit.
3. Se promedian las cinco probabilidades con igual peso.
4. Si hay varios cortes de una paciente, se promedian sus probabilidades con igual peso por corte.
5. Se aplica el umbral final: **p ≥ 0,35 → pCR; p < 0,35 → no-pCR**.

No se promedian pesos, logits ni clases. La interfaz actual recibe un triplete y
muestra la predicción de **un corte**; las métricas de test corresponden a la
agregación por paciente de los cortes disponibles.

El umbral 0,35 se seleccionó en validación OOF y se fijó antes de evaluar test.
El 0,30 que aparece en la configuración histórica se utilizó para las métricas
discretas del entrenamiento y no es el umbral final.

## Resultados y limitaciones

### Validación

La media del AUC por paciente en los cinco folds, evaluados en su checkpoint
seleccionado, fue **0,6117 ± 0,0407**. La dispersión es la desviación estándar
muestral entre folds, no un intervalo de confianza.

Cada paciente OOF se evalúa con la red que no la vio al entrenar. Esta validación
no mide directamente el conjunto de cinco redes. Además, los mismos folds
sirvieron para explorar configuraciones y elegir épocas; sus resultados pueden
ser optimistas.

### Test final

Evaluación sobre **176 pacientes**, con 123 no-pCR y 53 pCR, y umbral fijo 0,35.

| Métrica | Resultado |
|---|---:|
| ROC-AUC | 0,5693 |
| IC del 95 % del AUC | 0,4786–0,6616 |
| Balanced accuracy | 0,5179 |
| Macro-F1 | 0,5157 |
| Sensibilidad para pCR | 0,3774 |
| Especificidad | 0,6585 |
| Precisión para pCR | 0,3226 |
| Accuracy | 0,5739 |
| Brier score | 0,2078 |

Matriz de confusión, con filas reales y columnas predichas:

| | Predicción no-pCR | Predicción pCR |
|---|---:|---:|
| **Real no-pCR** | 81 | 42 |
| **Real pCR** | 33 | 20 |

El intervalo del AUC se obtuvo mediante 2.000 réplicas bootstrap estratificadas
por paciente, semilla 42, manteniendo fijo el predictor. No incorpora la
incertidumbre de la exploración y selección del modelo.

Fuente: [métricas del test final](experiments/experimento-7-wd001-folds/20261025T114348_995603Z/final_test/metrics.json).

### Cómo interpretar estos resultados

- El intervalo del AUC incluye **0,5** y la balanced accuracy está próxima a ese
  valor. Los resultados no permiten afirmar una discriminación clínica sólida.
- Predecir siempre no-pCR obtiene una accuracy de **0,6989** por el desbalance,
  pero sensibilidad pCR de cero y balanced accuracy de 0,5. Por eso accuracy
  por sí sola no resume la utilidad del clasificador.
- Persiste sobreajuste, especialmente en el fold 3. La regularización y la
  selección de checkpoints no garantizan su eliminación.
- No se ha ajustado un calibrador. El score no debe interpretarse como una
  probabilidad clínica individual certificada.
- Las métricas por paciente no demuestran el mismo rendimiento con un único
  corte, con tamaños distintos ni con pacientes de otra distribución.
- El test ya se evaluó con el predictor cerrado. No debe usarse para reajustar
  arquitectura, umbral o hiperparámetros y seguir considerándolo independiente.

## Funcionamiento de la aplicación

La interfaz está escrita en HTML, CSS y JavaScript. ONNX Runtime Web ejecuta el
predictor en CPU mediante WebAssembly. Los modelos ONNX incorporan la
normalización de cada fold, sigmoid y la media de las cinco redes.

| Recurso | Ubicación y función |
|---|---|
| Interfaz, lector PNG y lógica de inferencia | `web/` |
| Runtime JavaScript y WebAssembly | `web/vendor/` |
| Hash del predictor, umbral y versión del runtime | `web/model/manifest.json` |
| URL y hash del modelo Grad-CAM | `web/model/gradcam.json` |
| Modelo de predicción y modelo Grad-CAM | Hugging Face |
| Checkpoints PyTorch y exportaciones archivadas | `experiments/` |

Los modelos remotos se verifican mediante **SHA-256** antes de utilizarlos.
La URL de `model.onnx` está fijada a una revisión; la de Grad-CAM utiliza `main`
y también exige el hash esperado. Un archivo remoto distinto no se acepta
silenciosamente.

La aplicación está alojada en **Vercel** y descarga los modelos desde
**Hugging Face**. Las imágenes seleccionadas se procesan en el navegador y el
código de inferencia no las envía a un servidor. La descarga de los modelos es
una conexión externa independiente.

Grad-CAM muestra contribuciones espaciales a la salida de las redes. No es una
segmentación, no identifica necesariamente el tumor y no demuestra causalidad.

## Contenido del repositorio

| Ubicación | Contenido |
|---|---|
| [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md) | Explicación y diagrama estático de la arquitectura utilizada |
| [`experiments/modelo_final.json`](experiments/modelo_final.json) | Identidad de las cinco redes, épocas seleccionadas, hashes, agregación y umbral |
| [`experiments/experimento-7-wd001-folds/`](experiments/experimento-7-wd001-folds/) | Configuración, pesos finales, métricas, curvas y exportaciones del experimento elegido |
| [`src/breastdcedl/`](src/breastdcedl/) | Implementación de la CNN, preprocesado, inferencia y evaluación |
| [`scripts/`](scripts/) | Código de entrenamiento, análisis y preparación utilizado en el proyecto |
| [`web/`](web/) | Interfaz, figuras y lógica de predicción en el navegador |

El archivo del experimento conserva una selección de evidencias del trabajo:
los cinco pesos finales, métricas por época, normalización, curvas principales,
resultados de test y la comparación con BCE ponderada. No incluye todos los
checkpoints intermedios ni las revisiones con imágenes de pacientes.

## Referencias y condiciones de uso

- [Artículo de BreastDCEDL](https://doi.org/10.1038/s41597-026-06589-6).
- [Registro del dataset en Zenodo](https://doi.org/10.5281/zenodo.18114231).
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391).
- [Exportación ONNX de PyTorch](https://docs.pytorch.org/docs/stable/onnx.html).
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/).

Consultar las condiciones de los datos y materiales originales antes de
redistribuirlos. El runtime conserva su [licencia de terceros](web/vendor/LICENSE).
Este README no establece una licencia nueva para el código, los pesos o los datos.

**Autor:** Álvaro Santamaría Antón.
