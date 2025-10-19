# Práctica 1: Implementación manual de una Red Neuronal con una capa oculta

**Curso:** Redes Neuronales  
**Instructor:** Sabino Miranda

## 1. Objetivos

### 1.1. Objetivo
Implementar desde cero una red neuronal multicapa (MLP) con una capa oculta y una neurona de salida para resolver el problema clásico XOR y posteriormente adaptarla para problemas de clasificación reales utilizando diferentes configuraciones de hiperparámetros.

### 1.2. Objetivos específicos
1) Comprender el funcionamiento interno del forward y backward propagation en una red neuronal.
2) Implementar una red neuronal desde cero para implementar los mecanismos de aprendizaje de la red.
3) Aplicar la red al problema XOR y posteriormente generalizarla a datasets reales y sintéticos.
4) Experimentar con diferentes configuraciones:
   - Número de neuronas en la capa oculta
   - Inicialización de pesos (Distribución Normal y Xavier)
   - Tasa de aprendizaje (learning rate)
   - Tamaño de lote (batch size)
   - Número de épocas (Epochs)
5) Analizar y graficar el comportamiento del error cuadrático medio (MSE) al variar las configuraciones.
6) Comparar el desempeño de las configuraciones mediante métricas y visualizaciones de las gráficas.

## 2. Primera Parte: Red neuronal para resolver el problema de la XOR

1) Implementar una red neuronal, usar como base la clase MLP_TODO del notebook [MLP_XOR_TODO_class](https://github.com/uacm-cs/redes_neuronales/blob/main/07_MLP_XOR_TODO_class.ipynb), de una capa oculta y una capa de salida con las siguientes características:
   - Entradas: 2 características (features)
   - Capa oculta: 2 neuronas
   - Capa de salida: 1 neurona
   - Función de activación: sigmoide
   - Función de error: MSE con factor $\frac{1}{2}$

2) Implementar las funciones:
   - sigmoid(z) y sigmoid_derivative(z)
   - forward
   - backward
   - loss_function_MSE
   - update
   - xavier_initialization
   - train
   - predict

3) Entrenar la red con los 4 patrones de la tabla XOR.

4) Graficar la evolución del error respecto a cada época.

5) Mostrar los resultados de salida de predicción para las 4 combinaciones de entrada.

## 3. Segunda Parte: Extensión a datasets de clasificación

Reutilizar la arquitectura y código de la primera parte, para ahora aplicar la red a los siguientes conjuntos de datos.

**Nota:** Para todos los datasets, la clase esperada (objetivo o target) es la última columna en el conjunto de datos, denotada como y.

### 3.1. Descripción de los datasets

A continuación se detallan las características de cada dataset que se utilizará en la práctica. Datasets utilizados en la práctica provenientes de sklearn.datasets:

- iris
- breast cancer
- wine

#### 3.1.1. Dataset: Iris (clasificación de plantas)

El conjunto de datos contiene tres clases, donde cada clase se refiere a un tipo de planta de iris. El conjunto de datos original se adaptó para clasificación binaria. Se usan solo las clases 0 y 1 (Setosa y Versicolor) para clasificación binaria.

**Fuente:** sklearn.datasets.load_iris() [1]  
**Número de características:** 4  
**Etiquetas de clase (y):**
- Clase 0: Setosa
- Clase 1: Versicolor

**Columnas:**

| Columna | Descripción |
|---------|-------------|
| 0: sepal length (cm) | Longitud del sépalo |
| 1: sepal width (cm) | Ancho del sépalo |
| 2: petal length (cm) | Longitud del pétalo |
| 3: petal width (cm) | Ancho del pétalo |
| y: (0 o 1) | Clase del ejemplo actual: Setosa=0 o Versicolor=1 |

#### 3.1.2. Dataset: Breast Cancer (tumores malignos vs benignos)

El conjunto de datos contiene características que se calculan a partir de una imagen digitalizada de una aspiración con aguja fina (PAAF) de una masa mamaria.

**Fuente:** Breast Cancer Wisconsin dataset, UCI Machine Learning Repository [2]  
**Número de características:** 30  
**Etiquetas de clase (y):**
- Clase 0: Maligno
- Clase 1: Benigno

**Columnas:** Cada columna representa una medida de los núcleos de células en imágenes de tumores de mama. Incluyen:

| Columna | Descripción |
|---------|-------------|
| 0: mean radius | Promedio del radio del núcleo |
| 1: mean texture | Promedio de la textura |
| 2: mean perimeter | Promedio del perímetro |
| 3: mean area | Promedio del área |
| 4: mean smoothness | Promedio de la suavidad |
| ... | 30 características en total |
| y: (0 o 1) | Clase del ejemplo actual: Maligno=0, Benigno=1 |

#### 3.1.3. Dataset: Wine (Vinos)

El conjunto de datos **Wine Recognition**[3] contiene mediciones químicas de vinos cultivados en la región de Piamonte, Italia, y producidos a partir de tres variedades de uva diferentes. El conjunto se adaptó para clasificación binaria por lo cual solo se seleccionaron las clases 0 y 1: Vino tipo Barolo y Vino tipo Grignolino.

- **Número de características:** 13 (todas numéricas y continuas)
- **Número de clases adaptadas:** 2
- **Etiquetas de clase (y):**
  - Clase 0: Vino tipo Barolo
  - Clase 1: Vino tipo Grignolino

**Descripción de las variables del dataset Wine Recognition (versión binaria):**

| Columna | Descripción de la variable |
|---------|----------------------------|
| 0 | Alcohol |
| 1 | Ácido málico |
| 2 | Cenizas |
| 3 | Alcalinidad de las cenizas |
| 4 | Magnesio |
| 5 | Fenoles totales |
| 6 | Flavonoides |
| 7 | Fenoles no flavonoides |
| 8 | Proantocianidinas |
| 9 | Intensidad de color |
| 10 | Matiz |
| 11 | OD280/OD315 de vinos diluidos |
| 12 | Prolina |
| y | Clase del ejemplo actual (Barolo=0, Grignolino=1) |

#### 3.1.4. Archivos de entrenamiento (train) y prueba (test)

Los conjuntos de datos utilizados se encuentran en la subcarpeta [datasets](https://github.com/uacm-cs/redes_neuronales/blob/main). Cada archivo contiene un conjunto de datos en formato CSV, donde la primera fila corresponde al encabezado de las columnas.

**Tabla 1: Descripción de los archivos de entrenamiento y prueba**

| Archivo | Descripción |
|---------|-------------|
| iris_train.csv | Datos de entrenamiento del conjunto Iris |
| iris_test.csv | Datos de prueba del conjunto Iris |
| breast_cancer_train.csv | Datos de entrenamiento del conjunto Breast Cancer |
| breast_cancer_test.csv | Datos de prueba del conjunto Breast Cancer |
| wine_train.csv | Datos de entrenamiento del conjunto Wine |
| wine_test.csv | Datos de prueba del conjunto Wine |

**Estructura de los archivos**

Cada archivo contiene:
- Primera fila: encabezado con los nombres de las columnas (características y clase)
- Columnas:
  - Las primeras columnas corresponden a las variables de entrada (X)
  - La última columna corresponde a la etiqueta o clase (y)

**Ejemplo de Código para carga y conversión a NumPy**

```python
import pandas as pd

# --- Cargar los conjuntos de datos ---
# header=0 : La primera fila son los encabezados, no forma parte de los datos
iris_train = pd.read_csv("datasets/iris_train.csv", header=0)
iris_test = pd.read_csv("datasets/iris_test.csv", header=0)

# --- Separar características (X) y etiquetas (y) ---
def split_features_labels(df):
    X = df.iloc[:,:-1].to_numpy() # todas las columnas excepto la última
    y = df.iloc[:,-1].to_numpy() # última columna como etiquetas
    return X, y

X_iris_train, y_iris_train = split_features_labels(iris_train)
X_iris_test, y_iris_test = split_features_labels(iris_test)
```

### 3.2. Pasos generales por realizar

1) Cargar los datos
2) Usar los datos originales y usar datos normalizados
3) Entrenar la red con el conjunto de datos (*dataset*) *train*
4) Evaluar el desempeño con el conjunto de datos (*dataset*) *test*
5) Generar las gráficas correspondientes al Error y Métrica de desempeño (*Accuracy*) para la configuración de parámetros actual
6) Para cada conjunto de datos, entrenar y evaluar todas las configuraciones propuestas
7) Discutir los resultados obtenidos para todas las configuraciones en el conjunto de datos procesado

### 3.3. Normalización y Métrica de desempeño
#### 3.3.1. Normalización Z-score (Estandarización)

La **normalización Z-score**, también conocida como **estandarización**, es una técnica que transforma los valores de un conjunto de datos para que tengan una media igual a cero y una desviación estándar igual a uno.

Matemáticamente, se define como:

$$X_{i,norm} = \frac{X_i - \mu}{\sigma}$$

donde:
- $X_i$ es el valor original del atributo
- $\mu$ es la media de los valores del atributo
- $\sigma$ es la desviación estándar del atributo
- $X_{i,norm}$ es el valor normalizado del atributo

**Notas:**
- Cuando se normalizan los datos, la normalización se aplica al conjunto de datos _train_ y _test_
- Considerar que los datos están en formato de Numpy, por lo cual, se puede hacer uso del _broadcasting_ para simplificar las operaciones de normalización

#### 3.3.2. Exactitud (Accuracy)
La **exactitud**, o _accuracy_, mide la proporción de ejemplos correctamente clasificados con respecto al total de ejemplos evaluados. Se define como:

$$Accuracy = \frac{N_{\text{aciertos}}}{N_{\text{total}}}$$

donde:
- $N_{\text{aciertos}}$: número de predicciones correctas
- $N_{\text{total}}$: número total de ejemplos evaluados

### 3.4. Configuraciones por experimentar
Para cada conjunto de datos se debe medir el rendimiento de la red neuronal con cada una de las combinaciones de parámetros de la tabla siguiente.

**Tabla 2: Parámetros sugeridos para el entrenamiento del modelo**

| Parámetro | Valores sugeridos |
|-----------|-------------------|
| Neuronas en capa oculta | 2, 4, 8, 16, 32, 128 |
| Inicialización | Distribución Normal / Xavier |
| Normalización de datos | Sin normalizar / Normalización z-score |
| Learning rate | 0.01, 0.1, 0.5 |
| Batch size | 8, 16, 32, 64 |
| Función de activación | Sigmoide |
| Num. épocas (epochs) | 10,000 |
| Métrica de desempeño | Accuracy |
| Dataset | Iris, Breast Cancer, Wine |

**Registrar para cada experimento:**
- Gráfica del error vs. épocas
- Métrica de desempeño
- Generar tablas comparativas y/o gráficas de los resultados
- Comentarios analítico de los resultados

## 4. Entrega en PDF
1. Subir a classroom un reporte en formato PDF con los experimentos de todas las combinaciones indicadas en los parámetros de la red neuronal y datasets.

2. **Nombre del archivo del Reporte:** El formato del nombre del archivo que se debe subir será numero_de_practica_nombre_del_alumno sin espacios.
   - **Ejemplo:** 01_Practica_Juan_Perez_Perez.pdf

3. **Reporte:**
   - El reporte debe incluir una estructura: Introducción, desarrollo y conclusiones
   - Incluir como anexo el código comentado y modularizado
   - Incluir una descripción muy breve de la configuración que se está analizando
   - Presentar los resultados obtenidos en forma de tablas y gráficos, si aplica
   - Incluir como anexo el código de su implementación
   - Incluir una discusión sobre los resultados y las conclusiones de los experimentos

## Sugerencia para el reporte
- Incluir una tabla comparativa con las configuraciones probadas
- Graficar el error para cada configuración
- Discutir:
  - ¿Qué inicialización funcionó mejor?
  - ¿Cómo afecta el número de neuronas ocultas?
  - ¿Qué tasa de aprendizaje converge más rápido o es más estable?
  - ¿Qué diferencias se observan entre los datasets respecto a la cantidad de parámetros?
  - Otras

## Referencias

[1] Iris dataset, UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets/iris
[2] Breast Cancer Wisconsin dataset, UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
[3] Wine Recognition dataset, UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets/wine
[4] Scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
