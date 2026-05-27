# 🤖 Machine Learning con Python — Ejercicios de Máster

Colección de ejercicios prácticos de Machine Learning desarrollados durante el máster, utilizando `scikit-learn` para construir pipelines end-to-end completos: desde la limpieza de datos hasta el tuneado de hiperparámetros.

---

## 📋 Descripción General

Este repositorio contiene dos ejercicios independientes que cubren el ciclo completo de un proyecto de Machine Learning supervisado:

1. **Ejercicio 1** — Clasificación sobre el dataset Titanic
2. **Ejercicio 2** — Clasificación sobre el dataset KDD Cup 99 (detección de intrusiones de red)

Ambos ejercicios siguen una metodología común: preprocesamiento de datos, comparación de múltiples algoritmos mediante validación cruzada, y optimización del mejor modelo con `GridSearchCV`.

---

## 📁 Estructura del Repositorio

```
├── ejercicio1.ipynb          # Notebook: Titanic — clasificación y tuneado con Random Forest
├── ejercicio2.ipynb          # Notebook: KDD Cup 99 — modelos, PCA y Random Forest + OHE
├── requirements.txt          # Dependencias del proyecto
└── README.md
```

---

## 🗂️ Ejercicios

### Ejercicio 1 — Clasificación: Titanic

**Dataset:** [`titanic-2.csv`](https://raw.githubusercontent.com/eduardofc/data/main/titanic-2.csv)

**Objetivo:** Predecir la supervivencia de los pasajeros del Titanic construyendo un pipeline de ML completo con scikit-learn.

| Parte | Descripción |
|-------|-------------|
| **Parte 0** | Inicialización — librerías, semilla de reproducibilidad |
| **Parte 1** | Lectura y transformación de datos (feature engineering, imputación con KNN, escalado) |
| **Parte 2** | Selección de modelo — comparación de algoritmos mediante validación cruzada |
| **Parte 3** | Tuneado del modelo con `GridSearchCV` sobre Random Forest |

**Algoritmos evaluados:**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Linear Discriminant Analysis (LDA)
- Gaussian Naive Bayes
- Random Forest ✅ *(modelo final)*
- Extra Trees
- SVM

**Técnicas de preprocesamiento:**
- Ingeniería de características (`hasCabin`)
- Imputación con `KNNImputer` (5 vecinos)
- `ColumnTransformer` con `OneHotEncoder` y `RobustScaler`
- Pipelines con `sklearn.pipeline.Pipeline`
- Validación cruzada con `StratifiedKFold`

---

### Ejercicio 2 — Clasificación: KDD Cup 99

**Dataset:** [`kddcup99.csv`](https://raw.githubusercontent.com/eduardofc/data/main/kddcup99.csv)

**Objetivo:** Clasificar conexiones de red como normales o ataques (detección de intrusiones), explorando transformaciones progresivas y reducción de dimensionalidad.

| Parte | Descripción |
|-------|-------------|
| **Parte 0** | Inicialización |
| **Parte 1** | Lectura y transformación de datos |
| **Parte 2** | Evaluación de múltiples modelos en 6 bloques progresivos |

**Bloques de modelado:**

- **Modelos 1** — Benchmark base con validación cruzada
- **Modelos 2** — Pipelines con transformaciones adicionales
- **Modelos 3** — Variación de estrategia de transformación
- **Modelos 4** — Exploración de PCA (n_components = 2, 3, 4, 5) en bucle por algoritmo
- **Modelos 5** — Random Forest con `OneHotEncoder` sobre variables categóricas, evaluación train/test y matriz de confusión
- **Modelos 6** — Tuneado final con `GridSearchCV`, split train/test 80-20

---

## 🛠️ Tecnologías y Librerías

| Librería | Versión | Uso |
|----------|---------|-----|
| Python | 3.9 | Lenguaje base |
| scikit-learn | 1.2.1 | Modelos, pipelines, métricas |
| pandas | 1.5.3 | Manipulación de datos |
| numpy | 1.24.1 | Operaciones numéricas |
| matplotlib | 3.6.3 | Visualización |
| seaborn | 0.12.2 | Visualización estadística |

---

## ⚙️ Instalación y Uso

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/ml-python-master.git
cd ml-python-master
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python3.9 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Ejecutar los notebooks

```bash
jupyter notebook
```

Abrir `ejercicio1.ipynb` o `ejercicio2.ipynb` y ejecutar todas las celdas con **Cell → Run All Cells**.

---

## 📦 requirements.txt

```
scikit-learn==1.2.1
pandas==1.5.3
numpy==1.24.1
matplotlib==3.6.3
seaborn==0.12.2
jupyter
notebook
```

---

## 📊 Resultados Clave

| Ejercicio | Dataset | Mejor Modelo | Métrica |
|-----------|---------|-------------|---------|
| Ejercicio 1 | Titanic | Random Forest + GridSearchCV | Accuracy (CV) |
| Ejercicio 2 | KDD Cup 99 | Random Forest + OHE + GridSearchCV | Accuracy (CV + Test) |

---

## 🔑 Conceptos Cubiertos

- Pipelines end-to-end con `sklearn.pipeline.Pipeline`
- Preprocesamiento heterogéneo con `ColumnTransformer`
- Imputación avanzada (`KNNImputer`, `SimpleImputer`)
- Escalado de features (`StandardScaler`, `RobustScaler`, `MinMaxScaler`)
- Codificación de variables categóricas (`OneHotEncoder`, `LabelEncoder`)
- Selección de features (`SelectKBest`, `RFE`)
- Reducción de dimensionalidad (`PCA`)
- Validación cruzada (`StratifiedKFold`, `KFold`, `ShuffleSplit`)
- Optimización de hiperparámetros (`GridSearchCV`, `RandomizedSearchCV`)
- Evaluación de modelos (accuracy, F1-score, matriz de confusión)

---

## 📝 Notas

- Los datasets se cargan directamente desde GitHub, no es necesario descargarlos.
- La semilla de reproducibilidad es `seed = 99` en ambos ejercicios.
- Cada notebook incluye celdas de auto-evaluación que verifican la corrección de las soluciones.

---

## 👤 Autor

**Skarleth M Motiño Flores** — Proyecto de máster en Machine Learning con Python
