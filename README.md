# 🚀 Comparador Interactivo de Bases de Datos

Herramienta avanzada para la conciliación y el análisis exploratorio de dos conjuntos de datos (Excel/CSV) desarrollada con Streamlit.

## 📋 Características

- **Comparación de Estructura**: Análisis de columnas comunes y exclusivas
- **Limpieza Inteligente**: Imputación opcional de valores faltantes con detección automática del mejor método
- **Análisis Descriptivo**: Estadísticas, correlaciones y detección de outliers
- **Visualizaciones Interactivas**: Gráficos de correlación, box plots, análisis temporal y geográfico
- **Conciliación de Datos**: Identificación de valores únicos y comunes entre bases de datos
- **Exportación**: Reportes detallados y datos limpios en CSV

## 🛠️ Instalación

1. Clona este repositorio:
```bash
git clone <url-del-repositorio>
cd comparacion
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 🚀 Uso

Ejecuta la aplicación Streamlit:

```bash
streamlit run dashboard.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📁 Archivos Requeridos

La aplicación puede trabajar con:
- Archivos Excel (`.xlsx`)
- Archivos CSV (`.csv`)

Por defecto, busca los siguientes archivos:
- `Military Expenditure.xlsx` (BD1)
- `zomato_datos_limpios.xlsx` (BD2)

Puedes subir tus propios archivos desde la interfaz.

## 📊 Funcionalidades Principales

### 1. Análisis de Datos
- Comparación de estructura y tipos de datos
- Detección automática de valores faltantes y duplicados
- Estadísticas descriptivas completas

### 2. Limpieza Inteligente (Opcional)
- Opción de activar/desactivar imputación para transparencia
- Imputación por grupo (continente) para mayor precisión
- Detección automática del mejor método de imputación
- Corrección de datos geográficos

### 3. Visualizaciones Avanzadas
- Mapas de calor de correlación
- Análisis de outliers con box plots
- Tendencias temporales interactivas
- Distribución geográfica

### 4. Análisis Geográfico y Temporal
- Gasto militar por continente
- Evolución temporal de gastos
- Tasas de crecimiento comparativas

### 5. Conciliación de Datos
- Identificación de valores únicos y comunes
- Análisis de diferencias entre bases
- Visualización de conjuntos de datos

## 📤 Exportación

- Reportes detallados en formato TXT
- Datos limpios en CSV (con o sin imputación según tu elección)
- Métricas y estadísticas completas

## 🔧 Tecnologías Utilizadas

- **Streamlit**: Framework para aplicaciones web interactivas
- **Pandas**: Manipulación y análisis de datos
- **Plotly**: Visualizaciones interactivas
- **NumPy**: Operaciones numéricas

## 📝 Notas

- La imputación de valores faltantes es **opcional** y puede ser activada/desactivada desde el sidebar
- Para análisis exploratorio o ML, se recomienda activar la imputación
- Para reportes oficiales o investigación académica, se recomienda mantener los datos originales (desactivar imputación)

## 📄 Licencia

Este proyecto está disponible para uso educativo y de investigación.

