import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime

# ==============================
# ⚙️ CONFIGURACIÓN INICIAL
# ==============================
st.set_page_config(page_title="Comparador de Bases de Datos Avanzado", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Comparador Interactivo de Bases de Datos")
st.markdown("Herramienta avanzada para la conciliación y el análisis exploratorio de dos conjuntos de datos (Excel/CSV).")

# Rutas de archivos por defecto
FILE1_PATH = "Military Expenditure.xlsx"
FILE2_PATH = "zomato_datos_limpios.xlsx"

# ==============================
# 🤖 DETECCIÓN AUTOMÁTICA DEL MEJOR MÉTODO DE IMPUTACIÓN
# ==============================
def detectar_metodo_imputacion(df, col):
    """
    Detecta el mejor método de imputación según el tipo y características de los datos.
    Retorna: 'median', 'mean', 'mode', 'forward_fill', 'group_median'
    """
    try:
        if df[col].dtype in ['object', 'category']:
            return 'mode'
        
        # Para datos numéricos, analizar distribución
        skewness = abs(df[col].skew())
        unique_ratio = df[col].nunique() / len(df[col].dropna())
        
        # Si hay muchos outliers (alta asimetría), usar mediana
        if skewness > 1:
            return 'median'
        # Si los datos son relativamente uniformes, usar media
        elif skewness < 0.5:
            return 'mean'
        # Por defecto, mediana (más robusta)
        else:
            return 'median'
    except:
        return 'median'

# ==============================
# 🧹 FUNCIÓN DE LIMPIEZA AVANZADA CON IMPUTACIÓN INTELIGENTE
# ==============================
@st.cache_data
def limpiar_bd1(file_path_or_upload, manual_header_row=None, metodo_imputacion='auto'):
    """
    Limpia la base de datos con imputación inteligente OPCIONAL por grupo (continente).
    Si metodo_imputacion es None, NO se imputan valores faltantes.
    """
    try:
        # 1. Determinar el índice de la fila de encabezado
        if manual_header_row is not None and manual_header_row >= 0:
            header_row = int(manual_header_row)
            if not isinstance(file_path_or_upload, str):
                file_path_or_upload.seek(0)
        else:
            header_row = 0

        # 2. Cargar el DataFrame
        if isinstance(file_path_or_upload, str):
            # Verificar que el archivo existe antes de intentar leerlo
            if not os.path.exists(file_path_or_upload):
                raise FileNotFoundError(f"El archivo no existe: {file_path_or_upload}")
            if not os.path.isfile(file_path_or_upload):
                raise ValueError(f"La ruta no es un archivo: {file_path_or_upload}")
            
            if file_path_or_upload.endswith(".csv"):
                df = pd.read_csv(file_path_or_upload, header=header_row)
            else:
                df = pd.read_excel(file_path_or_upload, header=header_row)
        else:
            if file_path_or_upload.name.endswith(".csv"):
                file_path_or_upload.seek(0)
                df = pd.read_csv(file_path_or_upload, header=header_row)
            else:
                file_path_or_upload.seek(0)
                df = pd.read_excel(file_path_or_upload, header=header_row)

        # 3. Limpieza básica
        df.dropna(axis=1, how="all", inplace=True) 
        df.dropna(axis=0, how="all", inplace=True) 
        df = df.loc[:, ~df.columns.astype(str).str.contains("Unnamed", case=False, na=False)]
        
        # Corrección de nombres de columnas
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
        )
        
        # 4. CORRECCIÓN DE DATOS ESPECÍFICA (ÁFRICA)
        paises_africa = [
            'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 
            'Cape Verde', 'Central African Republic', 'Chad', 'Comoros', 'Congo, Dem. Rep.', 
            'Congo, Rep.', 'Cote d\'Ivoire', 'Djibouti', 'Egypt', 'Equatorial Guinea', 
            'Eritrea', 'Ethiopia', 'Gabon', 'Gambia, The', 'Ghana', 'Guinea', 'Guinea-Bissau', 
            'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 
            'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 
            'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 
            'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 
            'Swaziland', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe'
        ]
        
        country_col_candidates = [col for col in df.columns if 'Country' in col or 'Code' in col]
        
        if country_col_candidates and 'Continent' in df.columns:
            country_col = country_col_candidates[0]
            df['Country_Name_Extracted'] = df[country_col].astype(str).str.split(' - ').str[-1].str.strip()
            df['Continent'] = df['Continent'].fillna('').astype(str).str.strip()
            mask = (df['Continent'] == '') | (df['Continent'].str.lower() == 'nan')
            df.loc[mask & df['Country_Name_Extracted'].isin(paises_africa), 'Continent'] = 'Africa'
            df.drop(columns=['Country_Name_Extracted'], inplace=True, errors='ignore')
        
        # 5. IMPUTACIÓN INTELIGENTE (OPCIONAL)
        num_cols = df.select_dtypes(include=['number']).columns
        imputation_log = {}
        
        # Si metodo_imputacion es None, NO imputar
        if metodo_imputacion is None:
            for col in num_cols:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    imputation_log[col] = {
                        'method': 'sin_imputar',
                        'imputed': 0,
                        'missing': missing_count
                    }
            return df, imputation_log
        
        # Continuar con imputación si está activada
        if 'Continent' in df.columns:
            # Imputación por grupo (continente) - más precisa para datos geográficos
            for col in num_cols:
                missing_before = df[col].isnull().sum()
                if missing_before > 0:
                    if metodo_imputacion == 'auto':
                        metodo = detectar_metodo_imputacion(df, col)
                    else:
                        metodo = metodo_imputacion
                    
                    # Imputar por mediana/media del continente
                    if metodo == 'median':
                        df[col] = df.groupby('Continent')[col].transform(
                            lambda x: x.fillna(x.median())
                        )
                    elif metodo == 'mean':
                        df[col] = df.groupby('Continent')[col].transform(
                            lambda x: x.fillna(x.mean())
                        )
                    
                    # Si aún quedan NaN (continentes sin datos), usar global
                    if df[col].isnull().sum() > 0:
                        global_value = df[col].median() if metodo == 'median' else df[col].mean()
                        df[col].fillna(global_value, inplace=True)
                    
                    missing_after = df[col].isnull().sum()
                    imputation_log[col] = {
                        'method': f'{metodo}_por_continente',
                        'imputed': missing_before - missing_after,
                        'missing': missing_after
                    }
        else:
            # Imputación global si no hay columna de agrupación
            for col in num_cols:
                missing_before = df[col].isnull().sum()
                if missing_before > 0:
                    if metodo_imputacion == 'auto':
                        metodo = detectar_metodo_imputacion(df, col)
                    else:
                        metodo = metodo_imputacion
                    
                    if metodo == 'median':
                        df[col].fillna(df[col].median(), inplace=True)
                    elif metodo == 'mean':
                        df[col].fillna(df[col].mean(), inplace=True)
                    
                    missing_after = df[col].isnull().sum()
                    imputation_log[col] = {
                        'method': metodo,
                        'imputed': missing_before - missing_after,
                        'missing': missing_after
                    }

        return df, imputation_log

    except FileNotFoundError as e:
        st.error(f"❌ Error: Archivo no encontrado. {e}")
        st.info("💡 Sugerencia: Verifica que los archivos estén en el repositorio o usa el modo 'Subir archivos'.")
        return None, {}
    except Exception as e:
        st.error(f"❌ Error al limpiar la BD1: {e}")
        st.info(f"💡 Tipo de error: {type(e).__name__}")
        return None, {}

# ==============================
# 📚 FUNCIÓN PARA CARGAR DATOS GENERAL
# ==============================
@st.cache_data
def cargar_datos(file):
    """Carga datos sin limpieza avanzada."""
    if file is None:
        return None
    try:
        if isinstance(file, str):
            # Verificar que el archivo existe antes de intentar leerlo
            if not os.path.exists(file):
                raise FileNotFoundError(f"El archivo no existe: {file}")
            if not os.path.isfile(file):
                raise ValueError(f"La ruta no es un archivo: {file}")
            
            if file.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
        else:
            if file.name.endswith(".csv"):
                file.seek(0)
                df = pd.read_csv(file)
            else:
                file.seek(0)
                df = pd.read_excel(file)
                
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
        )
        return df
    except FileNotFoundError as e:
        file_name = file if isinstance(file, str) else (file.name if hasattr(file, 'name') else 'archivo')
        st.error(f"❌ Error: Archivo no encontrado: {file_name}")
        st.info("💡 Sugerencia: Verifica que los archivos estén en el repositorio o usa el modo 'Subir archivos'.")
        return None
    except Exception as e:
        file_name = file if isinstance(file, str) else (file.name if hasattr(file, 'name') else 'archivo')
        st.error(f"❌ Error al cargar {file_name}: {e}")
        return None

# ==============================
# ⚙️ SIDEBAR: CARGA DE ARCHIVOS
# ==============================
st.sidebar.header("📁 Carga de Archivos")

# Función para verificar si un archivo es realmente accesible
def verificar_archivo(archivo_path):
    """Verifica que el archivo existe y es accesible"""
    try:
        if not os.path.exists(archivo_path):
            return False
        # Intentar abrir el archivo para verificar que es accesible
        if archivo_path.endswith(".xlsx"):
            # Solo verificar que existe, no cargarlo completamente
            return os.path.isfile(archivo_path) and os.access(archivo_path, os.R_OK)
        elif archivo_path.endswith(".csv"):
            return os.path.isfile(archivo_path) and os.access(archivo_path, os.R_OK)
        return False
    except Exception:
        return False

# Verificar primero si los archivos por defecto existen y son accesibles
archivo1_ok = verificar_archivo(FILE1_PATH)
archivo2_ok = verificar_archivo(FILE2_PATH)
archivos_disponibles = archivo1_ok and archivo2_ok

if archivos_disponibles:
    # Si los archivos existen, usar rutas por defecto automáticamente
    file1, file2 = FILE1_PATH, FILE2_PATH
    st.sidebar.success("✅ Usando archivos por defecto del repositorio")
    st.sidebar.info(f"📄 BD1: `{FILE1_PATH}`\n📄 BD2: `{FILE2_PATH}`")
    
    # Opción para cambiar a modo manual si lo desea
    usar_manual = st.sidebar.checkbox("📤 Cambiar a modo subir archivos", value=False)
    if usar_manual:
        file1 = st.sidebar.file_uploader("Sube el Primer Archivo (BD1 - Limpieza Auto)", type=["xlsx", "csv"])
        file2 = st.sidebar.file_uploader("Sube el Segundo Archivo (BD2)", type=["xlsx", "csv"])
else:
    # Si no existen, mostrar opción de subir
    st.sidebar.warning("⚠️ Archivos por defecto no encontrados o no accesibles")
    if not archivo1_ok:
        st.sidebar.error(f"❌ No se encontró: {FILE1_PATH}")
    if not archivo2_ok:
        st.sidebar.error(f"❌ No se encontró: {FILE2_PATH}")
    st.sidebar.info("Por favor, sube los archivos manualmente:")
    file1 = st.sidebar.file_uploader("Sube el Primer Archivo (BD1 - Limpieza Auto)", type=["xlsx", "csv"])
    file2 = st.sidebar.file_uploader("Sube el Segundo Archivo (BD2)", type=["xlsx", "csv"])

st.sidebar.markdown("---")
st.sidebar.header("🛠️ Opciones BD1 (Limpieza Avanzada)")
manual_header = st.sidebar.number_input(
    "Fila de Encabezado REAL (BD1)", 
    min_value=0, 
    value=3
)
manual_header_bd1 = int(manual_header)

# Toggle para imputación
activar_imputacion = st.sidebar.checkbox(
    "🔄 Imputar valores faltantes",
    value=False,
    help="Activar para rellenar valores faltantes. Desactivar para análisis con datos originales (transparencia total)."
)

if activar_imputacion:
    metodo_imputacion = st.sidebar.selectbox(
        "Método de Imputación:",
        ["auto", "median", "mean"],
        help="Auto detecta el mejor método. Median es robusto ante outliers. Mean para distribuciones normales."
    )
else:
    metodo_imputacion = None
    st.sidebar.info("💡 Valores faltantes se mantendrán como NaN para análisis transparente.")

# ==============================
# 📈 PROCESAMIENTO Y DASHBOARD
# ==============================
if file1 and file2:
    result = limpiar_bd1(file1, manual_header_bd1, metodo_imputacion)
    
    if result is not None:
        df1, imputation_log = result
        df2 = cargar_datos(file2)

        if df1 is not None and df2 is not None:
            # Mensaje de estado según imputación
            if activar_imputacion:
                st.success("✅ Archivos cargados. BD1 con imputación aplicada.")
            else:
                st.success("✅ Archivos cargados. BD1 sin imputación (datos originales).")
            
            # Mostrar log de imputación
            if imputation_log:
                with st.expander("📊 Detalles de Procesamiento de Valores Faltantes"):
                    imp_df = pd.DataFrame.from_dict(imputation_log, orient='index')
                    
                    if activar_imputacion:
                        st.info("🔄 **Imputación activada**: Los valores faltantes fueron rellenados usando el método seleccionado.")
                    else:
                        st.warning("⚠️ **Sin imputación**: Los valores faltantes se mantienen como NaN para análisis transparente.")
                    
                    st.dataframe(imp_df, width='stretch')
                    
                    # Resumen visual
                    if activar_imputacion:
                        total_imputed = imp_df['imputed'].sum()
                        st.metric("Total de valores imputados", f"{total_imputed:,}")
                    else:
                        total_missing = imp_df['missing'].sum()
                        st.metric("Total de valores faltantes (sin imputar)", f"{total_missing:,}")

            # Obtener columnas comunes y diferentes
            columnas_comunes = list(set(df1.columns).intersection(df2.columns))
            columnas_dif_bd1 = list(set(df1.columns) - set(df2.columns))
            columnas_dif_bd2 = list(set(df2.columns) - set(df1.columns))
            
            faltantes1 = df1.isnull().sum().sum() 
            faltantes2 = df2.isnull().sum().sum()
            dup1 = df1.duplicated().sum()
            dup2 = df2.duplicated().sum()

            # ==============================
            # TABS INTERACTIVOS
            # ==============================
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Resumen y Vistas Previas",
                "🧩 Estructura de Datos",
                "📊 Análisis Descriptivo & Correlación",
                "🌍 Análisis Geográfico y Temporal",
                "⚖️ Comparación Detallada"
            ])

            # --- TAB 1: RESUMEN Y VISTAS PREVIAS ---
            with tab1:
                st.header("📋 Resumen y Vistas Previas")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📘 Base de Datos 1")
                    status = " (Con Imputación)" if activar_imputacion else " (Datos Originales)"
                    st.info(f"Dimensiones: {df1.shape[0]:,} filas × {df1.shape[1]} columnas{status}")
                    
                    n_rows_bd1 = st.slider("Filas a mostrar BD1:", 10, min(500, len(df1)), 50)
                    st.dataframe(df1.head(n_rows_bd1), width='stretch')
                    
                with col2:
                    st.subheader("📗 Base de Datos 2")
                    st.info(f"Dimensiones: {df2.shape[0]:,} filas × {df2.shape[1]} columnas")
                    
                    n_rows_bd2 = st.slider("Filas a mostrar BD2:", 10, min(500, len(df2)), 50)
                    st.dataframe(df2.head(n_rows_bd2), width='stretch')

                st.header("🔍 Información General de Columna")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Columnas BD1:**")
                    df1_info = pd.DataFrame({
                        'Tipo': df1.dtypes.astype(str),
                        'No Nulos': df1.count(),
                        'Nulos': df1.isnull().sum(),
                        '% Nulos': (df1.isnull().sum() / len(df1) * 100).round(2)
                    })
                    st.dataframe(df1_info, width='stretch')
                    
                with c2:
                    st.markdown("**Columnas BD2:**")
                    df2_info = pd.DataFrame({
                        'Tipo': df2.dtypes.astype(str),
                        'No Nulos': df2.count(),
                        'Nulos': df2.isnull().sum(),
                        '% Nulos': (df2.isnull().sum() / len(df2) * 100).round(2)
                    })
                    st.dataframe(df2_info, width='stretch')

            # --- TAB 2: ESTRUCTURA DE DATOS ---
            with tab2:
                st.header("🧩 Comparación de Estructura y Tipos")

                st.subheader("1. Columnas Comunes y Exclusivas")
                
                col_metrics = st.columns(3)
                with col_metrics[0]:
                    st.metric("Columnas Comunes", len(columnas_comunes))
                with col_metrics[1]:
                    st.metric("Solo en BD1", len(columnas_dif_bd1))
                with col_metrics[2]:
                    st.metric("Solo en BD2", len(columnas_dif_bd2))
                
                if columnas_comunes:
                    st.success(f"**Columnas comunes:** {', '.join(columnas_comunes)}")
                
                c1, c2 = st.columns(2)
                with c1:
                    if columnas_dif_bd1:
                        st.error(f"**Solo en BD1:** {', '.join(columnas_dif_bd1[:10])}{'...' if len(columnas_dif_bd1) > 10 else ''}")
                with c2:
                    if columnas_dif_bd2:
                        st.warning(f"**Solo en BD2:** {', '.join(columnas_dif_bd2[:10])}{'...' if len(columnas_dif_bd2) > 10 else ''}")

                st.subheader("2. Comparación de Tipos de Datos (Dtype)")
                if columnas_comunes:
                    tipos_df = pd.DataFrame({
                        "Columna": columnas_comunes,
                        "BD1 Dtype": [str(df1[col].dtype) for col in columnas_comunes],
                        "BD2 Dtype": [str(df2[col].dtype) for col in columnas_comunes],
                    })
                    
                    def highlight_diff(row):
                        is_match = row['BD1 Dtype'] == row['BD2 Dtype']
                        return ['background-color: rgba(0, 128, 0, 0.1)' if is_match else 'background-color: rgba(255, 0, 0, 0.1)' for _ in row]

                    st.dataframe(tipos_df.style.apply(highlight_diff, axis=1), width='stretch')
                else:
                    st.info("⚠️ No hay columnas comunes para comparar tipos de datos.")

            # --- TAB 3: ANÁLISIS DESCRIPTIVO & CORRELACIÓN ---
            with tab3:
                st.header("📊 Análisis de Calidad de Datos y Correlación")

                st.subheader("1. Resumen de Calidad de Datos")
                
                qc1, qc2 = st.columns(2)
                with qc1:
                    graf_faltantes = pd.DataFrame({
                        "Base": ["BD1" + (" (Imputada)" if activar_imputacion else " (Original)"), "BD2"],
                        "Valores Faltantes": [faltantes1, faltantes2]
                    })
                    fig = px.bar(graf_faltantes, x="Base", y="Valores Faltantes",
                                 color="Base", title="Valores Faltantes (Null/NaN)",
                                 color_discrete_map={
                                     "BD1 (Imputada)": "#2ecc71",
                                     "BD1 (Original)": "#e67e22", 
                                     "BD2": "#e74c3c"
                                 })
                    st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
                    
                    if not activar_imputacion and faltantes1 > 0:
                        st.info(f"💡 BD1 tiene {faltantes1:,} valores faltantes. Activa la imputación en el sidebar para rellenarlos.")
                
                with qc2:
                    graf_dup = pd.DataFrame({
                        "Base": ["BD1", "BD2"],
                        "Duplicados": [dup1, dup2]
                    })
                    fig = px.bar(graf_dup, x="Base", y="Duplicados",
                                 color="Base", title="Registros Duplicados",
                                 color_discrete_map={"BD1": "#3498db", "BD2": "#f39c12"})
                    st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)

                st.subheader("2. Estadísticas Descriptivas")
                desc_col1, desc_col2 = st.columns(2)
                
                with desc_col1:
                    st.markdown("**BD1 - Estadísticas Numéricas:**")
                    st.dataframe(df1.describe().T, width='stretch')
                    
                with desc_col2:
                    st.markdown("**BD2 - Estadísticas Numéricas:**")
                    st.dataframe(df2.describe().T, width='stretch')

                st.subheader("3. Detección de Outliers (Box Plot)")
                num_cols1 = df1.select_dtypes(include="number").columns
                num_cols2 = df2.select_dtypes(include="number").columns
                comunes_num = list(set(num_cols1).intersection(num_cols2))

                if comunes_num:
                    col_num = st.selectbox("Selecciona una columna numérica común:", comunes_num)
                    df_temp1 = df1[[col_num]].copy().assign(Base="BD1")
                    df_temp2 = df2[[col_num]].copy().assign(Base="BD2")
                    df_comparativo = pd.concat([df_temp1, df_temp2])
                    fig = px.box(df_comparativo.dropna(), x="Base", y=col_num,
                                 color="Base", title=f"Box Plot Comparativo: {col_num}")
                    st.plotly_chart(fig, config={'displayModeBar': False}, use_container_width=True)
                else:
                    st.info("⚠️ No hay columnas numéricas comunes.")

                st.subheader("4. Análisis de Correlación (Heatmaps)")
                corr_col1, corr_col2 = st.columns(2)
                
                with corr_col1:
                    st.markdown("#### BD1: Correlación")
                    num_df1 = df1.select_dtypes(include=np.number)
                    if not num_df1.empty and len(num_df1.columns) > 1:
                        corr1 = num_df1.corr()
                        fig1 = px.imshow(
                            corr1, 
                            text_auto=".2f", 
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, 
                            zmax=1,
                            title="Correlación BD1"
                        )
                        st.plotly_chart(fig1, config={'displayModeBar': False}, use_container_width=True)
                    else:
                        st.info("Insuficientes columnas numéricas.")

                with corr_col2:
                    st.markdown("#### BD2: Correlación")
                    num_df2 = df2.select_dtypes(include=np.number)
                    if not num_df2.empty and len(num_df2.columns) > 1:
                        corr2 = num_df2.corr()
                        fig2 = px.imshow(
                            corr2, 
                            text_auto=".2f", 
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, 
                            zmax=1,
                            title="Correlación BD2"
                        )
                        st.plotly_chart(fig2, config={'displayModeBar': False}, use_container_width=True)
                    else:
                        st.info("Insuficientes columnas numéricas.")

            # --- TAB 4: ANÁLISIS GEOGRÁFICO Y TEMPORAL (NUEVO) ---
            with tab4:
                st.header("🌍 Análisis Geográfico y Temporal")
                
                # Análisis Geográfico
                if 'Continent' in df1.columns:
                    st.subheader("1. Distribución por Continente (BD1)")
                    
                    continent_counts = df1['Continent'].value_counts().reset_index()
                    continent_counts.columns = ['Continente', 'Cantidad']
                    
                    col_geo1, col_geo2 = st.columns(2)
                    
                    with col_geo1:
                        fig_pie = px.pie(continent_counts, values='Cantidad', names='Continente',
                                        title='Distribución de Países por Continente')
                        st.plotly_chart(fig_pie, config={'displayModeBar': False}, use_container_width=True)
                    
                    with col_geo2:
                        fig_bar = px.bar(continent_counts, x='Continente', y='Cantidad',
                                        color='Continente',
                                        title='Cantidad de Países por Continente')
                        st.plotly_chart(fig_bar, config={'displayModeBar': False}, use_container_width=True)
                    
                    # Gasto Militar por Continente
                    st.subheader("2. Análisis de Gasto Militar por Continente")
                    
                    year_cols = [col for col in df1.columns if col.isdigit()]
                    if year_cols:
                        selected_year = st.selectbox("Selecciona un año:", sorted(year_cols, reverse=True))
                        
                        if selected_year in df1.columns and 'Continent' in df1.columns:
                            continent_spending = df1.groupby('Continent')[selected_year].agg(['mean', 'sum', 'median']).reset_index()
                            continent_spending.columns = ['Continente', 'Promedio', 'Total', 'Mediana']
                            
                            col_spend1, col_spend2 = st.columns(2)
                            
                            with col_spend1:
                                fig_total = px.bar(continent_spending, x='Continente', y='Total',
                                                  color='Continente',
                                                  title=f'Gasto Militar Total por Continente ({selected_year})')
                                st.plotly_chart(fig_total, config={'displayModeBar': False}, use_container_width=True)
                            
                            with col_spend2:
                                fig_avg = px.bar(continent_spending, x='Continente', y='Promedio',
                                                color='Continente',
                                                title=f'Gasto Militar Promedio por Continente ({selected_year})')
                                st.plotly_chart(fig_avg, config={'displayModeBar': False}, use_container_width=True)
                
                # Análisis Temporal
                st.subheader("3. Tendencias Temporales de Gasto Militar")
                
                year_cols = [col for col in df1.columns if col.isdigit()]
                if year_cols and 'Continent' in df1.columns:
                    # Preparar datos para series temporales
                    continent_choice = st.multiselect(
                        "Selecciona continentes a comparar:",
                        df1['Continent'].unique(),
                        default=df1['Continent'].unique()[:3]
                    )
                    
                    if continent_choice:
                        temporal_data = []
                        for continent in continent_choice:
                            continent_df = df1[df1['Continent'] == continent]
                            for year in sorted(year_cols):
                                avg_spending = continent_df[year].mean()
                                temporal_data.append({
                                    'Año': int(year),
                                    'Continente': continent,
                                    'Gasto Promedio': avg_spending
                                })
                        
                        temporal_df = pd.DataFrame(temporal_data)
                        
                        fig_temporal = px.line(temporal_df, x='Año', y='Gasto Promedio',
                                              color='Continente',
                                              title='Evolución del Gasto Militar Promedio por Continente',
                                              markers=True)
                        fig_temporal.update_layout(height=500)
                        st.plotly_chart(fig_temporal, config={'displayModeBar': False}, use_container_width=True)
                        
                        # Tasa de crecimiento
                        st.subheader("4. Tasa de Crecimiento del Gasto Militar")
                        if len(year_cols) >= 2:
                            first_year = sorted(year_cols)[0]
                            last_year = sorted(year_cols)[-1]
                            
                            growth_data = []
                            for continent in df1['Continent'].unique():
                                continent_df = df1[df1['Continent'] == continent]
                                initial = continent_df[first_year].mean()
                                final = continent_df[last_year].mean()
                                
                                if initial > 0:
                                    growth_rate = ((final - initial) / initial) * 100
                                    growth_data.append({
                                        'Continente': continent,
                                        'Crecimiento (%)': growth_rate,
                                        'Gasto Inicial': initial,
                                        'Gasto Final': final
                                    })
                            
                            growth_df = pd.DataFrame(growth_data).sort_values('Crecimiento (%)', ascending=False)
                            
                            fig_growth = px.bar(growth_df, x='Continente', y='Crecimiento (%)',
                                               color='Crecimiento (%)',
                                               color_continuous_scale='RdYlGn',
                                               title=f'Tasa de Crecimiento del Gasto Militar ({first_year}-{last_year})')
                            st.plotly_chart(fig_growth, config={'displayModeBar': False}, use_container_width=True)
                            
                            st.dataframe(growth_df, width='stretch')

            # --- TAB 5: COMPARACIÓN DETALLADA ---
            with tab5:
                st.header("⚖️ Análisis Detallado por Columna")
                
                if not columnas_comunes:
                    st.info("ℹ️ Estas dos bases de datos no comparten nombres de columna exactos.")
                
                if columnas_comunes:
                    st.subheader("1. Distribución de Frecuencia (Top 15)")
                    col_sel = st.selectbox("Selecciona una columna común:", columnas_comunes)
                    
                    if col_sel:
                        st.write(f"**Valores únicos BD1:** {df1[col_sel].nunique():,} | **Valores únicos BD2:** {df2[col_sel].nunique():,}")

                        c1, c2 = st.columns(2)
                        with c1:
                            freq1 = df1[col_sel].value_counts().reset_index().head(15)
                            freq1.columns = [col_sel, 'Frecuencia']
                            st.dataframe(freq1, width='stretch')
                            fig1 = px.bar(freq1, x=col_sel, y="Frecuencia",
                                         title=f"Top 15 BD1 - {col_sel}",
                                         color='Frecuencia',
                                         color_continuous_scale='Blues')
                            st.plotly_chart(fig1, config={'displayModeBar': False}, use_container_width=True)
                            
                        with c2:
                            freq2 = df2[col_sel].value_counts().reset_index().head(15)
                            freq2.columns = [col_sel, 'Frecuencia']
                            st.dataframe(freq2, width='stretch')
                            fig2 = px.bar(freq2, x=col_sel, y="Frecuencia",
                                         title=f"Top 15 BD2 - {col_sel}",
                                         color='Frecuencia',
                                         color_continuous_scale='Reds')
                            st.plotly_chart(fig2, config={'displayModeBar': False}, use_container_width=True)

                    st.subheader("2. Valores Únicos Exclusivos por Base (Conciliación)")
                    col_diff_sel = st.selectbox("Selecciona una columna para conciliar:", columnas_comunes, key='diff_col')

                    if col_diff_sel:
                        set1 = set(df1[col_diff_sel].dropna().astype(str).unique())
                        set2 = set(df2[col_diff_sel].dropna().astype(str).unique())

                        solo_en_bd1 = sorted(list(set1 - set2))
                        solo_en_bd2 = sorted(list(set2 - set1))
                        comunes_valores = sorted(list(set1.intersection(set2)))

                        st.markdown(f"**Análisis de la columna: `{col_diff_sel}`**")

                        dc1, dc2, dc3 = st.columns(3)
                        with dc1:
                            st.metric("Valores Comunes", len(comunes_valores))
                        with dc2:
                            st.metric("Solo en BD1", len(solo_en_bd1))
                        with dc3:
                            st.metric("Solo en BD2", len(solo_en_bd2))

                        # Visualización de conjuntos
                        venn_data = pd.DataFrame({
                            'Categoría': ['Comunes', 'Solo BD1', 'Solo BD2'],
                            'Cantidad': [len(comunes_valores), len(solo_en_bd1), len(solo_en_bd2)]
                        })
                        fig_venn = px.bar(venn_data, x='Categoría', y='Cantidad',
                                         color='Categoría',
                                         title=f'Distribución de Valores Únicos - {col_diff_sel}')
                        st.plotly_chart(fig_venn, config={'displayModeBar': False}, use_container_width=True)

                        expander1 = st.expander(f"Valores que **Solo Existen en BD1** ({len(solo_en_bd1)})")
                        if solo_en_bd1:
                            expander1.code(', '.join(solo_en_bd1[:100]) + (', ...' if len(solo_en_bd1) > 100 else ''), language='text')
                        else:
                            expander1.info("No hay valores exclusivos en BD1")

                        expander2 = st.expander(f"Valores que **Solo Existen en BD2** ({len(solo_en_bd2)})")
                        if solo_en_bd2:
                            expander2.code(', '.join(solo_en_bd2[:100]) + (', ...' if len(solo_en_bd2) > 100 else ''), language='text')
                        else:
                            expander2.info("No hay valores exclusivos en BD2")

            # ==============================
            # 📤 DESCARGAR REPORTE
            # ==============================
            st.divider()
            st.header("📤 Descargar Reporte Comparativo")
            
            # Obtener nombres de archivo
            file1_name = file1 if isinstance(file1, str) else (file1.name if hasattr(file1, 'name') else 'Archivo 1')
            file2_name = file2 if isinstance(file2, str) else (file2.name if hasattr(file2, 'name') else 'Archivo 2')
            
            # Reporte mejorado
            reporte = f"""
╔═══════════════════════════════════════════════════════════════╗
║          REPORTE COMPARADOR DE BASES DE DATOS                 ║
╚═══════════════════════════════════════════════════════════════╝

📁 ARCHIVOS ANALIZADOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Base de Datos 1: {file1_name}
• Base de Datos 2: {file2_name}
• Fecha del Reporte: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Estado de Imputación: {'ACTIVADA' if activar_imputacion else 'DESACTIVADA (Datos Originales)'}

📊 DIMENSIONES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BD1: {df1.shape[0]:,} filas × {df1.shape[1]} columnas
BD2: {df2.shape[0]:,} filas × {df2.shape[1]} columnas

🧩 ESTRUCTURA DE COLUMNAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Columnas comunes: {len(columnas_comunes)}
  {', '.join(columnas_comunes) if columnas_comunes else 'Ninguna'}

• Solo en BD1 ({len(columnas_dif_bd1)}):
  {', '.join(columnas_dif_bd1[:20])}{'...' if len(columnas_dif_bd1) > 20 else ''}

• Solo en BD2 ({len(columnas_dif_bd2)}):
  {', '.join(columnas_dif_bd2[:20])}{'...' if len(columnas_dif_bd2) > 20 else ''}

🧹 CALIDAD DE DATOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BD1 {'(después de imputación)' if activar_imputacion else '(sin imputación - datos originales)'}:
  • Valores faltantes: {faltantes1}
  • Duplicados: {dup1}
  • Imputación: {'APLICADA - ' + metodo_imputacion if activar_imputacion else 'NO APLICADA'}

BD2:
  • Valores faltantes: {faltantes2}
  • Duplicados: {dup2}

📋 LOG DE PROCESAMIENTO BD1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            if imputation_log:
                if activar_imputacion:
                    reporte += "IMPUTACIÓN REALIZADA:\n"
                    for col, info in imputation_log.items():
                        reporte += f"• {col}: {info.get('imputed', 0)} valores imputados ({info.get('method', 'N/A')})\n"
                else:
                    reporte += "SIN IMPUTACIÓN - Valores faltantes preservados para análisis transparente:\n"
                    for col, info in imputation_log.items():
                        reporte += f"• {col}: {info.get('missing', 0)} valores faltantes preservados\n"
            else:
                reporte += "• No había valores faltantes en columnas numéricas\n"

            reporte += f"""
📈 ESTADÍSTICAS DESCRIPTIVAS BD1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{df1.describe().to_string()}

📈 ESTADÍSTICAS DESCRIPTIVAS BD2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{df2.describe().to_string()}

═══════════════════════════════════════════════════════════════
NOTA: Para análisis de correlación, tendencias temporales, mapas
geográficos y detalles de conciliación, consulte la aplicación
interactiva.
═══════════════════════════════════════════════════════════════
"""
            
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                st.download_button(
                    "📥 Descargar Reporte TXT",
                    data=reporte,
                    file_name=f"reporte_comparador_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col_download2:
                # Opción para descargar datos limpios
                csv1 = df1.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar BD1 Limpia (CSV)",
                    data=csv1,
                    file_name=f"bd1_limpia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    else:
        st.error("❌ Error al procesar los archivos. Verifica que sean válidos.")

else:
    st.info("📌 **Instrucciones:**\n\n1. Sube tus archivos o usa las rutas por defecto\n2. Ajusta el índice de encabezado (recomendado: 3 para BD1)\n3. Selecciona el método de imputación\n4. Explora los diferentes análisis en las pestañas")
    
    # Preview de características
    with st.expander("✨ Características del Comparador"):
        st.markdown("""
        ### 🎯 Funcionalidades Principales:
        
        **📋 Análisis de Datos:**
        - Comparación de estructura y tipos de datos
        - Detección automática de valores faltantes y duplicados
        - Estadísticas descriptivas completas
        
        **🧹 Limpieza Inteligente (OPCIONAL):**
        - ✅ **Opción de activar/desactivar imputación** para transparencia
        - Imputación por grupo (continente) para mayor precisión cuando está activa
        - Detección automática del mejor método de imputación
        - Corrección de datos geográficos (continentes)
        
        **💡 ¿Cuándo usar imputación?**
        - ✅ SÍ: Para análisis estadísticos, ML, o cuando los faltantes son aleatorios
        - ❌ NO: Cuando los faltantes tienen significado (datos clasificados, países inexistentes en ese periodo)
        
        **📊 Visualizaciones Avanzadas:**
        - Mapas de calor de correlación
        - Análisis de outliers con box plots
        - Tendencias temporales interactivas
        - Distribución geográfica
        
        **🌍 Análisis Geográfico y Temporal:**
        - Gasto militar por continente
        - Evolución temporal de gastos
        - Tasas de crecimiento comparativas
        
        **⚖️ Conciliación de Datos:**
        - Identificación de valores únicos y comunes
        - Análisis de diferencias entre bases
        - Visualización de conjuntos de datos
        
        **📤 Exportación:**
        - Reportes detallados en TXT
        - Datos limpios en CSV (con o sin imputación)
        - Métricas y estadísticas completas
        """)
    
    with st.expander("🎓 Guía de Uso Rápida"):
        st.markdown("""
        ### 📝 Pasos para usar el comparador:
        
        1. **Carga de Archivos:**
           - Selecciona "Usar rutas por defecto" o "Subir archivos"
           - Sube archivos Excel (.xlsx) o CSV
        
        2. **Configuración BD1:**
           - Ajusta la fila de encabezado (normalmente fila 3 para Military Expenditure)
           - **DECIDE: ¿Imputar o mantener datos originales?**
             - ☑️ Activar: Para análisis estadísticos completos
             - ☐ Desactivar: Para transparencia total (recomendado para reportes oficiales)
           - Si activas imputación, selecciona el método:
             - `auto`: Detecta automáticamente el mejor método
             - `median`: Robusto ante outliers (recomendado)
             - `mean`: Para distribuciones normales
        
        3. **Exploración:**
           - **Resumen**: Vista previa y información de columnas
           - **Estructura**: Compara tipos de datos y columnas
           - **Análisis Descriptivo**: Estadísticas, correlaciones y outliers
           - **Geográfico/Temporal**: Análisis específico de BD1 (gasto militar)
           - **Comparación**: Conciliación detallada de valores
        
        4. **Exportación:**
           - Descarga el reporte completo
           - Exporta los datos limpios en CSV (refleja tu elección de imputación)
        
        ### ⚠️ Importante sobre Imputación:
        
        **Valores faltantes en Gasto Militar pueden significar:**
        - 🚫 País no existía en ese periodo
        - 🔒 Datos clasificados/secretos de defensa
        - ⚔️ Conflictos que impidieron reportar
        - 🏝️ Países sin ejército permanente
        
        **Recomendación:** Para análisis exploratorio o ML → Activa imputación
        Para reportes oficiales o investigación académica → Desactiva imputación
        """)