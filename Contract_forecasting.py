import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xg
import scipy.stats as stats
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def crear_placeholder():
    placeholder = st.sidebar.empty()  # Crear un placeholder vacío
    return placeholder

@st.cache_data
def carga_historicos():
    datosHis = pd.read_csv("data/Contratos_antes_SICEP.csv", sep=',', header=0, low_memory=False)
    datosHis = datosHis.rename(columns={'BASE': 'PERIODO_ADJ'})
    datosHis.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1, inplace=True)
    datosHis = datosHis[['AGENTE_COMPRADOR', 'AGENTE_VENDEDOR', 'PERIODO_ADJ','CONTRATO', 'FECHA', 'KW', 'PRECIO']]
    return datosHis

@st.cache_data
def carga_archivos(archivo, nombre):
    alerta_kw = crear_placeholder()
    alerta_precios = crear_placeholder()
    required_columns = ['CONTRATO','ADJUDICACION','VALOR','FECHA']
    datos = ''
    if archivo is not None:
        datos = pd.read_csv(archivo, sep=';', header=0, low_memory=False)
        datos = datos[datos['ADJUDICACION'] != 'ANTES DE SICEP']
        datos['CONTRATO'] = datos['CONTRATO'].apply(lambda x: str(int(float(x))) if x.endswith('.0') and x.replace('.0', '').isdigit() else x)
        missing_columns = [col for col in required_columns if col not in datos.columns]
        if "kw" in nombre.lower(): 
            if missing_columns:
                alerta_kw.warning(f":no_entry: El archivo cargado no es correcto. Faltan las siguientes columnas: {', '.join(missing_columns)}")
                return pd.DataFrame()  # Devuelve un DataFrame vacío si hay columnas faltantes
            else:
                datos.drop(['TIPO','CODIGO_CONVOCATORIA','CONCEPTO','CODIGO_COMERCIALIZADOR','ADJUDICACION_AÑO'],axis=1, inplace=True)
                alerta_kw.success(":heavy_check_mark: Base de datos de energía cargada correctamente")
        if "precios" in nombre.lower():
            if missing_columns:
                alerta_precios.warning(f":no_entry: El archivo cargado no es correcto. Faltan las siguientes columnas: {', '.join(missing_columns)}")
                return pd.DataFrame()  # Devuelve un DataFrame vacío si hay columnas faltantes
            else:
                datos.drop(['TIPO','IPP_BASE','PERIODO_BASE','CODIGO_CONVOCATORIA',
                        'CODIGO_COMERCIALIZADOR','CONCEPTO'],axis=1, inplace=True)
                alerta_precios.success(":heavy_check_mark: Base de datos de precio cargada correctamente")
    else:
        return pd.DataFrame()
    return datos

@st.cache_data
def gen_BD_Contratos(kW,precios):
    Contratos_DF = pd.merge(kW, precios, on=['CONTRATO','FECHA','AGENTE_VENDEDOR','AGENTE_COMPRADOR'], how='inner')
    Contratos_DF.drop('ADJUDICACION_x',axis=1,inplace=True)
    Contratos_DF = Contratos_DF.rename(columns={'VALOR_x': 'KW', 'VALOR_y': 'PRECIO', 'ADJUDICACION_y':'PERIODO_ADJ'})
    Contratos_DF = Contratos_DF[['AGENTE_COMPRADOR', 'AGENTE_VENDEDOR', 'PERIODO_ADJ','CONTRATO', 'FECHA', 'KW', 'PRECIO']]
    return Contratos_DF

@st.cache_data
def unif_BD_contratos(historicos,BD_Carga):
    Contratos_DF_f1 = pd.concat([historicos, BD_Carga], ignore_index=True)
    Contratos_DF_f2 = Contratos_DF_f1[(Contratos_DF_f1['KW'] > 0) & (Contratos_DF_f1['PRECIO'] > 0)]
    Contratos_DF_f2.loc[:,'FECHA'] = pd.to_datetime(Contratos_DF_f2['FECHA'])
    Contratos_DF_f2.loc[:,'PERIODO_ADJ'] = pd.to_datetime(Contratos_DF_f2['PERIODO_ADJ'])
    return Contratos_DF_f2

@st.cache_data
def preparar_BD_IA(data):
    result = data.groupby('CONTRATO').agg(
      AÑO_FIRMA = ('PERIODO_ADJ',lambda x: x.min().year),
      AÑO_EJECUCION = ('FECHA',lambda x: x.min().year),
      GW_TOTAL = ('KW', lambda x: x.sum() / 1e9),
      PLAZO = ('FECHA', lambda x: int(round(abs(((x.min() - data.loc[x.index,'PERIODO_ADJ'].min()).days) / 365.25 )))),
      DURACION = ('FECHA', lambda x: int(round(abs((x.max() - x.min()).days) / 365.25))),
      PRECIO_CONTRATO = ('PRECIO', lambda x: round((x * data.loc[x.index, 'KW']).sum() / data.loc[x.index, 'KW'].sum(), 2))
    ).reset_index()
    #result.drop('CONTRATO',axis=1,inplace=True)
    return result

@st.cache_data
def entrenar(datos,alpha):
    X, y = datos[['GW_TOTAL','PLAZO','DURACION']], datos['PRECIO_CONTRATO']
    xgb_r = xg.XGBRegressor(objective ='reg:squarederror',tree_method="hist",
                        eval_metric=mean_absolute_percentage_error,n_estimators = 100,
                        seed = 42, learning_rate=0.05) 
    xgb_r.fit(X, y)
    pred_train = xgb_r.predict(X) 
    residuos = pred_train - y.values 
    std_error = np.std(residuos, ddof=1)
    n = len(X)
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    return xgb_r, t_critical, std_error

def pronostico(modelo,datos,t_crit,std):
    pred_test = modelo.predict(datos)
    pred_test_low = pred_test[0] - t_crit * std
    pred_test_high = pred_test[0] + t_crit * std
    return pred_test_high, pred_test_low

def main():
    font = """
      <style>
          @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;700&display=swap');

          body, h1, h2, h3, h4, h5, h6, p, .stDataFrame, .css-1y0t9o9, .stButton>button, .css-1wa3eu0, .css-10jvk68, .css-1y0t9o9 {
              font-family: 'Prompt', sans-serif !important;
          }

      </style>
    """
    st.set_page_config(page_title="Pronóstico Contratos Vatia",page_icon="images/icon.png",layout="wide")
    st.markdown(font, unsafe_allow_html=True)
    est_pron = False
    st.sidebar.header("Pronósticos Contratos")
    st.markdown('<br>', unsafe_allow_html=True)
    st.sidebar.write('**Notificaciones**')
    st.markdown('<br>', unsafe_allow_html=True)
    with st.expander('📤 **Cargar históricos de contratos**',expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            arch1 = st.file_uploader(':zap: Energia', type='csv', key='kw_up')
            if arch1 is not None:
                kw = carga_archivos(arch1, arch1.name)
            else:
                kw = pd.DataFrame()  # Valor predeterminado si no hay archivo

            if kw.empty:
                st.sidebar.warning(":warning: Por favor, sube un archivo válido de energía contratada.")

        with col2:
            arch2 = st.file_uploader(':heavy_dollar_sign: Precios', type='csv', key='precios_up')
            if arch2 is not None:
                precios = carga_archivos(arch2, arch2.name)
            else:
                precios = pd.DataFrame()  # Valor predeterminado si no hay archivo

            if precios.empty:
                st.sidebar.warning(":warning: Por favor, sube un archivo válido de precios de contrato.")

    if not kw.empty and not precios.empty:
        if 'Contratos' not in st.session_state:
            BD_con = gen_BD_Contratos(kw,precios)
            datosHis = carga_historicos()
            BD_prev = unif_BD_contratos(datosHis,BD_con)
            contratos = preparar_BD_IA(BD_prev)
            st.session_state['Contratos'] = contratos
        contratos_f2 = st.session_state['Contratos']
        modelo, t_c, std = entrenar(contratos_f2,0.9)
        st.markdown('<br>', unsafe_allow_html=True)
        contenedor = st.sidebar.container(border=True)
        contenedor.write('**Parámetros del Contrato**') 
        plazo_ejec = contenedor.slider(':calendar: Plazo de ejecución (Años)', 0, 15, 15)
        duracion = contenedor.slider(':calendar: Duración (Años)', 0, 15, 15)
        energia = contenedor.number_input(":zap: Energía a contratar (GWh)",key='precio_input', min_value=0.0000,step=0.0001,format="%.4f")
        if energia > 0:
            pron_up, pron_down = pronostico(modelo,pd.DataFrame([[energia,plazo_ejec,duracion]],columns=['GW_TOTAL','PLAZO','DURACION']),t_c,std)
        else:
            pron_up, pron_down = round(0.00,2), round(0.00,2)

        with st.expander('📊 **Información de Pronóstico de Contrato:**',expanded=True):
            col3, col4 = st.columns([3,3])
            col4.metric(':arrow_up::heavy_dollar_sign: Precio máximo de firma (COP/kWh)', round(pron_up,2))
            col3.metric(':arrow_down::heavy_dollar_sign: Precio mínimo de firma (COP/kWh)', round(pron_down,2))

if __name__ == "__main__":
    main()
