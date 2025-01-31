import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xg
import scipy.stats as stats
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime

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
def carga_FactorPSI():
  datosPSI = pd.read_csv("data/FactorPSI.csv", sep=';', header=0, low_memory=False)
  return datosPSI

def carga_archivos(archivo, nombre, _alerta):
    required_columns = ['CONTRATO','ADJUDICACION','VALOR','FECHA']
    datos = ''
    if archivo is not None:
        datos = pd.read_csv(archivo, sep=';', header=0, low_memory=False)
        missing_columns = [col for col in required_columns if col not in datos.columns]    
        if not missing_columns:
            datos = datos[datos['ADJUDICACION'] != 'ANTES DE SICEP']
            datos['CONTRATO'] = datos['CONTRATO'].apply(lambda x: str(int(float(x))) if x.endswith('.0') and x.replace('.0', '').isdigit() else x)
            if "kw" in nombre.lower():
                datos.drop(['TIPO','CODIGO_CONVOCATORIA','CONCEPTO','CODIGO_COMERCIALIZADOR','ADJUDICACION_AÑO'],axis=1, inplace=True)
                _alerta.success(":heavy_check_mark: :zap: Base de datos de energía cargada correctamente")
            if "precios" in nombre.lower():
                datos.drop(['TIPO','IPP_BASE','PERIODO_BASE','CODIGO_CONVOCATORIA',
                        'CODIGO_COMERCIALIZADOR','CONCEPTO'],axis=1, inplace=True)
                _alerta.success(":heavy_check_mark: :heavy_dollar_sign: Base de datos de precio cargada correctamente")
        else:
            _alerta.warning(f":no_entry: El archivo cargado no es correcto. Faltan las siguientes columnas: {', '.join(missing_columns)}")
            return pd.DataFrame()

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
      MES_FIRMA = ('PERIODO_ADJ',lambda x: x.min().month),
      AÑO_EJECUCION = ('FECHA',lambda x: x.min().year),
      GW_TOTAL = ('KW', lambda x: x.sum() / 1e9),
      PLAZO = ('FECHA', lambda x: int(round(abs(((x.min() - data.loc[x.index,'PERIODO_ADJ'].min()).days) / 365.25 )))),
      DURACION = ('FECHA', lambda x: int(round(abs((x.max() - x.min()).days) / 365.25))),
      PRECIO_CONTRATO = ('PRECIO', lambda x: round((x * data.loc[x.index, 'KW']).sum() / data.loc[x.index, 'KW'].sum(), 2))
    ).reset_index()
    return result

def transfer_PSI_columns(df1, df2):
    df1_relevant = df1.rename(columns={
        'AÑO': 'AÑO_FIRMA',
        'MES': 'MES_FIRMA'
    })
    merged_df = pd.merge(df2, df1_relevant, on=['AÑO_FIRMA', 'MES_FIRMA'],how='left')
    st.write(merged_df)
    return merged_df

@st.cache_data
def entrenar(datos,alpha):
    X, y = datos[['GW_TOTAL','PLAZO','DURACION','APORTES', 'VOLUMEN_UTIL', 'PROM_PRECIO_BOLSA']], datos['PRECIO_CONTRATO']
    xgb_r = xg.XGBRegressor(objective ='reg:squarederror',tree_method="hist",
                        eval_metric=mean_absolute_percentage_error,n_estimators = 100,
                        seed = 42, learning_rate=0.023)
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

def totales(datos, modelo, t_c, std, energia, duracion, aportes, vol_util, PPB):
    resultados = np.zeros((8, 3))
    año_actual = datetime.now().year
    for plazo in range(8):
        fore_up, fore_down = pronostico(modelo,pd.DataFrame([[energia, plazo, duracion, aportes, vol_util, PPB]],
                                                            columns=['GW_TOTAL','PLAZO','DURACION','APORTES', 'VOLUMEN_UTIL', 'PROM_PRECIO_BOLSA']),t_c,std)
        resultados[plazo, 0], resultados[plazo, 1], resultados[plazo, 2] = año_actual + plazo, fore_up, fore_down
    return pd.DataFrame(resultados, columns=['ds', 'p_max', 'p_min'])

def graficar(resultados):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resultados['ds'], y=resultados['p_max'], mode='lines+markers', marker=dict(size=10,color='rgba (174, 234, 0, 1)', opacity=0.8),
                             line=dict(width=3, color='rgba (174, 234, 0, 1)', dash='dash'), name='', showlegend=False))
    fig.add_trace(go.Scatter(x=resultados['ds'], y=resultados['p_min'], mode='lines+markers', marker=dict(size=10,color='rgba (174, 234, 0, 1)', opacity=0.8),
                             fill='tonexty', line=dict(width=3, color='rgba (174, 234, 0, 1)', dash='dash'), name='Precio Contratos', fillcolor='rgba(9, 90, 129, 0.5)'))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='white', gridcolor='lightgray', mirror=False, tickfont_color='white',
                     title_text='<b>Año</b>', titlefont_size=18, tickformat="%b %Y", tickfont_size=16, title_font_color='white',)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='white', gridcolor='lightgray', mirror=False, tickfont_color='white',
                     title_text='<b>Precio (COP/kWh)</b>', titlefont_size=18, tickfont_size=16, title_font_color='white',
                     tickformat='.2f')
    fig.update_traces(hovertemplate='<b>📆 Año:</b> %{x}<br><b>💲Precio Unitario Contrato:</b> %{y:.2f} COP/kWh<extra></extra>')
    fig.update_layout(title='', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family="Prompt", color='white'),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.98,
                                  font_size=16, font_color='white'),
                      showlegend=True, margin=dict(l=50, r=50, t=30, b=50))
    return fig

def estilo():
     st.markdown(
     """
     <style type=text/css>
            @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;700&display=swap');

            body, h1, h2, h3, h4, h5, h6, p, .stDataFrame, .css-1y0t9o9, .stButton>button, .css-1wa3eu0, .css-10jvk68, .css-1y0t9o9 .stMetricValue {
            font-family: 'Prompt', sans-serif !important;
            }

            div[data-testid="stAppViewBlockContainer"]{
                padding-top:0px;
            }

            body {
                margin-top: 0 !important;
            }

            div[data-testid="stVerticalBlock"] {
                margin-top: 0 !important;
            }

            [data-testid="stSidebar"] > div:first-child {
                display: flex;
                justify-content: flex-start;
                align-items: flex-start;
                flex-direction: column;
            }

            div[data-testid="stMetric"] {
                background-color: rgba(9, 90, 129, 0.5);
                border-radius: 20px;         /* Esquinas redondeadas */
                padding: 15px;              /* Añadir algo de padding */
                color: white;               /* Cambiar color del texto a blanco */
                font-family: 'Prompt', sans-serif !important;
            }

            hr:nth-of-type(1) {
                width: 95% !important;  
            }

            div[data-testid="stSelectbox"], div[data-testid="stAlert"] {
                border-radius: 20px;         /* Esquinas redondeadas */
                padding: 0px;              /* Añadir algo de padding */
                color: white;               /* Cambiar color del texto a blanco */
                font-family: 'Prompt', sans-serif !important;
                width: 95% !important;
            }

            [data-testid="stSidebar"] img {
                margin-top: -70px  !important; /* Ajustar según el espacio requerido */
                margin-left: 0px;
            }
            
            .css-1v0mbdj {
                margin-top: 0px !important;
            }

            .st-key-cont-load, .st-key-cont-result, .st-key-cont-result-2, .st-key-cont-maestro{
                background-color:#396425ff !important;
                padding: 1.5em;
                border-radius: 20px;
            }

            h3, label[data-testid="stMetricLabel"] p{
                font-size: 1.3em !important;
            }

            div[data-testid="stMetricDelta"] {
                font-size: 1.3em;
            }

            div[data-testid="stMetricValue"]{
                font-size:2.8em;
            }

            button[data-baseweb="tab"] p{
                font-size:18px!important;
                font-weight: bold;
            }
      </style>
      """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Pronóstico Contratos Vatia",page_icon="images/icon.png",layout="wide")
    estilo()
    color_dinamico = "#aeea00" #Verde Manzana
    est_pron = False
    st.sidebar.image("images/LogoVatia.png",caption="",use_container_width=True)
    st.sidebar.divider()
    st.header("Pronósticos Contratos")
    st.markdown('<br>', unsafe_allow_html=True)
    st.sidebar.write('**🚨 Notificaciones**')
    ph_alerta_1 = crear_placeholder()
    ph_alerta_2 = crear_placeholder() 
    st.markdown('<br>', unsafe_allow_html=True)
    st.sidebar.divider()
    st.sidebar.write('**📤📋 Cargar históricos de contratos**')
    ph_carga_1 = crear_placeholder()
    ph_carga_2 = crear_placeholder() 
    arch1 = ph_carga_1.file_uploader(':zap: Energía', type='csv', key='kw_up')
    arch2 = ph_carga_2.file_uploader(':heavy_dollar_sign: Precios', type='csv', key='precios_up')
    
    if arch1 is not None:
        kw = carga_archivos(arch1, arch1.name, ph_alerta_1)
    else:
        kw = pd.DataFrame()  # Valor predeterminado si no hay archivo

    if kw.empty:
        ph_alerta_1.warning(":warning: Por favor, sube un archivo válido de energía contratada.")

    if arch2 is not None:
        precios = carga_archivos(arch2, arch2.name, ph_alerta_2)
    else:
        precios = pd.DataFrame()  # Valor predeterminado si no hay archivo

    if precios.empty:
        ph_alerta_2.warning(":warning: Por favor, sube un archivo válido de precios de contrato.")

    if not kw.empty and not precios.empty:
        if 'Contratos' not in st.session_state:
            BD_con = gen_BD_Contratos(kw,precios)
            datosHis = carga_historicos()
            BD_prev = unif_BD_contratos(datosHis,BD_con)
            contratos = preparar_BD_IA(BD_prev)
            BD_PSI = carga_FactorPSI()
            contratos_PSI = transfer_PSI_columns(BD_PSI,contratos)
            st.session_state['Contratos'] = contratos_PSI
        contratos_f2 = st.session_state['Contratos']
        contenedor = st.container(key='cont-maestro')
        alpha_lit = contenedor.selectbox(':dart: Rango de Precisión', ['Alto','Medio','Bajo'],key='alpha-sel')
        st.markdown('<br>', unsafe_allow_html=True)
        if alpha_lit == 'Alto':
            alpha = 0.85
        elif alpha_lit == 'Medio':
            alpha = 0.7
        elif alpha_lit == 'Bajo':
            alpha = 0.5
        modelo, t_c, std = entrenar(contratos_f2,alpha)
        tab1, tab2 = st.tabs(["📈 Pronóstico de MC","📊 Valoración Precio de Contratos"])
        with tab2.container(key='cont-result'):
            mensaje = '📋 Condiciones del Contrato'
            st.write(f'<p style="color:{color_dinamico}; font-size:18px; font-weight:bold">{mensaje}</p>', unsafe_allow_html=True)
            col1, col2 = st.columns([3,3])
            plazo_ejec = col1.slider('🗓️ Plazo de ejecución (Años)', 0, 15, 15, key='plazo_uni')
            duracion_cu = col2.slider('⏳ Duración (Años)', 0, 15, 15, key='duracion_uni')
            aportes_cu = (col1.slider('🌧️ Aportes (%)', -100.0, 100.0, 0.0, key='aportes_uni')/100) + 1
            volumen_cu = col2.slider('💧⚡ Volumen Util Embalses (%)', 0.0, 100.0, 50.0, key='volumen_uni')/100
            energia_cu = col1.number_input("⚡ Energía a contratar (GWh)",key='precio_input_uni', min_value=0.00000,step=0.00001,format="%.5f")
            pBolsa_cu = col2.number_input("💲⚡ Precio de Bolsa (COP/kWh)",key='pbolsa_input_uni', min_value=0.00,step=0.01,format="%.2f")
          
            if energia_cu > 0:
                if pBolsa_cu > 0:
                    pron_up, pron_down = pronostico(modelo,pd.DataFrame([[energia_cu,plazo_ejec,duracion_cu,aportes_cu,volumen_cu,pBolsa_cu]],
                                                                        columns=['GW_TOTAL','PLAZO','DURACION','APORTES', 'VOLUMEN_UTIL', 'PROM_PRECIO_BOLSA']),t_c,std)
                else:
                    st.warning(":warning:💲 Por favor, ingrese un precio de bolsa válido.")
                    pron_up, pron_down = round(0.00,2), round(0.00,2)
            else:
                st.warning(":warning:⚡ Por favor, ingrese una cantidad de energía a contratar válida.")
                pron_up, pron_down = round(0.00,2), round(0.00,2)
            
            st.divider()
            mensaje = '📊 Información: Pronóstico de Contrato'
            st.write(f'<p style="color:{color_dinamico}; font-size:18px; font-weight:bold">{mensaje}</p>', unsafe_allow_html=True)
            col3, col4 = st.columns([3,3])
            col4.metric(':arrow_up:💲Precio máximo de firma (COP/kWh)', round(pron_up,2))
            col3.metric(':arrow_down:💲Precio mínimo de firma (COP/kWh)', round(pron_down,2))
            
        
        with tab1.container(key='cont-result-2'):
            mensaje = '📋 Condiciones del Contrato'
            st.write(f'<p style="color:{color_dinamico}; font-size:18px; font-weight:bold">{mensaje}</p>', unsafe_allow_html=True)
            col5,col6 = st.columns([2000,1])
            duracion_g = col5.slider('⏳ Duración (Años)', 0, 15, 15, key='duracion-gra')
            col7,col8 = st.columns([1,1])
            aportes_g = (col7.slider('🌧️ Aportes (%)', -100.0, 100.0, 0.0, key='aportes-gra')/100)+1
            volumen_g = col8.slider('💧⚡ Volumen Util Embalses (%)', 0.0, 100.0, 50.0, key='volumen-gra')/100
            energia_g = col7.number_input("⚡ Energía a contratar (GWh)",key='precio-input-gra', min_value=0.00000,step=0.00001,format="%.5f")
            pBolsa_g = col8.number_input("💲⚡ Precio de Bolsa (COP/kWh)",key='pbolsa-input-gra', min_value=0.00,step=0.01,format="%.2f")
            if energia_g > 0:
                if pBolsa_g > 0:
                    st.divider()
                    st.markdown('<br>', unsafe_allow_html=True)
                    res_graf = totales(contratos_f2,modelo,t_c,std,energia_g, duracion_g, aportes_g, volumen_g, pBolsa_g)
                    st.plotly_chart(graficar(res_graf), use_container_width=True)
                else:
                    st.warning(":warning:💲 Por favor, ingrese un precio de bolsa válido.")
            else:
                st.warning(":warning:⚡ Por favor, ingrese una cantidad de energía a contratar válida.")

if __name__ == "__main__":
    main()
