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
        if not missing_columns:
            if "kw" in nombre.lower():
                datos.drop(['TIPO','CODIGO_CONVOCATORIA','CONCEPTO','CODIGO_COMERCIALIZADOR','ADJUDICACION_AÑO'],axis=1, inplace=True)
                alerta_kw.success(":heavy_check_mark: Base de datos de energía cargada correctamente")
            if "precios" in nombre.lower():
                datos.drop(['TIPO','IPP_BASE','PERIODO_BASE','CODIGO_CONVOCATORIA',
                        'CODIGO_COMERCIALIZADOR','CONCEPTO'],axis=1, inplace=True)
                alerta_precios.success(":heavy_check_mark: Base de datos de precio cargada correctamente")
        else:
            alerta_precios.warning(f":no_entry: El archivo cargado no es correcto. Faltan las siguientes columnas: {', '.join(missing_columns)}")
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

def totales(datos,modelo,t_c,std):
    resultados = np.zeros((8, 3))
    energia = 0.2
    año_actual = datetime.now().year
    for plazo in range(8):
        fore_up, fore_down = pronostico(modelo,pd.DataFrame([[energia,plazo,1]],columns=['GW_TOTAL','PLAZO','DURACION']),t_c,std)
        resultados[plazo, 0], resultados[plazo, 1], resultados[plazo, 2] = año_actual + plazo, fore_up, fore_down
    return pd.DataFrame(resultados, columns=['ds', 'p_max', 'p_min'])

def graficar(resultados):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resultados['ds'], y=resultados['p_max'], mode='lines+markers', marker=dict(size=10,color='rgba (9, 90, 129, 1)', opacity=0.8),
                             line=dict(width=3, color='rgba (9, 90, 129, 1)', dash='dash'), name='', showlegend=False))
    fig.add_trace(go.Scatter(x=resultados['ds'], y=resultados['p_min'], mode='lines+markers', marker=dict(size=10,color='rgba (9, 90, 129, 1)', opacity=0.8),
                             fill='tonexty', line=dict(width=3, color='rgba (9, 90, 129, 1)', dash='dash'), name='Precio Contratos', fillcolor='rgba(173, 216, 230, 0.5)'))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgray', mirror=False, tickfont_color='black',
                     title_text='<b>Año</b>', titlefont_size=18, tickfont_size=16, title_font_color='black',)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgray', mirror=False, tickfont_color='black',
                     title_text='<b>Precio ($COP/kWh)</b>', titlefont_size=18, tickfont_size=16, title_font_color='black',
                     tickformat='.2f')
    fig.update_traces(hovertemplate='Año:</b> %{x}<br><b>Precio Contrato:</b> %{y:.2f} GW-mes<extra></extra>')
    fig.update_layout(title='', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(220, 220, 220, 220)',
                      font=dict(family="Prompt", color='black'),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.98,
                                  font_size=16, font_color='black'), 
                      showlegend=True, margin=dict(l=50, r=50, t=30, b=50) )
    return fig

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
    st.sidebar.markdown(
    """
    <style type=text/css>
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            justify-content: flex-start;
            align-items: flex-start;
            flex-direction: column;
        }
        
        [data-testid="stSidebar"] img {
            margin-top: -70px;  /* Ajustar según el espacio requerido */
            margin-left: 0px;
        }
        .css-1v0mbdj {
            margin-top: 0px !important;
        }
        .css-1v0mbdj.e1tzin5v3 {
            color: #ffffff !important; /* Color verde para st.sidebar.success */
        }
    
        .css-1v0mbdj.e1tzin5v4 {
            color: #ff9800 !important; /* Color naranja para st.sidebar.warning */
        }

        .st-key-cont-load, .st-key-cont-result{
            background-color:#396425ff !important;
            padding: 10px;
            border-radius: 10px;
        }
        
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.image("images/LogoVatia.png",caption="")
    st.sidebar.divider()
    st.sidebar.header("Pronósticos Contratos")
    st.markdown('<br>', unsafe_allow_html=True)
    st.sidebar.write('**Notificaciones**')
    st.markdown('<br>', unsafe_allow_html=True)
    
    with st.container(key='cont-load'):
        st.write('📤 **Cargar históricos de contratos**')
        col1, col2 = st.columns(2)
        with col1:
            arch1 = st.file_uploader(':zap: Energía', type='csv', key='kw_up')
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
        st.markdown('<br>', unsafe_allow_html=True)
        contenedor = st.sidebar.container(border=True)
        contenedor.write('**Parámetros del Contrato**')
        plazo_ejec = contenedor.slider(':calendar: Plazo de ejecución (Años)', 0, 15, 15)
        duracion = contenedor.slider(':calendar: Duración (Años)', 0, 15, 15)
        energia = contenedor.number_input(":zap: Energía a contratar (GWh)",key='precio_input', min_value=0.0000,step=0.0001,format="%.4f")
        alpha_lit = contenedor.selectbox('Rango de Precisión', ['Alto','Medio','Bajo'],key='alfa')
        if alpha_lit == 'Alto':
            alpha = 0.9
        elif alpha_lit == 'Medio':
            alpha = 0.7
        elif alpha_lit == 'Bajo':
            alpha = 0.5
        modelo, t_c, std = entrenar(contratos_f2,alpha)
        if energia > 0:
            pron_up, pron_down = pronostico(modelo,pd.DataFrame([[energia,plazo_ejec,duracion]],columns=['GW_TOTAL','PLAZO','DURACION']),t_c,std)
            res_graf = totales(contratos_f2,modelo,t_c,std)
        else:
            pron_up, pron_down = round(0.00,2), round(0.00,2)

        with st.container(key='cont-result'):
            st.write('📊 **Información de Pronóstico de Contrato:**')
            col3, col4 = st.columns([3,3])
            col4.metric(':arrow_up::heavy_dollar_sign: Precio máximo de firma (COP/kWh)', round(pron_up,2))
            col3.metric(':arrow_down::heavy_dollar_sign: Precio mínimo de firma (COP/kWh)', round(pron_down,2))
            st.markdown('<br>', unsafe_allow_html=True)
            if energia > 0:
                st.plotly_chart(graficar(res_graf))
if __name__ == "__main__":
    main()
