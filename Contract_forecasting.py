import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import xgboost as xg
import scipy.stats as stats
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import VotingRegressor

if "archivo_cargado_MNR" not in st.session_state:
    st.session_state.archivo_cargado_MNR = False

if "archivo_cargado_MR" not in st.session_state:
    st.session_state.archivo_cargado_MR = False

if 'rerun_done' not in st.session_state:
    st.session_state.rerun_done = False

def crear_placeholder(container):
    placeholder = container.empty()  # Crear un placeholder vacío
    return placeholder

def limpiar_col_monet(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].dropna().astype(str).str.contains('\$').any():
                df[col] = df[col].replace({'\$': '', ',': ''}, regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def dif_MR_MNR(df, col1, col2, tw=0):
    if tw > 0:
        df = df.tail(tw)
    else:
        df = df
    return (df[col1] - df[col2]).mean()

@st.cache_data
def carga_historicos():
    datosHis = pd.read_csv("data/Contratos_antes_SICEP.csv", sep=',', header=0, low_memory=False)
    datosHis = datosHis.rename(columns={'BASE': 'PERIODO_ADJ'})
    datosHis.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1, inplace=True)
    datosHis = datosHis[['AGENTE_COMPRADOR', 'AGENTE_VENDEDOR', 'PERIODO_ADJ','CONTRATO', 'FECHA', 'KW', 'PRECIO']]
    return datosHis

def carga_FactorPSI(archivo, _alerta):
    required_columns = ['AÑO','MES','PROM_PRECIO_BOLSA','APORTES','VOLUMEN_UTIL']
    datosPSI = pd.read_csv(archivo, sep=';', header=0, low_memory=False)
    missing_columns = [col for col in required_columns if col not in datosPSI.columns]
    if not missing_columns:
        _alerta.success("✔️ 💧 Base de datos de hidrología cargada correctamente")
    else:
        return pd.DataFrame()
    return datosPSI

def carga_MNR(archivo, _alerta):
    required_columns = ['Año','Mes','MC','Precio Promedio Contratos No Regulados']
    datosMNR = pd.read_csv(archivo, sep=';', header=0, low_memory=False)
    missing_columns = [col for col in required_columns if col not in datosMNR.columns]
    if not missing_columns:
        _alerta.success("✔️ 💲 Base de datos MNR cargada correctamente")
    else:
        return pd.DataFrame()
    return datosMNR

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
                _alerta.success("✔️ ⚡ Base de datos de energía cargada correctamente")
            if "precios" in nombre.lower():
                datos.drop(['TIPO','IPP_BASE','PERIODO_BASE','CODIGO_CONVOCATORIA',
                        'CODIGO_COMERCIALIZADOR','CONCEPTO'],axis=1, inplace=True)
                _alerta.success("✔️ 💲 Base de datos de precio cargada correctamente")
        else:
            _alerta.warning(f"⛔ El archivo cargado no es correcto. Faltan las siguientes columnas: {', '.join(missing_columns)}")
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
    Contratos_DF_f2.loc[:, 'FECHA'] = pd.to_datetime(Contratos_DF_f2['FECHA'], errors='coerce')
    Contratos_DF_f2.loc[:,'PERIODO_ADJ'] = pd.to_datetime(Contratos_DF_f2['PERIODO_ADJ'], errors='coerce')
    return Contratos_DF_f2

@st.cache_data
def preparar_BD_IA(data):
    result = data.groupby('CONTRATO').agg(
      AÑO_FIRMA = ('PERIODO_ADJ',lambda x: x.min().year),
      MES_FIRMA = ('PERIODO_ADJ',lambda x: x.min().month),
      AÑO_EJECUCION = ('FECHA',lambda x: x.min().year),
      GW_TOTAL = ('KW', lambda x: x.sum() / 1e6),
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
    return merged_df

@st.cache_data
def entrenar(datos,alpha):
    X, y = datos[['GW_TOTAL','PLAZO','DURACION','APORTES', 'VOLUMEN_UTIL', 'PROM_PRECIO_BOLSA']], datos['PRECIO_CONTRATO']
    xgb_r = xg.XGBRegressor(objective ='reg:squarederror', tree_method="hist", eval_metric=mean_absolute_percentage_error,n_estimators = 100,
                            seed = 42, learning_rate=0.025, n_jobs=-1, max_depth=5)
    xgb_r.fit(X, y)
    krr_model = KernelRidge(kernel='rbf', alpha=0.1, gamma=5e-5)
    krr_model.fit(X, y)
    w = 0.45
    voting_reg = VotingRegressor(estimators=[('xgb', xgb_r), ('krr', krr_model)], weights=[w, 1-w])
    voting_reg.fit(X, y)
    pred_train = voting_reg.predict(X)
    residuos = pred_train - y.values
    std_error = np.std(residuos, ddof=1)
    n = len(X)
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    return voting_reg, t_critical, std_error

def pronostico(modelo,datos,t_crit,std):
    pred_test = modelo.predict(datos)
    pred_test_low = pred_test[0] - t_crit * std
    pred_test_high = pred_test[0] + t_crit * std
    return pred_test_high, pred_test_low

def totales(datos, modelo, t_c, std, energia, duracion, aportes, vol_util, PPB):
    resultados = np.zeros((16, 3))
    año_actual = datetime.now().year
    for plazo in range(16):
        fore_up, fore_down = pronostico(modelo,pd.DataFrame([[energia, plazo, duracion, aportes, vol_util, PPB]],
                                                            columns=['GW_TOTAL','PLAZO','DURACION','APORTES', 'VOLUMEN_UTIL', 'PROM_PRECIO_BOLSA']),t_c,std)
        resultados[plazo, 0], resultados[plazo, 1], resultados[plazo, 2] = año_actual + plazo, fore_up, fore_down
    return pd.DataFrame(resultados, columns=['ds', 'p_max', 'p_min']), año_actual, año_actual + 15

def graficar(resultados, dif_MNR, x_range=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resultados['ds'], y=resultados['p_max'], mode='lines+markers', marker=dict(size=10,color='rgba (174, 234, 0, 1)', opacity=0.8),
                             line=dict(width=3, color='rgba (174, 234, 0, 1)', dash='dash'), name='', showlegend=False))
    fig.add_trace(go.Scatter(x=resultados['ds'], y=resultados['p_min'], mode='lines+markers', marker=dict(size=10,color='rgba (174, 234, 0, 1)', opacity=0.8),
                             fill='tonexty', line=dict(width=3, color='rgba (174, 234, 0, 1)', dash='dash'), name='Valoración de energía por contrato', fillcolor='rgba(9, 90, 129, 0.5)'))
    fig.add_trace(go.Scatter(x=resultados['ds'], y=resultados['p_min']-dif_MNR, mode='lines+markers', marker=dict(size=10,color='rgba (234, 154, 0, 1)', opacity=0.8),
                             line=dict(width=3, color='rgba (234, 154, 0, 1)', dash='dash'), name='Valoración Mercado No Regulado'))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='white', gridcolor='lightgray', mirror=False, tickfont_color='white',
                     title_text='<b>Año</b>', tickformat="%Y", dtick="M12", tickfont_size=16, title_font_color='white')
    if x_range:
        fig.update_xaxes(range=[x_range[0]-0.1, x_range[1]+0.1])
    fig.update_yaxes(showline=True, linewidth=2, linecolor='white', gridcolor='lightgray', mirror=False, tickfont_color='white',
                     title_text='<b>Precio (COP/kWh)</b>', tickfont_size=16, title_font_color='white',
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
                border-radius: 5px;         /* Esquinas redondeadas */
                padding: 15px;              /* Añadir algo de padding */
                color: white;               /* Cambiar color del texto a blanco */
                font-family: 'Prompt', sans-serif !important;
            }

            hr:nth-of-type(1) {
                width: 100% !important;
            }

            div[data-testid="stSelectbox"], div[data-testid="stAlert"] {
                border-radius: 20px;         /* Esquinas redondeadas */
                padding: 0px;              /* Añadir algo de padding */
                color: white;               /* Cambiar color del texto a blanco */
                font-family: 'Prompt', sans-serif !important;
                width: 100% !important;
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
                padding: 20px;
                border-radius: 5px;
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

            [data-testid='stFileUploaderDropzoneInstructions'] > div:{
                content: 'Arrastre aquí los archivos';
            }

            [data-testid='stBaseButton-secondary'] {
                text-indent: -9999px; line-height: 0;
            }

            [data-testid='stBaseButton-secondary']::after { line-height: initial;
                content: "Cargar"; text-indent: 0;
            }

            .st-key-styled_expander details,
            .st-key-styled_expander_2 details {
                border-color: green;
            }
            .st-key-styled_expander summary,
            .st-key-styled_expander_2 summary {
                border-radius: .5rem;
            }

            .st-key-styled_expander svg[data-testid="stExpanderToggleIcon"],
            .st-key-styled_expander_2 svg[data-testid="stExpanderToggleIcon"] {
                color: white;
            }

            .st-key-styled_expander div[data-testid="stExpanderDetails"] p h1,
            .st-key-styled_expander_2 div[data-testid="stExpanderDetails"] p h1{
                color: white;
            }

            .st-key-styled_tabs button{
                width:50%;
                border-radius:5px;
            }

            .st-key-styled_tabs button[aria-selected="true"]{
                background-color:#396425ff;
                border-radius:5px;
            }

            .st-key-styled_tabs div[data-baseweb="tab-panel"]{
                border-radius:5px;
                padding:10px;
                background-color:#396425ff;
            }

            .st-key-styled_tabs{
                background-color:grey;
                border-radius:5px;
            }

            </style>
      """, unsafe_allow_html=True)
    
def main():
    st.set_page_config(page_title="Pronóstico Contratos Vatia", page_icon="images/icon.png", layout="wide")
    estilo()
    color_dinamico = "#aeea00"  # Verde Manzana
    st.sidebar.image("images/LogoVatia.png",caption="",use_container_width=True)

    st.sidebar.divider()
    st.header("Pronósticos Contratos")


    # Sección de notificaciones (4 placeholders)
    exp_notif = st.sidebar.container(key="styled_expander").expander("**🚨 Notificaciones**", expanded=True)
    ph_alerta = [crear_placeholder(exp_notif) for _ in range(4)]
    st.sidebar.divider()

    # Sección Históricos MR
    cont_mr = st.sidebar.container(key="styled_expander_2")
    exp_mr = cont_mr.expander("**📤📋 Históricos (MR)**", expanded=not st.session_state.archivo_cargado_MR)
    exp_mr.markdown(
        "**Por favor, cargar la siguiente información:**\n- 💲 Precios por contrato\n- ⚡ Energía pactada por contrato\n- 💧 Hidrología y Precio de Bolsa"
    )
    archivos = crear_placeholder(exp_mr).file_uploader('', type='csv', key='kw_up', accept_multiple_files=True)

    # Sección Históricos MNR
    exp_mnr = cont_mr.expander("**📤📋 Históricos (MNR)**", expanded=not st.session_state.archivo_cargado_MNR)
    exp_mnr.markdown("**Por favor, cargar la siguiente información:**\n- 💲 Precios MNR")
    arch_MNR = crear_placeholder(exp_mnr).file_uploader('', type='csv', key='kw_MNR')

    # Seleccionar archivos para MR según nombre
    arch1 = arch2 = arch3 = None
    if archivos:
        for f in archivos:
            n = f.name.lower()
            if "kw" in n:
                arch1 = f
            elif "precio" in n:
                arch2 = f
            elif "pb_hidro" in n:
                arch3 = f

    kw = carga_archivos(arch1, arch1.name, ph_alerta[0]) if arch1 else pd.DataFrame()
    if kw.empty: ph_alerta[0].warning(":warning: Por favor, carga un archivo válido de energía contratada.")
    precios = carga_archivos(arch2, arch2.name, ph_alerta[1]) if arch2 else pd.DataFrame()
    if precios.empty: ph_alerta[1].warning(":warning: Por favor, carga un archivo válido de precios de contrato.")
    BD_PSI = carga_FactorPSI(arch3, ph_alerta[2]) if arch3 else pd.DataFrame()
    if BD_PSI.empty: ph_alerta[2].warning(":warning: Por favor, carga un archivo válido de hidrología.")

    if arch1 and arch2 and arch3:
      st.session_state.archivo_cargado_MR = True

    BD_MNR = limpiar_col_monet(carga_MNR(arch_MNR, ph_alerta[3])) if arch_MNR else pd.DataFrame()
    if arch_MNR:
      st.session_state.archivo_cargado_MNR = True

    if BD_MNR.empty: ph_alerta[3].warning(":warning: Por favor, carga un archivo válido de historicos de MNR.")

    if not kw.empty and not precios.empty and not BD_PSI.empty and not BD_MNR.empty:
        if not st.session_state.rerun_done:
            st.session_state.rerun_done = True
            st.rerun()
        if 'Contratos' not in st.session_state:
            BD_con = gen_BD_Contratos(kw, precios)
            BD_prev = unif_BD_contratos(carga_historicos(), BD_con)
            contratos = preparar_BD_IA(BD_prev)
            contratos_PSI = transfer_PSI_columns(BD_PSI, contratos)
            contratos_PSI.dropna(inplace=True)
            st.session_state['Contratos'] = contratos_PSI

        contratos_f2 = st.session_state['Contratos']
        contenedor = st.container(key='cont-maestro')
        alpha_lit = contenedor.selectbox('🎯 Rango de Precisión', ['Alto', 'Medio', 'Bajo'], key='alpha-sel')
        st.markdown('<br>', unsafe_allow_html=True)
        alpha = 0.9 if alpha_lit == 'Alto' else 0.7 if alpha_lit == 'Medio' else 0.5
        modelo, t_c, std = entrenar(contratos_f2, alpha)
        
        with st.container(key="styled_tabs"):
            tab1, tab2 = st.tabs(["📈 Valoración de Energía por Contratos", "📊 Valoración Precio de Contratos"])

            with tab2.container(key='cont-result'):
                st.write(f'<p style="color:{color_dinamico}; font-size:18px; font-weight:bold">📋 Condiciones del Contrato</p>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                plazo_ejec = col1.slider('🗓️ Inicio del contrato (Años)', 0, 15, 2, key='plazo_uni')
                duracion_cu = col2.slider('⏳ Duración (Años)', 0, 15, 1, key='duracion_uni')
                energia_cu = st.number_input("⚡ Energía a contratar por contrato (GWh)", key='precio_input_uni', min_value=0, step=1, format="%.0d")

                st.divider()
                st.write(f'<p style="color:{color_dinamico}; font-size:18px; font-weight:bold">📰 Información del Mercado</p>', unsafe_allow_html=True)
                col3, col4,  = st.columns(2)

                volumen_cu = col3.slider('💧⚡ Volumen útil embalses (%)', 0, 100, 50, key='volumen_uni') / 100.0
                aportes_cu = col4.slider('🌧️ Aportes sobre la media (%)', 0, 200, 100, key='aportes-uni') / 100.0
                pBolsa_cu = st.number_input("💲⚡ Precio de bolsa (COP/kWh)", key='pbolsa_input_uni', min_value=0, step=1, format="%.0d")

                if energia_cu > 0 and pBolsa_cu > 0:
                    df_input = pd.DataFrame([[energia_cu, plazo_ejec, duracion_cu, aportes_cu, volumen_cu, pBolsa_cu]],
                                            columns=['GW_TOTAL','PLAZO','DURACION','APORTES','VOLUMEN_UTIL','PROM_PRECIO_BOLSA'])
                    pron_up, pron_down = pronostico(modelo, df_input, t_c, std)
                else:
                    st.warning("⚠️ Ingrese valores válidos para energía y precio de bolsa.")
                    pron_up, pron_down = 0.00, 0.00
                st.divider()
                st.write(f'<p style="color:{color_dinamico}; font-size:18px; font-weight:bold">📊 Información: Pronóstico de Contrato</p>', unsafe_allow_html=True)
                col5, col6 = st.columns(2)
                col5.metric('💲Precio máximo de firma (COP/kWh)', round(pron_up, 2))
                col6.metric('💲Precio mínimo de firma (COP/kWh)', round(pron_down, 2))

            with tab1.container(key='cont-result-2'):
                st.write(f'<p style="color:{color_dinamico}; font-size:18px; font-weight:bold">📋 Condiciones del Contrato</p>', unsafe_allow_html=True)
                energia_g = st.number_input("⚡ Energía a contratar por contrato (GWh)", key='precio-input-gra', min_value=0, step=1, format="%.0d")

                st.divider()
                st.write(f'<p style="color:{color_dinamico}; font-size:18px; font-weight:bold">📰 Información del Mercado</p>', unsafe_allow_html=True)
                col6, col7 = st.columns(2)

                volumen_g = col6.slider('💧⚡ Volumen útil embalses (%)', 0, 100, 50, key='volumen-gra') / 100.0
                aportes_g = col7.slider('🌧️ Aportes sobre la media (%)', 0, 200, 100, key='aportes-gra') / 100.0
                pBolsa_g = st.number_input("💲⚡ Precio de bolsa (COP/kWh)", key='pbolsa-input-gra', min_value=0, step=1, format="%.0d")

                if energia_g > 0 and pBolsa_g > 0:
                    st.divider(); st.markdown('<br>', unsafe_allow_html=True)
                    res_graf, año_actual, año_max = totales(contratos_f2, modelo, t_c, std, energia_g, 1, aportes_g, volumen_g, pBolsa_g)
                    x_range = st.slider("↔️ Rango de análisis", año_actual, año_max, (año_actual, año_max), key='x_range_slider')
                    st.plotly_chart(graficar(res_graf, dif_MR_MNR(BD_MNR, 'MC', 'Precio Promedio Contratos No Regulados', 36), x_range=x_range), use_container_width=True)
                else:
                    st.warning("⚠️ Ingrese valores válidos para energía y precio de bolsa.")

if __name__ == "__main__":
    main()
