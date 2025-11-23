import streamlit as st
import yfinance as yf
import numpy as np
import investpy
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from streamlit_echarts import st_echarts
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================



def buscar_ativos():
    # =============================================================================
    # MÉTODO 0 ACOES BRASILEIRAS
    # =============================================================================

    SeP = yf.Ticker('^GSPC')

    histSeP = SeP.history("2d")
    preco_anteriorSeP = histSeP["Close"].iloc[-2]
    preco_atualSeP = histSeP["Close"].iloc[-1]
    varSeP = ((preco_atualSeP - preco_anteriorSeP) / preco_anteriorSeP) * 100
    print(varSeP)

    Ibov = yf.Ticker('^BVSP')
    histIbov = Ibov.history("2d")
    preco_anteriorIbov = histIbov["Close"].iloc[-2]
    preco_atualIbov = histIbov["Close"].iloc[-1]
    varIbov = ((preco_atualIbov - preco_anteriorIbov) / preco_anteriorIbov) * 100
    print(varIbov)

    dol = yf.Ticker('USDBRL=X')
    histdol = dol.history("2d")
    preco_anteriordol = histdol["Close"].iloc[-2]
    preco_atualdol = histdol["Close"].iloc[-1]
    vardol = ((preco_atualdol - preco_anteriordol) / preco_anteriordol) * 100
    print(vardol)

    # dolar = yf.Ticker('')

    def get_all_brazil_stocks():
        """
        Usando investpy para pegar ações brasileiras
        """
        try:
            # Busca todas as ações do Brasil
            br_stocks = investpy.stocks.get_stocks(country='brazil')

            # Converte para formato yfinance (.SA)
            tickers = [stock['symbol'] + '.SA' for stock in br_stocks.to_dict('records')]

            return sorted(tickers)

        except Exception as e:
            print(f"Erro: {e}")
            return []

    # Usar
    tickers = get_all_brazil_stocks()

    # =============================================================================
    # MÉTODO 1: FUNDSPY (Específica para fundos brasileiros)
    # =============================================================================
    def get_fundspy():
        """
        FundsPy: Biblioteca especializada em fundos brasileiros
        Fonte: CVM (Comissão de Valores Mobiliários)

        pip install fundspy
        """
        try:
            import fundspy

            # Buscar fundos
            # Nota: FundsPy foca em fundos de investimento gerais, não FIIs especificamente
            # Mas é excelente para fundos multimercado, renda fixa, etc.

            fundos = fundspy.search.funds_by_name("")  # Buscar todos

            return fundos

        except ImportError:
            print("FundsPy não instalado")
            print("   Instale com: pip install fundspy")
            return None
        except Exception as e:
            print(f"Erro: {e}")
            return None

    # =============================================================================
    # MÉTODO 3: PANDAS DATAREADER (Yahoo Finance) ⭐⭐
    # =============================================================================
    def get_pandas_datareader():
        """
        Pandas DataReader com Yahoo Finance
        Gera lista de tickers potenciais

        pip install pandas-datareader
        """

        try:
            import pandas_datareader as pdr

            # Prefixos comuns de FIIs mais negociados
            prefixos_fiis = [
                'HGLG', 'MXRF', 'KNRI', 'VISC', 'BTLG', 'XPML', 'KNCR', 'PVBI',
                'CPTS', 'RZTR', 'RBRR', 'HGRU', 'ALZR', 'CVBI', 'KNIP', 'KNSC',
                'GGRC', 'RECT', 'JSRE', 'TRXF', 'VILG', 'HGRE', 'HGBS', 'BRCO',
                'LVBI', 'XPLG', 'MALL', 'HSML', 'JRDM', 'HGCR', 'BCFF', 'VGIR',
                'RBRF', 'KFOF', 'BTAL', 'TGAR', 'KNCA', 'KNUQ', 'XPIN', 'VCJR',
                'VINO', 'RBRP', 'RBRY', 'GTWR', 'HGFF', 'HGPO', 'HSAF', 'RBRX'
            ]

            # Prefixos de ETFs
            prefixos_etfs = [
                'BOVA', 'SMAL', 'IVVB', 'HASH', 'MATB', 'DIVO', 'FIND', 'PIBB',
                'ISUS', 'SPXI', 'GOLD', 'NDIV', 'BOVV', 'ECOO', 'WRLD', 'ACWI',
                'NASD', 'GOVE', 'XBOV', 'BOVX', 'IMAB', 'B5MB', 'IRFM', 'FIXA',
                'BITH', 'ETHE', 'XFIX', 'DIVO', 'CRIP', 'CSMO', 'BBSD', 'BBSE'
            ]

            fiis = [f"{p}11.SA" for p in prefixos_fiis]
            etfs = [f"{p}11.SA" for p in prefixos_etfs]

            return {
                'fiis': fiis,
                'etfs': etfs,
                'todos': fiis + etfs
            }

        except ImportError:
            print(" pandas-datareader não instalado")
            print("   Instale com: pip install pandas-datareader")
            return None
        except Exception as e:
            print(f" Erro: {e}")
            return None

    # =============================================================================
    # MÉTODO 4: BUSCA INTELIGENTE COM YFINANCE
    # =============================================================================
    def get_yfinance_smart():

        # Alfabeto para gerar combinações
        letras = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # Gerar FIIs: 4 letras + 11
        # Focar em combinações comuns/conhecidas para não explodir a lista
        fiis_comuns_inicio = ['H', 'K', 'M', 'V', 'R', 'B', 'X', 'G', 'C', 'P', 'T', 'J', 'A', 'L']

        fiis = []
        for letra1 in fiis_comuns_inicio:
            for letra2 in ['G', 'N', 'V', 'B', 'C', 'M', 'R', 'Z', 'T', 'P', 'S']:
                for letra3 in ['R', 'I', 'L', 'C', 'P', 'S', 'T', 'F', 'G']:
                    ticker = f"{letra1}{letra2}{letra3}{'I' if letra3 != 'I' else 'R'}11.SA"
                    fiis.append(ticker)

        # ETFs conhecidos (mais limitados)
        etfs = [
            # Índices Brasil
            'BOVA11.SA', 'SMAL11.SA', 'PIBB11.SA', 'BOVV11.SA', 'FIND11.SA',
            'MATB11.SA', 'ISUS11.SA', 'DIVO11.SA', 'GOVE11.SA', 'XBOV11.SA',
            'BOVX11.SA', 'BOVB11.SA', 'MIDL11.SA',

            # Índices Internacionais
            'IVVB11.SA', 'WRLD11.SA', 'ACWI11.SA', 'NASD11.SA', 'SPXI11.SA',

            # Renda Fixa
            'IMAB11.SA', 'B5MB11.SA', 'IRFM11.SA', 'FIXA11.SA', 'XFIX11.SA',
            'IB5M11.SA', 'IMBB11.SA', 'B5P211.SA',

            # Commodities & Crypto
            'GOLD11.SA', 'BITH11.SA', 'HASH11.SA', 'ETHE11.SA', 'QETH11.SA',
            'CRIP11.SA', 'BIT011.SA',

            # Setoriais
            'UTIL11.SA', 'CSMO11.SA', 'ECOO11.SA', 'NERV11.SA',

            # Smart Beta
            'NDIV11.SA', 'BBSD11.SA', 'BBSE11.SA'
        ]

        # Remover duplicatas
        fiis = list(set(fiis))
        etfs = list(set(etfs))

        return {
            'fiis': fiis,
            'etfs': etfs,
            'todos': fiis + etfs
        }

    # =============================================================================
    # VALIDAÇÃO RÁPIDA (Verificar se existem)
    # =============================================================================
    def validar_rapido(tickers, amostra=50):
        """
        Valida uma amostra de tickers rapidamente
        """
        print(f"\n Validando amostra de {amostra} tickers...")

        import random
        amostra_tickers = random.sample(tickers, min(amostra, len(tickers)))

        validos = []

        for ticker in amostra_tickers:
            try:
                data = yf.download(ticker, period='5d', progress=False)
                if not data.empty and len(data) > 0:
                    validos.append(ticker)
                    print(f"   {ticker}", end='\r')
            except:
                pass

        taxa_sucesso = len(validos) / len(amostra_tickers) * 100

        return validos, taxa_sucesso

    # =============================================================================
    # COMBINAR RESULTADOS DE TODOS OS MÉTODOS
    # =============================================================================
    def combinar_todos_metodos():

        todos_fiis = set()
        todos_etfs = set()

        # Método 3: Pandas DataReader (sempre funciona)
        resultado3 = get_pandas_datareader()
        if resultado3:
            todos_fiis.update(resultado3['fiis'])
            todos_etfs.update(resultado3['etfs'])

        # Método 4: YFinance Smart (sempre funciona)
        resultado4 = get_yfinance_smart()
        if resultado4:
            todos_fiis.update(resultado4['fiis'])
            todos_etfs.update(resultado4['etfs'])

        # Converter para lista
        fiis_final = sorted(list(todos_fiis))
        etfs_final = sorted(list(todos_etfs))

        return {
            'fiis': fiis_final,
            'etfs': etfs_final,
            'todos': fiis_final + etfs_final
        }

    # =============================================================================
    # EXECUTAR
    # =============================================================================

    # Combinar todos os métodos
    resultado = combinar_todos_metodos()

    etfs = resultado["etfs"]
    funds = resultado["fiis"]

    ativos = []
    ativos.extend(etfs)
    ativos.extend(funds)
    ativos.extend(tickers)

    return ativos




st.set_page_config(
    page_title="LMF",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Datas para análise
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365)

STOCKS = buscar_ativos()
DEFAULT_STOCKS = [""]


# ==============================================================================
# FUNÇÕES DE ANÁLISE
# ==============================================================================

def rodar_correlacao(ativos_lista):
    """
    Calcula e exibe um heatmap de correlação entre os ativos.
    """
    import seaborn as sns
    from matplotlib.patches import Patch

    # Baixar dados históricos
    dados = yf.download(ativos_lista, start=START_DATE, end=END_DATE, progress=False)["Close"]

    # Remover ativos com dados faltantes
    dados = dados.dropna(axis=1, how='any')

    # Calcular retornos diários
    retornos = dados.pct_change().dropna()

    # Calcular correlação
    correlacao = retornos.corr()


    class_colors = {'Ações': 'blue', 'FIIs': 'turquoise', 'ETFs': 'midnightblue'}

    # Criar heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlacao, annot=True, cmap='ocean', center=0, linewidths=0.5, ax=ax)

    plt.title("Correlação entre os Ativos - Colorido por Classe", fontsize=13, fontweight="semibold")
    plt.tight_layout()

    return fig


def rodar_montecarlo(tickers, pesos_config):
    """
    Executa simulação de Monte Carlo para o portfólio.
    """
    from scipy import stats

    # Parâmetros da simulação
    dias_historico = 252 * 2
    dias_projecao = 252
    num_simulacoes = 10000
    investimento_inicial = 10000

    # Download e preparação dos dados
    data_fim = datetime.now()
    data_inicio = data_fim - timedelta(days=dias_historico + 100)

    dados = yf.download(tickers, start=data_inicio, end=data_fim, progress=False)

    # Usar Adj Close se disponível, senão Close
    if 'Adj Close' in dados.columns.levels[0]:
        precos = dados['Adj Close']
    else:
        precos = dados['Close']

    # Remover ativos com muitos dados faltantes (>30%)
    limite_dados = len(precos) * 0.7
    precos = precos.dropna(thresh=limite_dados, axis=1)
    precos = precos.ffill().dropna()

    # Atualizar lista de tickers
    tickers = precos.columns.tolist()

    # Calcular retornos diários
    retornos = precos.pct_change().dropna()
    media_retornos = retornos.mean()
    desvio_retornos = retornos.std()
    cov_diaria = retornos.cov()

    # Função de simulação
    def simular_monte_carlo(preco_inicial, media, desvio, dias, num_sim):
        dt = 1
        simulacoes = np.zeros((dias, num_sim))

        for i in range(num_sim):
            precos_sim = [preco_inicial]
            for _ in range(dias - 1):
                drift = (media - 0.5 * desvio ** 2) * dt
                shock = desvio * np.random.normal() * np.sqrt(dt)
                preco_novo = precos_sim[-1] * np.exp(drift + shock)
                precos_sim.append(preco_novo)
            simulacoes[:, i] = precos_sim

        return simulacoes

    # Simulações individuais
    resultados_individuais = {}

    for ticker in tickers:
        preco_atual = precos[ticker].iloc[-1]
        media = media_retornos[ticker]
        desvio = desvio_retornos[ticker]

        simulacoes = simular_monte_carlo(preco_atual, media, desvio, dias_projecao, num_simulacoes)
        precos_finais = simulacoes[-1, :]

        resultados_individuais[ticker] = {
            'simulacoes': simulacoes,
            'preco_atual': preco_atual,
            'preco_medio_final': precos_finais.mean(),
            'retorno_medio': (precos_finais.mean() / preco_atual - 1) * 100,
            'percentil_5': np.percentile(precos_finais, 5),
            'percentil_95': np.percentile(precos_finais, 95),
            'probabilidade_lucro': (precos_finais > preco_atual).sum() / num_simulacoes * 100
        }

    # Simulação do portfólio
    pesos = np.array([pesos_config.get(ticker, 0) for ticker in tickers])
    pesos = pesos / pesos.sum()

    retornos_portfolio = np.zeros(num_simulacoes)

    for i in range(num_simulacoes):
        sim_retornos = np.random.multivariate_normal(media_retornos, cov_diaria, dias_projecao)
        sim_df = pd.DataFrame(sim_retornos, columns=retornos.columns)
        port_ret_diario = (sim_df @ pesos).mean()
        retornos_portfolio[i] = (1 + port_ret_diario) ** dias_projecao - 1

    # Estatísticas do portfólio
    media_port = np.mean(retornos_portfolio)
    mediana_port = np.median(retornos_portfolio)
    desvio_port = np.std(retornos_portfolio)
    p5_port = np.percentile(retornos_portfolio, 5)
    p25_port = np.percentile(retornos_portfolio, 25)
    p75_port = np.percentile(retornos_portfolio, 75)
    p95_port = np.percentile(retornos_portfolio, 95)
    prob_lucro_port = (retornos_portfolio > 0).sum() / num_simulacoes * 100

    # Paleta de cores
    cores = {
        'trajetorias': '#4DBDD8',
        'media': '#0A1C5A',
        'p5': '#3A87B2',
        'p95': '#96E5E8',
        'atual': '#080D1D',
        'fundo': '#F8F9FA'
    }

    # Figura 1: Trajetórias individuais
    fig1, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig1.patch.set_facecolor(cores['fundo'])
    fig1.suptitle(f'Simulação Individual - {num_simulacoes} Trajetórias por Ativo',
                  fontsize=16, fontweight='bold', color=cores['atual'])

    for idx, ticker in enumerate(tickers):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor('white')
        sims = resultados_individuais[ticker]['simulacoes']

        for i in range(min(100, num_simulacoes)):
            ax.plot(sims[:, i], alpha=0.15, color=cores['trajetorias'], linewidth=0.5)

        ax.plot(sims.mean(axis=1), color=cores['media'], linewidth=2.5, label='Média')
        ax.plot(np.percentile(sims, 5, axis=1), color=cores['p5'],
                linewidth=2, linestyle='--', label='P5')
        ax.plot(np.percentile(sims, 95, axis=1), color=cores['p95'],
                linewidth=2, linestyle='--', label='P95')
        ax.axhline(y=resultados_individuais[ticker]['preco_atual'], color=cores['atual'],
                   linestyle=':', linewidth=1.5, label='Atual')

        ax.set_title(f'{ticker}', fontweight='bold', color=cores['atual'])
        ax.set_xlabel('Dias', color=cores['atual'])
        ax.set_ylabel('Preço (R$)', color=cores['atual'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, color='gray')

    plt.tight_layout()

    # Figura 2: Distribuição dos retornos
    fig2, ax = plt.subplots(figsize=(14, 8))
    fig2.patch.set_facecolor(cores['fundo'])
    ax.set_facecolor('white')

    n, bins, patches = ax.hist(retornos_portfolio * 100, bins=60,
                               color=cores['trajetorias'], edgecolor=cores['atual'],
                               alpha=0.7, density=True, linewidth=0.5)

    mu = media_port * 100
    sigma = desvio_port * 100
    x = np.linspace(retornos_portfolio.min() * 100, retornos_portfolio.max() * 100, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=cores['media'], linewidth=3,
            label=f'Distribuição Normal (μ={mu:.2f}%, σ={sigma:.2f}%)')

    ax.axvline(media_port * 100, color=cores['media'], linestyle='--', linewidth=2.5,
               label=f'Média: {media_port * 100:.2f}%')
    ax.axvline(mediana_port * 100, color='#152A9D', linestyle='--', linewidth=2.5,
               label=f'Mediana: {mediana_port * 100:.2f}%')
    ax.axvline(p5_port * 100, color=cores['p5'], linestyle='--', linewidth=2.5,
               label=f'P5: {p5_port * 100:.2f}%')
    ax.axvline(p95_port * 100, color=cores['p95'], linestyle='--', linewidth=2.5,
               label=f'P95: {p95_port * 100:.2f}%')
    ax.axvline(0, color=cores['atual'], linestyle=':', linewidth=2, alpha=0.6)

    ax.axvspan(p5_port * 100, p95_port * 100, alpha=0.15, color=cores['p95'],
               label='Intervalo 90% de confiança')

    ax.set_title(f'Distribuição de Retornos do Portfólio\n' +
                 f'{num_simulacoes:,} Simulações de Monte Carlo - Horizonte: {dias_projecao} dias',
                 fontsize=14, fontweight='bold', pad=20, color=cores['atual'])
    ax.set_xlabel('Retorno Anual (%)', fontsize=12, color=cores['atual'])
    ax.set_ylabel('Densidade de Probabilidade', fontsize=12, color=cores['atual'])
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.2, color='gray')

    # Figura 3: Ranking dos ativos
    fig3, ax = plt.subplots(figsize=(12, 8))
    fig3.patch.set_facecolor(cores['fundo'])
    ax.set_facecolor('white')

    ranking = sorted([(t, resultados_individuais[t]['retorno_medio'],
                       resultados_individuais[t]['probabilidade_lucro'])
                      for t in tickers], key=lambda x: x[1], reverse=True)

    y_pos = np.arange(len(ranking))
    retornos_rank = [r[1] for r in ranking]
    tickers_rank = [r[0] for r in ranking]
    probs_rank = [r[2] for r in ranking]

    colors_bar = [cores['p95'] if r > 0 else cores['p5'] for r in retornos_rank]
    bars = ax.barh(y_pos, retornos_rank, color=colors_bar, alpha=0.8,
                   edgecolor=cores['atual'], linewidth=1.5)

    for i, (ret, prob) in enumerate(zip(retornos_rank, probs_rank)):
        ax.text(ret + 0.5 if ret > 0 else ret - 0.5, i, f'{prob:.0f}%',
                va='center', ha='left' if ret > 0 else 'right', fontsize=9,
                fontweight='bold', color=cores['atual'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tickers_rank, color=cores['atual'])
    ax.set_xlabel('Retorno Esperado (%)', fontsize=12, color=cores['atual'])
    ax.set_title('Ranking de Ativos por Retorno Esperado\n(Números = Probabilidade de Lucro)',
                 fontsize=14, fontweight='bold', color=cores['atual'])
    ax.axvline(0, color=cores['atual'], linestyle='-', linewidth=1.5)
    ax.grid(True, alpha=0.2, axis='x', color='gray')
    ax.tick_params(colors=cores['atual'])

    plt.tight_layout()

    return fig1, fig2, fig3, {
        'media': media_port,
        'mediana': mediana_port,
        'desvio': desvio_port,
        'p5': p5_port,
        'p25': p25_port,
        'p75': p75_port,
        'p95': p95_port,
        'prob_lucro': prob_lucro_port
    }


@st.cache_data
def load_data(tickers, period):
    """
    Carrega dados históricos dos tickers.
    """
    if not tickers:
        return pd.DataFrame()

    tickers_obj = yf.Tickers(tickers)
    data = tickers_obj.history(period=period)

    if data is None or data.empty:
        raise RuntimeError("YFinance returned no data.")

    return data["Close"]


def stocks_to_str(stocks):
    """Converte lista de stocks em string."""
    return ",".join(stocks)


# ==============================================================================
# INTERFACE STREAMLIT
# ==============================================================================

st.title("Análise de Renda Variável - Asset Allocation")
st.write("Escolha ativos da bolsa brasileira que estão na sua carteira")

# Inicializar session state
if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = st.query_params.get(
        "stocks", stocks_to_str(DEFAULT_STOCKS)
    ).split(",")

if 'portfolio' not in st.session_state:
    # Portfolio padrão que aparece na inicialização
    st.session_state.portfolio = [
        {'name': 'PETR4.SA', 'value': 5.0},
        {'name': 'ITUB4.SA', 'value': 6.67},
        {'name': 'EGIE3.SA', 'value': 5.0},
        {'name': 'RDOR3.SA', 'value': 5.0},
        {'name': 'SBSP3.SA', 'value': 6.67},
        {'name': 'WEGE3.SA', 'value': 5.0},
        {'name': 'KNUQ11.SA', 'value': 8.33},
        {'name': 'KNCA11.SA', 'value': 8.33},
        {'name': 'XPML11.SA', 'value': 8.33},
        {'name': 'BTLG11.SA', 'value': 8.33},
        {'name': 'HASH11.SA', 'value': 16.67},
        {'name': 'GOLD11.SA', 'value': 16.67}
    ]

# Layout de colunas
cols = st.columns([1, 3])

# ==============================================================================
# COLUNA ESQUERDA - CONTROLE DE PORTFÓLIO
# ==============================================================================

left_cell = cols[0].container(border=True, height="stretch", vertical_alignment="center")

with left_cell:
    # Selectbox para escolher ativo
    selected_ticker = st.selectbox(
        "Ativo",
        options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
        placeholder="Escolha um ativo",
    )

    # Input de percentual
    percentual = st.number_input(
        "Insira um percentual",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        placeholder="Percentual..."
    )

    # Botões de ação
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Adicionar Ativo"):
            if selected_ticker and percentual > 0:
                existing = next(
                    (item for item in st.session_state.portfolio if item['name'] == selected_ticker),
                    None
                )
                if existing:
                    existing['value'] = percentual
                    st.success(f"Atualizado: {selected_ticker} - {percentual}%")
                else:
                    st.session_state.portfolio.append({
                        'name': selected_ticker,
                        'value': percentual
                    })
                    st.success(f"Adicionado: {selected_ticker} - {percentual}%")
                st.rerun()
            else:
                st.warning("Selecione um ativo e insira um percentual válido")

    with col2:
        if st.button("Limpar Carteira"):
            st.session_state.portfolio = []
            st.rerun()

    # Exibir portfólio atual
    if st.session_state.portfolio:
        total = sum(item['value'] for item in st.session_state.portfolio)
        st.write(f"**Total da carteira: {total:.1f}%**")

        if total > 100:
            st.error("A soma dos percentuais excede 100%!")
        elif total < 100:
            st.warning(f"Faltam {100 - total:.1f}% para completar a carteira")
        else:
            st.success("Carteira completa!")

        # Mostrar itens do portfólio
        for i, item in enumerate(st.session_state.portfolio):
            col_name, col_value, col_delete = st.columns([3, 2, 1])
            with col_name:
                st.write(item['name'])
            with col_value:
                st.write(f"{item['value']}%")
            with col_delete:
                if st.button("🗑️", key=f"delete_{i}"):
                    st.session_state.portfolio.pop(i)
                    st.rerun()

    # Gráfico de pizza
    # Preparar dados no formato correto para ECharts
    pie_data = []
    for item in st.session_state.portfolio:
        pie_data.append({
            "value": item['value'],
            "name": item['name']
        })

    options = {
        "title": {"text": "", "subtext": "", "left": "center"},
        "tooltip": {"trigger": "item"},
        "legend": {"orient": "horizontal", "left": "left"},
        "series": [
            {
                "name": "Carteira",
                "type": "pie",
                "radius": "50%",
                "data": pie_data,
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowOffsetX": 0,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }
        ],
    }

    st_echarts(options=options, height="400px")

    # Botão para rodar análises
    if st.button('Ver Análises da Minha Carteira', type="primary", use_container_width=True):
        if st.session_state.portfolio:
            # Marcar que as análises devem ser exibidas
            st.session_state.show_analyses = True
            st.session_state.tickers_analise = [item['name'] for item in st.session_state.portfolio]
            st.session_state.pesos_analise = {item['name']: item['value'] for item in st.session_state.portfolio}
        else:
            st.warning("Adicione ativos à carteira antes de rodar as análises")

# ==============================================================================
# SEÇÃO DE ANÁLISES (ABAIXO DOS GRÁFICOS)
# ==============================================================================

# Verificar se as análises devem ser exibidas
if 'show_analyses' in st.session_state and st.session_state.show_analyses:
    st.divider()

    with st.spinner('Processando análises...'):
        try:
            # Rodar correlação
            st.subheader("Análise de Correlação")
            fig_corr = rodar_correlacao(st.session_state.tickers_analise)
            st.pyplot(fig_corr)
            plt.close(fig_corr)

            # Rodar Monte Carlo
            st.subheader("Simulação de Monte Carlo")
            fig1, fig2, fig3, stats = rodar_montecarlo(
                st.session_state.tickers_analise,
                st.session_state.pesos_analise
            )

            st.pyplot(fig1)
            plt.close(fig1)
            st.pyplot(fig2)
            plt.close(fig2)
            st.pyplot(fig3)
            plt.close(fig3)

            # Exibir estatísticas
            st.subheader("Estatísticas do Portfólio")
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Retorno Médio", f"{stats['media'] * 100:.2f}%")
                st.metric("Retorno Mediano", f"{stats['mediana'] * 100:.2f}%")
                st.metric("Volatilidade", f"{stats['desvio'] * 100:.2f}%")

            with col_stat2:
                st.metric("Probabilidade de Lucro", f"{stats['prob_lucro']:.1f}%")
                st.metric("Cenário Pessimista (P5)", f"{stats['p5'] * 100:.2f}%")
                st.metric("Cenário Otimista (P95)", f"{stats['p95'] * 100:.2f}%")

        except Exception as e:
            st.error(f"Erro ao processar análises: {str(e)}")
            st.session_state.show_analyses = False

# ==============================================================================
# COLUNA DIREITA - GRÁFICOS
# ==============================================================================

# Extrair tickers e pesos
tickers_escolhidos = [item['name'] for item in st.session_state.portfolio]
pesos_escolhidos = [item['value'] for item in st.session_state.portfolio]

# Criar placeholder fixo para os gráficos
right_cell = cols[1].container(border=True, height="stretch", vertical_alignment="top")

if tickers_escolhidos:
    try:
        # Carregar dados
        data = load_data(tickers_escolhidos, "1y")

        # Verificar se há colunas vazias
        empty_columns = data.columns[data.isna().all()].tolist()

        if empty_columns:
            st.error(f"Erro ao carregar dados para: {', '.join(empty_columns)}.")
            st.stop()

        # Normalizar dados
        normalized = data.div(data.iloc[0])

        # Calcular valor do portfólio
        if isinstance(normalized, pd.Series):
            normalized = normalized.to_frame(name=tickers_escolhidos[0])

        pesos_decimal = [peso / 100 for peso in pesos_escolhidos]
        portfolio_value = pd.Series(0, index=normalized.index)

        for ticker, peso in zip(tickers_escolhidos, pesos_decimal):
            portfolio_value += normalized[ticker] * peso

        # Baixar IBOV e S&P500
        ibov = yf.download('^BVSP', start=START_DATE, end=END_DATE, progress=False)['Close']
        sp500 = yf.download("^GSPC", start=START_DATE, end=END_DATE, progress=False)["Close"]

        # Normalizar índices
        normalized_ibov = (ibov / ibov.iloc[0]).reindex(normalized.index).squeeze()
        normalized_sp500 = (sp500 / sp500.iloc[0]).reindex(normalized.index).squeeze()

        # Preparar dataframes para plotagem
        df_plot_ibov = pd.DataFrame({
            "Date": normalized.index,
            "Portfolio": portfolio_value.values,
            "IBOV": normalized_ibov.values
        })

        df_plot_sp500 = pd.DataFrame({
            "Date": normalized.index,
            "Portfolio": portfolio_value.values,
            "S&P500": normalized_sp500.values
        })

        # Converter para formato long
        df_long_ibov = df_plot_ibov.melt(
            id_vars="Date",
            var_name="Ativo",
            value_name="Preço Normalizado"
        )

        df_long_sp500 = df_plot_sp500.melt(
            id_vars="Date",
            var_name="Ativo",
            value_name="Preço Normalizado"
        )

        # Exibir gráficos
        with right_cell:
            # Gráfico 1: Todas as ações normalizadas
            st.subheader("Gráfico de todas as ações normalizadas")
            st.altair_chart(
                alt.Chart(
                    normalized.reset_index().melt(
                        id_vars=["Date"],
                        var_name="Stock",
                        value_name="Preço Normalizado"
                    )
                )
                .mark_line()
                .encode(
                    alt.X("Date:T"),
                    alt.Y("Preço Normalizado:Q").scale(zero=False),
                    alt.Color("Stock:N"),
                )
                .properties(height=400),
                use_container_width=True
            )

            # Gráfico 2: Portfólio vs Ibovespa
            st.subheader("Portfólio vs Ibovespa")
            chart_ibov = (
                alt.Chart(df_long_ibov)
                .mark_line()
                .encode(
                    x="Date:T",
                    y=alt.Y("Preço Normalizado:Q").scale(zero=False),
                    color="Ativo:N"
                )
                .properties(height=400)
            )
            st.altair_chart(chart_ibov, use_container_width=True)

            # Gráfico 3: Portfólio vs S&P500
            st.subheader("Portfólio vs S&P500")
            chart_sp500 = (
                alt.Chart(df_long_sp500)
                .mark_line()
                .encode(
                    x="Date:T",
                    y=alt.Y("Preço Normalizado:Q").scale(zero=False),
                    color="Ativo:N"
                )
                .properties(height=400)
            )
            st.altair_chart(chart_sp500, use_container_width=True)

    except yf.exceptions.YFRateLimitError:
        st.warning("YFinance está limitando as requisições. Tente novamente mais tarde.")
        load_data.clear()
        st.stop()
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
else:
    with right_cell:
        st.info("Adicione ativos à carteira para visualizar os gráficos")