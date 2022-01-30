#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:58:02 2020

@author: fantasista
"""
# Librerías
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd, numpy as np, yfinance as yf
from collections import defaultdict
import seaborn as sns

# Listados
cedears = 'AAPL ABEV ABT ADGO AIG AMD AMX AMZN ARCO AUY AXP BA BABA BCS BIDU BMY BNG BP BRFS C CAT CHL CRM CS CSCO CVX DESP DIS EBAY ERJ FB GE GILD GLOB GOLD GOOGL GSK HD HMY HON HPQ HSBC HWM IBM INTC ITUB JD JNJ JPM KO LMT LYG MCD MELI MMM MO MRK MSFT NEM NFLX NKE NOKA NVDA NVS OGZD.L PBR PEP PFE PG PTR PYPL QCOM RDS SAN SAP SBUX SID SLB SNAP SNE SNP T TEN TGT TSLA TWTR TXN TXR UGP UN USB V VALE VIST VOD VZ WFC WMT X XOM'
lider = 'ALUA.BA BBAR.BA BMA.BA BYMA.BA CEPU.BA COME.BA CRES.BA CVH.BA EDN.BA GGAL.BA HARG.BA PAMP.BA SUPV.BA TECO2.BA TGNO4.BA TGSU2.BA TRAN.BA TXAR.BA VALO.BA YPFD.BA'
gold = 'GOLD AUY GLD GDX GDXJ GOEX SLV'

tickers_merval = 'ALUA BBAR BMA BYMA CEPU COME CRES CVH EDN GGAL HARG PAMP SUPV TECO2 TGNO4 TGSU2 TRAN TXAR VALO YPFD '
tickers_merval += 'AGRO AUSO BHIP BOLT BPAT BRIO CADO CAPX CARC CECO2 CELU CGPA2 CTIO DGCU2 ESME FERR GAMI GARO GCLA GRIM HAVA INAG INTR INVJ IRCP IRSA LOMA LEDE LONG METR MIRG MOLA MOLI MORI OEST PATA PGR RICH ROSE SAMI SEMI '
tickers_merval = np.array(tickers_merval.split(), dtype = np.object) + '.BA'
tickers_merval = ' '.join(tickers_merval)

tickers_etfs = 'SPY EEM EWZ FXI XHB XLF XLK XLV XLU XLY XLT XLP XLB XLE XME XRT XOP VNQ IYR IYE ITB IBB TLT HYG USO GLD GDX GDXJ SLV SMH DBA DBB FXE OIH KBE KRE'

# Funciones auxiliares
norm = lambda x: (x - x.min()) / (x.max() - x.min())
# achicar = lambda x, y = 0.9: x.notna() * 1).sum() / len(m) > y

# Definiciones
def grupos(df,
           nclusters = 3,
           stdScaler = True,
           year = 2015,
           nivelmin = 0.9):
    """
    Parameters
    ----------
    #df: panda dataframe - cotizaciones distintos activos
    #nCLusters: int - número de clústeres
    #stdScaler: boolean - normaliza con standar scaler o pct_change
    #year: int - años desde análisis de datos
    #nivelmin: float - porcentaje mínimo de datos válidos por especie
    Returns
    -------
    collection dict clusters

    """
    # Limpieza de dataframe
    data = df[df.index.year >= year].copy()
    nivel = data.apply(lambda x: x.notna() * 1).sum() / len(data)
    cols = data.columns[nivel >= nivelmin]
    data = data[cols].dropna()
    
    # Kmeans
    if stdScaler:
        Xst = StandardScaler().fit_transform(data.values)
    else:
        Xst = data.pct_change().dropna().values
    km = KMeans(n_clusters = nclusters)
    Y = km.fit_predict(Xst.transpose())
    
    # Resultado
    result = defaultdict(list)
    for k,v in list(zip(cols, Y)):
        result[v].append(k)
    
    return result

def kde(df, col):
    """
    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    aux = df.pct_change().dropna()
    aux = aux.melt(id_vars = col)
    sns.lmplot(data = aux,
               x = col,
               y = 'value',
               hue = 'variable',
               fit_reg = True,
               col_wrap = 5,
               height = 2,
               aspect = 1)