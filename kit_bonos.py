#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:28:09 2020

@author: fantasista
"""
import pandas as pd, numpy as np, requests
from itertools import combinations
import matplotlib.pyplot as plt

def get_prices(ticker,
               url = r'https://www.rava.com/empresas/precioshistoricos.php',
               serie = None,
               merge = 'Cierre',
               columns = ['date', 'o', 'h', 'l', 'c', 'v']
               ):
    """
    Devuelve panda dataframe de rava.com
    ticker : int
    """
    if isinstance(ticker, str):
        payload = {'e' : ticker}
        attrs = {'class' : 'tablapanel'}
        r = requests.get(url = url,
                         params = payload)
        data = pd.read_html(io = r.text,
                            attrs = attrs,
                            header = 0,
                            index_col = [0],
                            thousands = '.',
                            decimal = ',',
                            converters = {'Volumen' : np.float64}
                            # parse_dates = [0],
                            # date_parser = lambda fecha: pd.to_datetime(fecha, dayfirst = True)
                            )[0]
        data.index = pd.to_datetime(data.index,
                                    dayfirst = True)
        return data if serie is None else data[serie]
    elif isinstance(ticker, (list, np.ndarray)):
        df = pd.DataFrame({e : get_prices(e, serie = merge) for e in ticker})
        return df
# def get_csv()
años = [29, 30, 35, 38, 41]
bonosNY = ['GD' + str(e) for e in años]
bonosAR = ['AL' + str(e) if e != 38 else 'AE' + str(e) for e in años]
mask = np.tile([False, True], len(bonosAR))
bonos = np.repeat(bonosAR, 2).astype(np.object)
bonos[mask] += 'D'

# pAR = get_prices(bonosAR)
# paridades = pd.DataFrame()
# for par in combinations(bonosAR, 2):
#     paridades[par[0] + '/' + par[1]] = pAR[par[0]] / pAR[par[1]]

# par30 = get_prices(bonos)
# par30['mep'] = par30['AL30'] / par30['AL30D']
# par30[bonos[np.invert(mask)]].div(par30['mep'], axis = 0)

todos = np.repeat(bonosAR + bonosNY, 2).astype(np.object)
m = [e % 2 == 1 for e in range(todos.size)]
todos[m] += 'D'
# rel = get_prices(todos)

# mep = rel.AL30 / rel.AL30D
# teoricos = rel[bonosAR + bonosNY].div(mep,
#                                       axis = 0)
# teoricos.columns = teoricos.columns + 'D'

# reales = rel[teoricos.columns]

def update(file_name):
    """
    lee y graba
    """
    get_prices(todos).to_excel(file_name + '.xlsx')
    

def arbitrajes(df,
               bono = 'al30',
               diferencia = False):
    """
    devuelve un df con precios en usd arbitrados por un bono
    df : dataframe con datos
    bono : str - ticker del bono a usar como base del arbitraje
    """
    df.columns = df.columns.str.casefold()
    tickers = [t for t in df.columns if t[-1] != 'd']
    tickersd = [t for t in df.columns if t[-1] == 'd']    
    if bono not in df.columns:
        raise ValueError('Bono no incluido.')
    elif diferencia:
        return df[tickers].div(df[tickersd].values,
                               axis = 0)
    else:
        mep = df[bono] / df[bono + 'd']
        result = df[tickers].div(mep,
                                 axis = 0)
        result.columns = result.columns + 'd'
        return result - df[tickersd]
    
def relacion(df,
             selector,
             **kwargs):
    """
    devuelve df con relaciones
    selector : función que devuelve True-Falsa dada una condición 
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Data ingresada no es dataframe.')
    else:
        data = df.copy()
        data.columns = data.columns.str.casefold()
        tickers = {1 : [t for t in data.columns if selector(t, **kwargs) == True],
                   0 : [t for t in data.columns if selector(t, **kwargs) == False]
                   }
        if len(tickers[0]) != len(tickers[1]):
            raise ValueError('Error en selector.')
        result = pd.DataFrame(data = data[tickers[1]].values / data[tickers[0]].values,
                              index = data.index,
                              columns = tickers[1])
        return result

# Función Ratios de bonos 

def ratios(precios,
           ticker = None,
           restriccion = None,
           col_fechas = 'Fecha'
           ):
    """
    Devuelve panda df con ratios entre ticker y resto aplicada la restricción
    precios : df - precios y tickers en columnas
    ticker : str - ticker a relacionar
    restriccion : función restrictiva
    col_fechas : str - columna con fechas si tiene index datetime
    Sin ticker ni restricción toma el primer ticker de columnas y aplica a todo
    """
    precios = precios.copy()

    if isinstance(precios.index, pd.DatetimeIndex) == False and col_fechas in precios.columns:
        precios.set_index(col_fechas,
                          inplace = True)
    elif isinstance(precios.index, pd.DatetimeIndex) == False and col_fechas not in precios.columns:
        raise IndexError('Datos de fechas en dataframe no en columnas.')

    if ticker is None:
        ticker = precios.columns[0]
    elif ticker not in precios.columns:
        raise ValueError('Ticker no incluído en especies dadas.')
    
    mask = []
    especies = precios.columns.copy()
    mask.append(especies != ticker)
    if restriccion is not None:
        if isinstance(restriccion, (list, np.ndarray)):
            for r in restriccion:
                mask.append([r(t) for t in especies])
        else:
            mask.append([restriccion(t) for t in especies])
    
    mask = np.logical_and.reduce(mask)
    especies = especies[mask]
    
    result = precios[especies].div(precios[ticker], axis = 0)
    result.columns += '/' + ticker
    return result

# show plot

def show_ratio(ratios,
                col,
                ma = 5,
                dev = 2,
                figsize = (12, 6)
                ):
    """
    plotea una de las columnas
    ratios: panda df con ratios
    col: str - columna a plotear
    """
    if isinstance(ratios, pd.Series):
        ratios = ratios.copy()
        agg_d = {'m' : 'mean', 's' : 'std'}
        mean, std = ratios.agg(agg_d)
        rmean = ratios.rolling(ma).agg(agg_d)
        ax = ratios.plot(kind = 'line',
                        figsize = figsize,
                        color = 'indigo',
                        )
        rmean['m'].plot(color = 'orangered',
                    ax = ax)
        
        for d in [+ dev, - dev]:
            ax.plot(rmean.index.values,
                    rmean.m + rmean.s * d,
                    color = 'green',
                    linestyle = '-.')
        ax.axhline(y = mean,
                   color = 'k',
                   linestyle = '--',
                   label = 'Media')

        ax.hlines(y = [mean + std * dev, mean - std * dev],
                  xmin = rmean.index.min(),
                  xmax = rmean.index.max(),
                  color = 'indigo',
                  linestyle = ':')
        
        ax.set_title('Ratio ' + ratios.name)
        ax.set_xlabel('Fechas')
        ax.set_ylabel('Ratio')
        plt.show()   
    elif isinstance(ratios, pd.DataFrame):
        show_ratio(ratios[col],
                  col = col,
                  ma = ma,
                  dev = dev,
                  figsize = figsize)
    else:
        raise TypeError('Ratios no es dataframe.')

#Función display auxiliar
def _funcion_display(prices,
                     col_fechas,
                     tickerbase,
                     tickercomp,
                     dolar,
                     legislacion,
                     ma,
                     dev,
                     figsize):
    """
    Función Auxiliar para display de ploteo usando interactive widhets e ipython
    """
    #Armado de opciones para display
    m1 = np.isin(prices.columns, col_fechas)
    m2 = prices.columns.str.endswith('D') if dolar else np.invert(prices.columns.str.endswith('D'))
    m3 = np.ones(prices.columns.shape, dtype = bool) # OjOOOOOOO
    if legislacion == 'NY':
        m3 = prices.columns.str.startswith('GD')
    elif legislacion == 'AR':
        m3 = np.invert(prices.columns.str.startswith('GD'))
    mask = np.logical_and.reduce([m1, m2, m3])
    tickerbase_cols = prices.columns[np.logical_and(m1, m2)]
    tickercomp_cols = prices.columns[mask]
    cols = [col_fechas, tickerbase, tickercomp]
    ratio_par = ratios(prices[cols])
    show_ratio(ratio_par,
                col = tickercomp + '/' + tickerbase,
                ma = ma,
                dev = dev,
                figsize = figsize
                )
# display_ratio

import ipywidgets as widgets
from IPython.display import display

def display_pair(cotizaciones,
                col_fechas = 'Fecha'):
    """
    Función para plotear paridades
    """
    #Armado de display widgets usando handler-observe para actualización automática
    #Selecciín Inicial
    #global handler
    handler = {'tickerbase': cotizaciones.columns[1:2],
              'tickercomp' : cotizaciones.columns[2:3],
              'dolar' : False,
              'legislacion' : 'AR'}
    
    #Funciones handler
    def handler_tickerbase(change):
        handler['tickerbase'] = change['new']
        _update()
        
    def handler_tickercomp(change):
        handler['tickercomp'] = change['new']
        _update()
    
    def handler_dolar(change):
        handler['dolar'] = change['new']
        _update()
        
    def handler_legislacion(change):
        handler['legislacion'] = change['new']
        _update()
        
    def _ticker_constructor(cots,
                            dolar,
                            legislacion,
                            ticker_extra = None):
        """
        devuelve iterable con tickers para usar en tickerbase y tickercomp
        dolar: bool - especies en dolar o pesos
        legislacion: str - AR-NY-All
        col_fecha: columna con datos datetime para eliminar
        ticker_extra: str o iterable - ticker(s) extra a eliminar
        """
        #Armado de opciones para display
        m1 = np.isin(cots.columns, ticker_extra, invert = True)
        m2 = cots.columns.str.endswith('D') if dolar else np.invert(cots.columns.str.endswith('D'))
        m3 = np.ones(cots.columns.shape, dtype = bool) # OjOOOOOOO
        if legislacion == 'NY':
            m3 = cots.columns.str.startswith('GD')
        elif legislacion == 'AR':
            m3 = np.invert(cots.columns.str.startswith('GD'))
        mask = np.logical_and.reduce([m1, m2, m3])
        #tickerbase_cols = prices.columns[np.logical_and(m1, m2)]
        return cots.columns[mask]
    
    def _update():
        """
        funcion update tickerbase dropdown
        """
        tickerbase.options = _ticker_constructor(cots = cotizaciones,
                                                dolar = handler['dolar'],
                                                legislacion = handler['legislacion'],
                                                ticker_extra = col_fechas)
        tickercomp.options = _ticker_constructor(cots = cotizaciones,
                                                dolar = handler['dolar'],
                                                legislacion = handler['legislacion'],
                                                ticker_extra = [col_fechas, handler['tickerbase']])
     
    def _display(_tickerbase,
                 _tickercomp,
                 _ma,
                 _dev,
                 _figsize):
        """
        Función Auxiliar para display de ploteo usando interactive widhets e ipython
        """
        #Armado de opciones para display
        cols = [col_fechas, _tickerbase, _tickercomp]
        ratio_par = ratios(cotizaciones[cols])
        show_ratio(ratio_par,
                    col = _tickercomp + '/' + _tickerbase,
                    ma = _ma,
                    dev = _dev,
                    figsize = _figsize
                    )
    
    #Layer 1
    dolar = widgets.ToggleButtons(options = [('Pesos', False), ('Dolar', True)],
                                  value = handler['dolar'],
                                description = 'Especies D',
                                 button_style = 'warning',
                                disabled = False,
                                indent = True
                                )
    
    legislacion = widgets.ToggleButtons(options = [('Argentina', 'AR'), ('New York', 'NY'), ('Todos', 'All')],
                                        value = handler['legislacion'],
                                        description = 'Legislación',
                                        button_style = 'success',
                                        disabled = False,
                                        indent = True
                                        )
    
    #Layer 2
    tickerbase = widgets.Dropdown(options = handler['tickerbase'],
                                 description = 'Ticker Base',
                                 disabled = False
                                 )
    
    tickercomp = widgets.Dropdown(options = handler['tickercomp'],
                                 description = 'Ticker Comp',
                                 disabled = False
                                 )
    
    #Layer 3
    ma = widgets.IntSlider(value = 5,
                          min = 5,
                          max = 20,
                          step = 5,
                          description = 'MA',
                          disabled = False,
                          orientation = 'horizontal',
                          )
    
    dev = widgets.FloatSlider(value = 2.,
                              min = 1.,
                              max = 3.,
                              step = .5,
                              description = 'Dev',
                              disabled = False,
                              orientation = 'horizontal',
                              )
        
    figsize = widgets.SelectionSlider(
        options = [('L', (15, 8)), ('M', (12, 6)), ('S', (8, 4))],
        value = (12,6),
        description = 'Size',
        disabled = False
        )
    
    #UIs
    ui = widgets.HBox([dolar, legislacion])
    ui_tickers = widgets.HBox([tickerbase, tickercomp])
    ui_tecs = widgets.HBox([ma, dev, figsize])
    
    widgets.AppLayout(header = ui,
                     center = ui_tickers,
                     footer = ui_tecs)
 
    #Ejecución Handlers
    dolar.observe(handler_dolar, names = 'value')
    legislacion.observe(handler_legislacion, names = 'value')
    tickerbase.observe(handler_tickerbase, names = 'value')
    
    #Componentes
    #tickerbase.options = _ticker_constructor(prices = prices,
    #                                        dolar = handler['dolar'],
    #                                        legislacion = handler['legislacion'],
    #                                        ticker_extra = col_fechas)

    def muestra():
        print(handler)
    
    out_d = {'_tickerbase' : tickerbase,
             '_tickercomp' : tickercomp,
             '_ma' : ma,
             '_dev' : dev,
             '_figsize' : figsize}
    
    out = widgets.interactive_output(_display,
                                     out_d)
    
    #show_ratio(ratios,
    #            col,
    #            ma = 5,
    #            dev = 2,
    #            figsize = (12, 6)
    #            ):

    display(ui, ui_tickers, ui_tecs, out)
    
    

####  Display Simple ####
def display_simple(prices,
                  col_fechas = 'Fecha'):
    """
    Función para usar con widgets display - ploteador
    Más directa
    """
    #Primer Layout
    dolar = widgets.ToggleButtons(options = [('Pesos', False), ('Dolar', True)],
                                description = 'Especies D',
                                disabled = False,
                                indent = False
                                )
    legislacion = widgets.ToggleButtons(options = [('Argentina', 'AR'), ('New York', 'NY'), ('Todos', 'All')],
                                       description = 'Legislación',
                                       disabled = False
                                       )
    ui = widgets.HBox([dolar, legislacion])
    #Segundo Layout
    #Funciones auxiliares
    def _tickercols(d, l):
        """
        Funcion auxiliar para seleccion de tickers
        """
        #Armado de opciones para display
        m1 = np.isin(prices.columns, col_fechas, invert = True)
        m2 = prices.columns.str.endswith('D') if d else np.invert(prices.columns.str.endswith('D'))
        m3 = np.ones(prices.columns.shape, dtype = bool) # OjOOOOOOO
        if l == 'NY':
            m3 = prices.columns.str.startswith('GD')
        elif l == 'AR':
            m3 = np.invert(prices.columns.str.startswith('GD'))
        mask = np.logical_and.reduce([m1, m2, m3])
        #tickerbase_cols = prices.columns[np.logical_and(m1, m2)]
        tickerbase_cols = prices.columns[mask]

        tickerbase = widgets.Dropdown(options = tickerbase_cols,
                             description = 'Ticker Base',
                             disabled = False
                             )
        
        tickercomp = widgets.Dropdown(options = tickerbase_cols[tickerbase_cols != tickerbase.value],
                             description = 'Ticker Comparado',
                             disabled = False                                     
                             )
        
        
        # Ojo armar handler para todo
        def handler(change):
            tickercomp.options = tickerbase_cols[tickerbase_cols != change['new']]
        
        tickerbase.observe(handler, names = 'value')       
        
        
        
        ui_aux = widgets.HBox([tickerbase, tickercomp])
        display(ui_aux)
        
    out = widgets.interactive_output(_tickercols, {'d': dolar, 'l': legislacion})
    display(ui, out)

def display_ratio(prices,
                  col_fechas = 'Fecha'
                  ):
    """
    función para usar con widgets display - ploteador
    restriccion dolar
    restriccion ar-ny
    """
    #Funciones Restrictivas
    #f_dolar = lambda x: x[-1] == 'D'
    #f_ar = lambda x: x[:2] != 'GD'
    
    #Función opciones básicas auxiliar
    def _options_display():
        """
        Devuelve un diccionario con las opciones básicas para el widget
        dolar
        legislacion
        tickerbase_cols
        tickercomo_cols
        """
        #Armado de opciones para display
        m1 = np.isin(prices.columns, col_fechas)
        m2 = prices.columns.str.endswith('D') if dolar else np.invert(prices.columns.str.endswith('D'))
        m3 = np.ones(prices.columns.shape, dtype = bool) # OjOOOOOOO
        if legislacion == 'NY':
            m3 = prices.columns.str.startswith('GD')
        elif legislacion == 'AR':
            m3 = np.invert(prices.columns.str.startswith('GD'))
        mask = np.logical_and.reduce([m1, m2, m3])
        tickerbase_cols = prices.columns[np.logical_and(m1, m2)]
        tickercomp_cols = prices.columns[mask]
    
    #Función display auxiliar
    def _funcion_display(tickerbase,
                        tickercomp,
                        dolar,
                        legislacion,
                        ma,
                        dev,
                        figsize):
        """
        Función Auxiliar para display de ploteo usando interactive widhets e ipython
        """
        #Armado de opciones para display
        m1 = np.isin(prices.columns, col_fechas)
        m2 = prices.columns.str.endswith('D') if dolar else np.invert(prices.columns.str.endswith('D'))
        m3 = np.ones(prices.columns.shape, dtype = bool) # OjOOOOOOO
        if legislacion == 'NY':
            m3 = prices.columns.str.startswith('GD')
        elif legislacion == 'AR':
            m3 = np.invert(prices.columns.str.startswith('GD'))
        mask = np.logical_and.reduce([m1, m2, m3])
        tickerbase_cols = prices.columns[np.logical_and(m1, m2)]
        tickercomp_cols = prices.columns[mask]
        cols = [col_fechas, tickerbase, tickercomp]
        ratio_par = ratios(prices[cols])
        show_ratio(ratio_par,
                    col = tickercomp + '/' + tickerbase,
                    ma = ma,
                    dev = dev,
                    figsize = figsize
                    )
    
    # OJO acá
    # Widgets 
    #r_show = widgets.interactive(
    #    _funcion_display,
    #    tickerbase = widgets.Dropdown(
    #        #options = tickerbase_cols,
    #        #value = tickerbase_cols[0]
    #        options = ['AL30', 'AL29']
    #        ), 
    #    tickercomp = widgets.Dropdown(
    #        #options = tickercomp_cols,
    #        #value = tickercomp_cols[1]
    #        options = ['AL30', 'AL29']
    #        ),
    #    dolar = widgets.ToggleButton(
    #        value = False,
    #        disabled = False
    #        ),
    #    legislacion = widgets.RadioButtons(
    #        options = ['AR', 'NY', 'Todos'],
    #        value = 'Todos'
    #        ),
    #    ma = (5, 20, 5),
    #    dev = (1, 3, .5),
    #    figsize = widgets.RadioButtons(
    #        options = [('L', (12, 6)), ('S', (8, 4))]),
    #        value = 'L'
    #    )
    
    #Uso Layouts Hbox etc
    
    
    display(r_show)
# Simple Ratio Ploteador    

def simpleratio(df,
                cols,
                ma = 20,
                dev = 2,
                dateindex = 'Fecha',
                figsize = (12, 6)):
    """
    plotea relacion entre las dos columnas dadas
    df : panda dataframes con datos
    cols : list - con columnas a relacionar
    ma : longitud de la media móvil
    dev : desvios 
    """
    data = df.set_index(dateindex).copy()
    data = df[cols].dropna()
    data['ratio'] = data[cols[0]].values / data[cols[1]].values
    data['media'] = data['ratio'].rolling(ma).mean()
    data['dev'] = data['ratio'].rolling(ma).std()
    
    mean, std = data['ratio'].mean(), data['ratio'].std()
    
    fechas = data.index.values.astype('<M8[ns]')
    data.reset_index(inplace = True)
    ax = data[['ratio']].plot(kind = 'line',
                              color = 'indigo',
                              figsize = figsize)
    
    data[['media']].plot(color = 'orangered',
                         ax = ax)
    
    for d in [+ dev, - dev]:
        ax.plot(data.index.values,
                data.media + data.dev * d,
                color = 'green',
                linestyle = '-.')
    
    ax.axhline(y = mean,
               color = 'k',
               linestyle = '--',
               label = 'Media')
    
    ax.hlines(y = [mean + std * dev, mean - std * dev],
              xmin = data.index.min(),
              xmax = data.index.max(),
              color = 'indigo',
              linestyle = ':')
    
    ax.set_title('Ratio ' + cols[0] + '/' + cols[1])
    ax.set_xlabel('Fechas')
    ax.set_ylabel('Ratio')
    #ax.set_xticks(fechas)
    #ax.set_xticklabels(np.datetime_as_string(fechas, unit = 'D'))
    #x_labels = ax.get_xticks()
    #ax.set_xticklabels([pd.to_datetime(e, unit = 'ms').strftime('%Y-%m-%d') for e in fechas])
    #ax.set_xticklabels(fechas[x_labels])
    plt.show()
    