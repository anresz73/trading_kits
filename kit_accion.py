# Creado 21/10/2020


import pandas as pd, numpy as np, requests
from itertools import combinations
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

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

def ratios(precios,
           ticker = None,
           #restriccion = None,
           combine = False,
           col_fechas = 'Fecha'
           ):
    """
    Devuelve panda df con ratios entre ticker y resto aplicada la restricción
    precios : df - precios y tickers en columnas
    ticker : str - ticker a relacionar
    #restriccion : list - tickers a no incluir
    combine : bool - combina todos los tickers
    col_fechas : str - columna con fechas si tiene index datetime
    Sin ticker ni restricción toma el primer ticker de columnas y aplica a todo
    """
    # Loop Controls
    precios = precios.copy()
    if isinstance(precios.index, pd.DatetimeIndex) == False and col_fechas in precios.columns:
        precios.set_index(col_fechas,
                          inplace = True)
    elif isinstance(precios.index, pd.DatetimeIndex) == False and col_fechas not in precios.columns:
        raise KeyError('Datos de fechas en dataframe no en columnas.')    
    
    if combine:
        combine = list(combinations(precios.columns, 2))
    else:
        if ticker is None:
            ticker = precios.columns[0]
        elif ticker not in precios.columns:
            raise ValueError('Ticker no incluído en especies dadas.')            
        combine = [(ticker, e) for e in precios.columns if e != ticker]
    
    #Armado de Ratios
    divs = lambda par: precios[par[0]].div(precios[par[1]], axis = 0)
    combine_tickers = [t0 + '/' + t1 for t0, t1 in combine]
    result = pd.concat([divs(e) for e in combine],
                       axis = 1
                      ).rename(columns = dict(enumerate(combine_tickers)))
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

###### Funcion Display Central ######

def display_ratios(cotizaciones,
                   col_fechas = 'Fecha'):
    """
    Función para plotear relaciones entre pares
    cotizaciones: panda df con cotizaciones
    col_fechas : str - nombre de columna con datetime si el index no es datetime
    """
    #Armado de display widgets usando handler-observe para actualización automática
    #Selecciín Inicial
    #global handler
    handler = {'ratio': cotizaciones.columns[1:2],
               'original' : cotizaciones,
               'year' : 'L',
               'compresion' : None 
              }
    
    #Funciones handler
    def handler_ratio(change):
        handler['ratio'] = change['new']
        _update()
        
    def handler_year(change):
        handler['year'] = change['new']
        _update()
    
    def handler_compresion(change):
        handler['compresion'] = change['new']
        _update()    
        
    def _ticker_constructor(cots,
                            col_fecha = None):
        """
        devuelve iterable con tickers para usar en tickerbase y tickercomp
        dolar: bool - especies en dolar o pesos
        legislacion: str - AR-NY-All
        col_fecha: columna con datos datetime para eliminar
        ticker_extra: str o iterable - ticker(s) extra a eliminar
        """
        cots = cots.copy()
        if isinstance(cots.index, pd.pd.DatetimeIndex) == False and col_fecha is not None:
            cots = cots.set_index(col_fecha)
        
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
     
    def _display(_ratio,
                 _ma,
                 _dev,
                 _figsize):
        """
        Función Auxiliar para display de ploteo usando interactive widgets e ipython
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
    year = widgets.ToggleButtons(options = ['L', 'M', 'C'],
                                  value = handler['dolar'],
                                description = 'Tiempo',
                                 button_style = 'warning',
                                disabled = False,
                                indent = True
                                )
    
    compresion = widgets.ToggleButtons(options = ['Max', 'Med', 'Min'],
                                        value = handler['legislacion'],
                                        description = 'Compresión',
                                        button_style = 'success',
                                        disabled = False,
                                        indent = True
                                        )
    
    #Layer 2
    tickerbase = widgets.Dropdown(options = handler['tickerbase'],
                                 description = 'Ticker 1',
                                 disabled = False
                                 )
    
    tickercomp = widgets.Dropdown(options = handler['tickercomp'],
                                 description = 'Ticker 2',
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
    
    #Display
    out_d = {'_tickerbase' : tickerbase,
             '_tickercomp' : tickercomp,
             '_ma' : ma,
             '_dev' : dev,
             '_figsize' : figsize}
    
    out = widgets.interactive_output(_display,
                                     out_d)

    display(ui, ui_tickers, ui_tecs, out)

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