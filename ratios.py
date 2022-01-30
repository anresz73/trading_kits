### class display_ratios ###
### 24/10/2020 ###

import pandas as pd, numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Clases Exceptions

class RatiosException(Exception):
    pass

class PlotException(Exception):
    pass


class Ratios:
    """
    clase crea objetos para obtener y plotear ratios entre los precios ingresados
    """
    def __init__(self, precios):
        """
        clase init
        inicializa a partir de un df con precios y tickers en columnas
        """
        self.precios = precios
        self.ratios = self._make_ratios(self.precios, combine = True)
    
    ### Getters y Setters ###
    def get_prices(self):
        return self.precios.copy()
    
    def set_prices(self, prices):
        self.precios = prices
    
    def get_ratios(self):
        return self.ratios.copy()
    
    def clean_prices(self, prices, date = 'Fecha'):
        """
        devuelve df con fechas en index
        """
        return prices.set_index(date)

    def get_returns(self, log = False):
        """
        devuelve df con retornos
        log : bool - true devuelve retornos logarítmicos
        """
        return self._make_returns(prices = self.get_prices(), log = log)
    
    def get_tickers(self):
        return self.get_prices().columns
    
    ##### Funciones Auxiliares #####

    def _compare_paresd(self, pares, signal = False):
        """
        devuelve df con comparacion entre especies y especiesD dadas en tickers
        tickers: iter - especies a comparar
        signal: bool - agrega columna signal con signal
        """
        pares = np.tile(pares, 2).astype(np.object).reshape(2, len(pares))
        pares[1, :] += 'D'
        #for i in range(pares.shape(1))
        result = pd.DataFrame({e[0] + '/' + e[1] : self._par_ratios(par = e) for e in pares},
                               index = self.get_prices().index)
        if signal == True:
            result['signal'] = result[result.columns[0]] / result[result.columns[1]]
        return result

    def _make_returns(self, prices, log = False):
        """
        devuelve panda con retornos
        log : bool - devuelve retornos porcentuales o logarítmicos
        """
        returns = prices.pct_change().dropna()
        if log:
            returns = np.log1p(returns)
        return returns
    
    def _par_ratios(self, par):
        """
        devuelve un ratio dato un par de tickers
        par : iter- con str de tickers
        """
        if np.isin(par, self.get_prices().columns).all() and len(par) == 2:
            par = self.get_prices()[par]
            return par.values[:, 0] / par.values[:, 1]
        else:
            raise RatiosException('Problema con los tickers.')
    
    def _make_ratios(self, prices, ticker = None, combine = False, col_fechas = 'Fecha'):
        """
        Devuelve panda df con ratios entre ticker y resto aplicada la restricción
        prices : df - precios y tickers en columnas
        ticker : str - ticker a relacionar
        #restriccion : list - tickers a no incluir
        combine : bool - combina todos los tickers
        col_fechas : str - columna con fechas si tiene index datetime
        Sin ticker ni restricción toma el primer ticker de columnas y aplica a todo
        """
        # Loop Controls
        prices = prices.copy()
        if isinstance(prices.index, pd.DatetimeIndex) == False and col_fechas in prices.columns:
            prices.set_index(col_fechas, 
                             inplace = True)
        elif isinstance(prices.index, pd.DatetimeIndex) == False and col_fechas not in prices.columns:
            raise RatiosException('Datos de fechas en dataframe no en columnas.')    

        if combine:
            combine = list(combinations(prices.columns, 2))
        else:
            if ticker is None:
                ticker = prices.columns[0]
            elif ticker not in prices.columns:
                raise RatiosException('Ticker no incluído en especies dadas.')            
            combine = [(ticker, e) for e in prices.columns if e != ticker]

        #Armado de Ratios
        divs = lambda par: prices[par[0]].div(prices[par[1]], axis = 0)
        combine_tickers = [t0 + '/' + t1 for t0, t1 in combine]
        result = pd.concat([divs(e) for e in combine],
                           axis = 1
                          ).rename(columns = dict(enumerate(combine_tickers)))
        return result
    
    # show plot

    def _plot_ratio(self, ratios, col = None, ma = 5, dev = 2, figsize = (12, 6), axes = False):
        """
        plotea una de las columnas
        ratios: panda df con ratios
        col: str - columna a plotear
        """
        ratios = ratios.copy()
        if col is None:
            col = ratios.columns[0]
            
        if isinstance(ratios, pd.DataFrame):
            ratios = ratios[col]
            agg_d = {'m' : 'mean', 's' : 'std'}
            mean, std = ratios.agg(agg_d)
            rmean = ratios.rolling(ma).agg(agg_d)
            if axes == False:
                ax = ratios.plot(kind = 'line',
                                figsize = figsize,
                                color = 'indigo',
                                )
            else:
                ax = ratios.plot(kind = 'line',
                                figsize = figsize,
                                color = 'indigo',
                                ax = axes
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
            if axes == False:
                plt.show()
            else:
                return ax
        elif isinstance(ratios, pd.Series):
            raise PlotEception('Ratios es Series.')
        else:
            raise PlotException('Ratios no es dataframe.')
    

    ###### Funcion Display Auxiliar ######

    def _display_ratios(self):
        """
        Función para plotear relaciones entre pares
        cotizaciones: panda df con cotizaciones
        col_fechas : str - nombre de columna con datetime si el index no es datetime sin col fechas ojo
        """
        #Armado de display widgets usando handler-observe para actualización automática
        #Selecciín Inicial
        #global handler
        handler = {'ratio_ticker': self.get_ratios().columns,
                   'year' : self.get_ratios().index.year.unique(),
                  }

        ##Funciones handler
        #def handler_ratio(change):
        #    handler['ratio'] = change['new']
        #    _update()

        #def handler_year(change):
        #    handler['year'] = change['new']
        #    _update()

        #def handler_compresion(change):
        #    handler['compresion'] = change['new']
        #    _update()    

        #def _ticker_constructor(cots,
        #                        col_fecha = None):
        #    """
        #    devuelve iterable con tickers para usar en tickerbase y tickercomp
        #    dolar: bool - especies en dolar o pesos
        #    legislacion: str - AR-NY-All
        #    col_fecha: columna con datos datetime para eliminar
        #    ticker_extra: str o iterable - ticker(s) extra a eliminar
        #    """
        #    cots = cots.copy()
        #    if isinstance(cots.index, pd.pd.DatetimeIndex) == False and col_fecha is not None:
        #        cots = cots.set_index(col_fecha)
        #
        #    return cots.columns[mask]

        #def _update():
        #    """
        #    funcion update tickerbase dropdown
        #    """
        #    tickerbase.options = _ticker_constructor(cots = cotizaciones,
        #                                            dolar = handler['dolar'],
        #                                            legislacion = handler['legislacion'],
        #                                            ticker_extra = col_fechas)
        #    tickercomp.options = _ticker_constructor(cots = cotizaciones,
        #                                            dolar = handler['dolar'],
        #                                            legislacion = handler['legislacion'],
        #                                            ticker_extra = [col_fechas, handler['tickerbase']])

        #def _display(_ratio,
        #             _ma,
        #             _dev,
        #             _figsize):
        #   """
        #    Función Auxiliar para display de ploteo usando interactive widgets e ipython
        #    """
        #    #Armado de opciones para display
        #    cols = [col_fechas, _tickerbase, _tickercomp]
        #    ratio_par = ratios(cotizaciones[cols])
        #    show_ratio(ratio_par,
        #                col = _tickercomp + '/' + _tickerbase,
        #                ma = _ma,
        #                dev = _dev,
        #                figsize = _figsize
        #                )

        #Layer 1
        ratio_ticker = widgets.Dropdown(options = handler['ratio_ticker'],
                                        description = 'Ratios',
                                        disabled = False
                                        )
        
        year = widgets.Dropdown(options = handler['year'],
                                value = handler['year'][0],
                                description = 'Desde',
                                disabled = False,
                                )

        #compresion = widgets.ToggleButtons(options = ['Max', 'Med', 'Min'],
        #                                    value = handler['legislacion'],
        #                                    description = 'Compresión',
        #                                    button_style = 'success',
        #                                    disabled = False,
        #                                    indent = True
        #                                    )

        #tickercomp = widgets.Dropdown(options = handler['tickercomp'],
        #                             description = 'Ticker 2',
        #                             disabled = False
        #                             )

        #Layer 2
        ma = widgets.SelectionSlider(options = [5, 9, 20, 30, 50, 100, 200],
                                   value = 5,
                                   description = 'MA',
                                   disabled = False,
                                   button_style = 'success',
                                   indent = True
                                  )
        #ma = widgets.ToggleButtons(options = [5, 30, 50, 100, 200],
        #                           value = 5,
        #                           description = 'MA',
        #                           disabled = False,
        #                           button_style = 'success',
        #                           indent = True
        #                          )
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
        ui1 = widgets.HBox([ratio_ticker, year])
        #ui2 = widgets.HBox([tickerbase, tickercomp])
        ui2 = widgets.HBox([ma, dev, figsize])

        widgets.AppLayout(header = ui1,
                         footer = ui2)

        ##Ejecución Handlers
        #dolar.observe(handler_dolar, names = 'value')
        #legislacion.observe(handler_legislacion, names = 'value')
        #tickerbase.observe(handler_tickerbase, names = 'value')

        #Display
        out_d = {'col' : ratio_ticker,
                 'year' : year,
                 'ma' : ma,
                 'dev' : dev,
                 'figsize' : figsize}

        out = widgets.interactive_output(self.plot_ratio,
                                         out_d)

        display(ui1, ui2, out)
        
    ## Osciladores ##
    
    def _macd(self, prices, window = [26, 12, 9], plot = False, fecha = '2020-01'):
        """
        in: prices dataframe o series de precios
        devuelve dataframe macd columnas macd-signal-histogram
        """
        macd = pd.DataFrame()
        macd['macd'] = prices.ewm(span = window[1]).mean() - prices.ewm(span = window[0]).mean()
        macd['signal'] = macd['macd'].ewm(span = window[2]).mean()
        macd['histogram'] = macd['macd'] - macd['signal']
        if plot == False:
            return macd
        else:
            macd = macd[fecha:]
            data = macd['histogram']
            x = np.arange(data.shape[0])
            ylim = macd['macd'].abs().max() * 1.1

            # Colores Histograma
            colores = ['lawngreen', 'forestgreen', 'brown', 'lightcoral']
            colores_n = (data < 0.) * 2 + (data.diff() >= 0.) * 1
            colores = [colores[e] for e in colores_n.values]

            plt.figure(figsize = (12,7))
            plt.bar(x = x, height = data.values, color = colores)
            plt.ylim(-ylim , ylim)

            ax = plt.twinx()
            ax.plot(x, macd['macd'].values, color = 'mediumblue')
            ax.plot(x, macd['signal'].values, color = 'firebrick')
            ax.set_ylim(-ylim , ylim)
            ax.grid(True)

            ax.axhline(y = 0., color = 'k', linestyle = '--', label = 'Media', lw = 1.5)
            #ax.set_xticklabels(np.datetime_as_string(arr = macd.index.values, unit = 'D'))
            #ax.

            plt.show()
        
        
    ##### Funciones #####
    
    def plot_ratio(self, col, ma = 5, dev = 2, year = 2015, figsize = (12, 6), minvals = .2):
        """
        plotea ratios
        """
        r = self.get_ratios()
        # Filtrado por Año y por mínimo de valores not na
        r = r[r.index.year >= year][r.columns[r.notna().sum(axis = 0) / r.shape[0] > minvals]]
        self._plot_ratio(ratios = r, col = col, ma = ma, dev = dev, figsize = figsize)
    
    def plot_compare(self, pares, signal = False, figsize = (15,10)):
        """
        Plotea comparación de ratios entre especies d
        pares: list - pares de tickers a comparar
        """
        # Armado de Ratios
        r = self._compare_paresd(pares = pares, signal = signal)
        ncols = 3 if signal else 2
        # Ploteo
        fig, ax = plt.subplots(nrows = 1, ncols = ncols, figsize = figsize)
        for n in range(ncols):
            ax[n] = self._plot_ratio(ratios = r, col = r.columns[n], axes = ax[n])
        fig.suptitle('Comparación Ratios')
        plt.show()

    def mep(self, ticker, prices = True):
        """
        devuelve df con precio del mep de un bono
        ticker: str - ticker del bono a mepear
        prices: bool - True devuelve los precios del bono en $ y usd
        """
        ticker = [ticker, ticker + 'D']
        result = self.get_prices()[ticker]
        result['mep'] = self._par_ratios(par = ticker)
        return result if prices else result['mep']
    
    def plot_wealth(self,
                    fecha = None,
                    columns = None):
        """
        plotea gráfico a partir de fecha dada con valores wealth absolutos
        fecha : datetime - fecha
        columns : list de str - columnas
        """
        if fecha is None:
            fecha = self.get_returns().index[0]
        wealth = (self.get_returns()[fecha:].fillna(0.) + 1.).cumprod()
        if columns is not None:
            m = np.in1d(ar1 = columns, ar2 = wealth.columns)
            columns = np.array(columns)[m]
            wealth = wealth[columns]
        ax = wealth.plot(figsize = (12, 6), colormap = 'Set1', grid = True)
        ### ((micros[fecha:].pct_change().fillna(0.) + 1.) * [0.5, 0.25, 0.25]).sum(axis = 1).cumprod().plot(ax = ax, grid = True, ls = '--')