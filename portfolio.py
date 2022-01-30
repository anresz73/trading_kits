import pandas as pd, numpy as np, yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats
from scipy.optimize import minimize
import random, math
import requests as r

# gold = yf.download('GOLD AUY GLD GDX GDXJ GOEX SLV',
#                  interval = '1mo')['Adj Close']
# gold.columns = gold.columns.str.casefold()
# gold.index = gold.index.to_period('1M')

# gold['ohlc4'] = gold[['open', 'high', 'low', 'close']].sum(axis = 1) / 4.

# np.linspace
# gold = gold[gold.index.year > 2010]
# bins = pd.cut(gold.close, 20)
# g = gold.groupby(bins)['volume'].agg('sum')

def pricebyvolume(df, bins = 20):
    """
    Parameters
    ----------
    df : dataframe columns = ['open', 'high', 'low', 'close', 'volume']
    bins : int, nro bins. The default is 20.
    Returns
    -------
    pandas series index = intervalos / values suma de volumen

    """
    df = df.copy()
    df.columns = df.columns.str.casefold()
    df['vol'] = df.volume / df.volume.sum()
    df['green'] = df.open > df.close
    df['volup'] = df.vol * df.green
    df['voldown'] = df.vol * np.invert(df.green)
    b = pd.cut(df.close, bins)
    return df.groupby(b)[['volup', 'voldown']].sum()

# Volatility

def returns(prices):
    """
    Returns
    -------
    None.

    """
    return prices.pct_change()

def volatility(vol,
               n):
    """
    """
    return vol * np.sqrt(n)
    

# Drawdowns

def drawdown(return_df: pd.Series,
            plot = False):
    """
    Parameters
    ----------
    df : panda dataframe

    Returns
    -------
    drawdowns series

    """
    # df = return_df.pct_change().dropna()
    df = return_df
    wealth_index = 1000. * (1. + df).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    result =  pd.DataFrame({'Wealth' : wealth_index,
                             'Peaks' : previous_peaks,
                             'Drawdown' : drawdowns
                             })
    if plot == False:
        return result
    else:
        #Ploteo
        colormap = 'Spectral'
        fig, ax = plt.subplots(nrows = 2, ncols = 1,
                               figsize = (12, 7),
                               gridspec_kw = {'height_ratios' : [3, 1],
                                             'hspace' : 0.
                                             }
                              )
        result[['Wealth', 'Peaks']].plot(ax = ax[0], colormap = colormap, grid = True)
        ax[0].axes.get_xaxis().set_visible(False)
        ax[0].xaxis.label.set_visible(False)
        result['Drawdown'].plot(ax = ax[1], colormap = colormap)
        fig.suptitle('Ploteo Drawdown ' + df.name)
        plt.show()

def semideviation(r):
    """
    Devuelve la semi desviación aka desviación negativa de r
    r debe ser una Serie o DataFrame
    """
    is_negative = r < 0.
    return r[is_negative].std(ddof = 0)
    

def skewness(r):
    """
    Puede usarse scipy.stats.skew()
    Resuelve skewness de un df o series
    Devuelve float o Series
    """
    demeaned_r = r - r.mean()
    # Uso de la desviación std de la población, set degress of freedom = 0
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r ** 3.).mean()
    return exp / sigma_r ** 3.

def kurtosis(r):
    """
    Puede usarse scipy.stats.kurtosis()
    Resuelve kurtosis de un df o series
    Devuelve float o Series
    """
    demeaned_r = r - r.mean()
    # Uso de la desviación std de la población, set degress of freedom = 0
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r ** 4.).mean()
    return exp / sigma_r ** 4.

def is_normal(r,
             level = 0.01):
    """
    Aplicar el test Jarque-Bera para determinar si la Serie es normal.
    Test aplica al 1% por default.
    Devuelve True si acepta hipósesis de normalidad.
    """
    r = r.dropna()
    try:
        statistic, p_value = scipy.stats.jarque_bera(r)
    except:
        p_value = level
    return p_value > level

def var_historic(r,
                level = 5):
    """
    r series o df
    level = 5 nivel de confianza en %
    devuelve VaR histórica
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return - np.percentile(r, level)
    else:
        raise TypeError('No es Series o DataFrame.')

from scipy.stats import norm

def var_gaussian(r,
                 level = 5,
                 modified = False):
    """
    Devuelve VaR paramétrico Gaussiano de Series o DF
    Si modified = True devuelve VaR calculado con Corner Fish modification
    """
    # Calcula el z score asumiendo dist. normal
    # Modified
    z = norm.ppf(level / 100.)
    if modified:
        # Modifica el Z score tomando en cuenta skewness y kurtosis observada
        s, k = skewness(r), kurtosis(r)
        z = (z + 
                 (z ** 2. - 1.) * s / 6. +
                 (z ** 3. - 3. * z) * (k - 3.) / 24. -
                 (2. * z ** 3. - 5. * z) * (s ** 2.) / 36.
            )
    
    return -(r.mean() + z * r.std(ddof = 0))

def var_montecarlo(r,
                   level = 5,
                   k = 0.1):
    """
    Devuelve VaR por Método Montecarlo de Series o DF
    r : retornos
    level : intervalo de confianza (5 = 95%)
    k : tamaño portcentual de muestra
    """
    # r = r[r < 0.]
    # ind = int(level * k / 100)
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_montecarlo, level = level, k = k)
    elif isinstance(r, pd.Series):
        k = int(r.shape[0] * k)
        samples = random.sample(population = r.to_list(),
                                k = k)
        #return var_historic(r = samples, level = level)
        return - np.percentile(samples, level)
    else:
        raise TypeError('No es Series o DataFrame.')

def cvar_historic(r, level = 5):
    """
    Calcular la CVaR histórica de Series o DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level = level)
    else:
        raise TypeError('No es Series o DataFrame')

def summary_vars(r):
    """
    devuelve df con resumen de las VaRs y CVaR
    """
    funcs = [var_historic, var_montecarlo, var_gaussian, lambda x: var_gaussian(x, modified = True), cvar_historic]
    s_vars = r.agg(func = funcs).T
    s_vars.columns = ['VaR Historic', 'VaR Montecarlo', 'VaR Gaussian', 'VaR Cornish-Fisher', 'CVaR']
    return s_vars

# Efficient Frontier

def annualize_rets(r, periods_per_year):
    """
    Anualiza un set de retornos
    Ojo inferir los períodos por año
    """
    compounded_growth = (1. + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1.

def annualize_vol(r, periods_per_year):
    """
    Anualiza la volatilidad de un set de retornos
    """
    return r.std() * (periods_per_year ** 0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Calcula el sharpe ratio anualizado
    """
    rt_per_period = (1. + riskfree_rate) ** (1. / periods_per_year) - 1.
    excess_ret = r - rt_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights) ** 0.5

def plot_ef2(n_points, er, cov, style = '.-'):
    """
    Plotea 2-asset Efficient Frontier
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError('plot_ef2 puede plotear solamente fronteras de 2 activos.')
    weights = [np.array([w, 1. - w]) for w in np.linspace(0., 1., n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns' : rets,
        'Volatility' : vols
    })
    return ef.plot.line(x = 'Volatility', y = 'Returns', style = style)

def minimize_vol(target_return, er, cov):
    """
    target_ret -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0), ) * n
    return_is_target = {
        'type' : 'eq',
        'args' : (er, ),
        'fun' : lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type' : 'eq',
        'fun' : lambda weights: np.sum(weights) - 1.
    }
    results = minimize(portfolio_vol,
                       init_guess,
                       args = (cov, ),
                       method = 'SLSQP',
                       options = {'disp' : False},
                       constraints = (return_is_target, weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

def optimal_weights(n_points, er, cov):
    """
    -> lista de weights para hacer funcionar el optimizador para minimizar la volatilidad
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def gmv(cov,
        panda = False):
    """
    Devuelve los weights del Global Minimun Volatility porfolio 
    dada una matriz de covarianzas
    """
    n = cov.shape[0]
    result = msr(0, np.repeat(1, n), cov)
    if panda:
        result = pd.Series(result,
                           index = cov.columns)
    return result

def plot_ef(n_points,
            er,
            cov,
            show_cml = False,
            style = '.-',
            riskfree_rate = 0.,
            show_ew = False,
            show_gmv = False):
    """
    Plotea N-asset Efficient Frontier
    cml : capital market line / dot : tangency portfolio
    gmv : global minimun volatility portfolio
    ew : equal weighted portfolio
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns' : rets,
        'Volatility' : vols
    })
    ax = ef.plot.line(x = 'Volatility',
                      y = 'Returns',
                      style = style)
    
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1 / n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Display EW
        ax.plot([vol_ew],
                [r_ew],
                color = 'goldenrod',
                marker = 'o',
                markersize = 10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Display GMV
        ax.plot([vol_gmv],
                [r_gmv],
                color = 'midnightblue',
                marker = 'o',
                markersize = 10)
        
    if show_cml:
        ax.set_xlim(left = 0)
        rf = 0.1
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add CML Capital Market Line
        cml_x = [0., vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x,
                cml_y,
                color = 'green',
                marker = 'o',
                linestyle = 'dashed',
                markersize = 12,
                linewidth = 2)
    return ax

def msr(riskfree_rate, er, cov):
    """
    riskfree_rate + ER + COV -> W
    Devuelve los weights del portfolio que dan el máximo sharpe ratio
    dado una tasa de riesgo libre, retorno esperado y matriz de covarianzas
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0), ) * n

    weights_sum_to_1 = {
        'type' : 'eq',
        'fun' : lambda weights: np.sum(weights) - 1.
    }
    
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Devuelve la inversa del sharpe ratio dado el weight
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return - (r - riskfree_rate) / vol
    
    results = minimize(neg_sharpe_ratio,
                       init_guess,
                       args = (riskfree_rate, er, cov, ),
                       method = 'SLSQP',
                       options = {'disp' : False},
                       constraints = (weights_sum_to_1),
                       bounds = bounds
                      )
    return results.x

# CPPI

def run_cppi(risky_r,
             safe_r = None,
             m = 3.,
             start = 1000.,
             floor = 0.8,
             riskfree_rate = 0.03,
             drawdown = None):
    """
    risky_r : panda con returns MENSUALES 
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: asset value history, risk budget history, risky weight history
    """
    #Set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r.values, columns = ['R'])
    
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate / 12. #Fast way to set all values to a number
    
    # Set up some DataFrames to saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1. - drawdown)
        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1.)
        risky_w = np.maximum(risky_w, 0.)
        # risky_w = np.clip(m * cushion, 0., 1.)
        safe_w = 1. - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        ## Update the Account Value for this time step
        account_value = risky_alloc * (1. + risky_r.iloc[step]) + safe_alloc * (1. + safe_r.iloc[step])
        # Save the values so i can look at the history and plot
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
    
    risky_wealth = start * (1. + risky_r).cumprod()
    backtest_result = {
        'Wealth' : account_history,
        'Risky Wealth' : risky_wealth,
        'Risk Budget' : cushion_history,
        'Risky Allocation' : risky_w_history,
        'm' : m,
        'start' : start,
        'floor' : floor,
        'risky_r' : risky_r,
        'safe_r' : safe_r
    }
    return backtest_result

def summary_stats(r,
                  riskfree_rate = 0.03):
    """
    Datos de ingreso returns en meses
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year = 12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year = 12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate = riskfree_rate, periods_per_year = 12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified = True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        'Annualized Return' : ann_r,
        'Annualized Vol' : ann_vol,
        'Skewness' : skew,
        'Kurtosis' : kurt,
        'Cornish-Fisher VaR (5%)' : cf_var5,
        'Historic CVaR (5%)' : hist_cvar5,
        'Sharpe Ratio' : ann_sr,
        'Max Drawdown' : dd
    })

# Geometric Brownian Motion Model
def gbm(n_years = 10,
        n_scenarios = 1000,
        mu = 0.07,
        sigma = 0.15,
        steps_per_year = 12,
        s_0 = 100.,
        prices = True):
    """
    Evolution of a stock price using a geometric brownian motion model
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year) + 1
    # xi is a random normal number
    # loop directo en matriz
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc = (1. + mu) ** dt, scale = (sigma * np.sqrt(dt)), size = (n_steps, n_scenarios))
    rets_plus_1[0] = 1.
    # to prices
    ret_val = s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else pd.DataFrame(rets_plus_1 - 1.)
    
    # result = pd.DataFrame(rets_plus_1)
    # if prices:
    #     result = s_0 * result.cumprod()
    # return (result - 1.)
    return ret_val

# ALternariva GBM
def gbm_alt(mu = .07,
            sigma = [.15],
            n = 50,
            dt = 0.1,
            x0 = 100,
            random_seed = 1,
            plot = False):
    """
    GBM Alternative
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if not sigma:
        sigma = np.arange(0.8, 2, 0.2)
    else:
        sigma = np.array(sigma)
    
    x = np.exp(
               (mu - sigma ** 2 / 2.) * dt
               + sigma * np.random.normal(0, np.sqrt(dt), size = (len(sigma), n)).T
    )
    x = np.vstack([np.ones(len(sigma)), x])
    x = x0 * x.cumprod(axis = 0)
    
    if not plot:
        return pd.DataFrame(x)
    else:
        plt.plot(x)
        plt.legend(np.round(sigma, 2))
        plt.xlabel("$t$")
        plt.ylabel("$x$")
        plt.title("Realizations of Geometric Brownian Motion with different variances\n $\mu=1$")
        plt.show()


# Show GMB
def show_gbm(n_scenarios,
             mu,
             sigma, s_0 = 100.):
    """
    Draw the results of a stock price evolution under a Geometric Brownian Motion Model
    """
    # s_0 = 100
    prices = gbm(n_scenarios = n_scenarios, mu = mu, sigma = sigma, s_0 = s_0)
    ax = prices.plot(legend = False, color = 'indianred', alpha = 0.5, linewidth = 2, figsize = (12, 5))
    ax.axhline(y = s_0, ls = ':', color = 'k')
    # ax.set_ylim(top = 400)
    # Draw a dot at origin
    ax.plot(0, s_0, marker = 'o', color = 'darkred', alpha = 0.2)

## Show CPPI Monte Carlo

def show_cppi_i(n_scenarios = 50,
                mu = 0.07,
                sigma = 0.15,
                m = 3,
                floor = 0.,
                riskfree_rate = 0.03,
                y_max = 100.):
    """
    Plot the Results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios = n_scenarios, mu = mu, sigma = sigma, prices = False, steps_per_year = 12)
    # risky_r = pd.DataFrame(sim_rets)
    # Run the Backtest
    btr = run_cppi(risky_r = pd.DataFrame(sim_rets), riskfree_rate = riskfree_rate, m = m, start = start, floor = floor)
    wealth = btr['Wealth']
    y_max = wealth.values.max() * y_max / 100.
    ax = wealth.plot(legend = False, alpha = 0.3, color = 'indianred', figsize = (12, 6))
    ax.axhline(y = start, ls = ':', color = 'k')
    ax.axhline(y = start * floor, ls = '--', color = 'r')
    ax.set_ylim(top = y_max)

def show_cppi(n_scenarios = 50, mu = 0.07, sigma = 0.15, m = 3, floor = 0., riskfree_rate = 0.03, steps_per_year = 12,y_max = 100):
    """
    Plot the Results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios = n_scenarios, mu = mu, sigma = sigma, prices = False, steps_per_year = steps_per_year)
    risky_r = pd.DataFrame(sim_rets)
    # Run the Backtest
    btr = run_cppi(risky_r = risky_r, riskfree_rate = riskfree_rate, m = m, start = start, floor = floor)
    wealth = btr['Wealth']
    # Calculate Terminal Wealth Stats
    y_max = wealth.values.max() * y_max / 100.
    terminal_wealth = wealth.iloc[-1]
    
    tw_mean, tw_median = terminal_wealth.mean(), terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start * floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures / n_scenarios
    
    e_shortfall = np.dot(terminal_wealth - start * floor, failure_mask) / n_failures if n_failures > 0 else 0.
    
    # Plot
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows = 1, ncols = 2, sharey = True, gridspec_kw = {'width_ratios' : [3, 2]}, figsize = (24, 9))
    plt.subplots_adjust(wspace = 0.)
    
    wealth.plot(ax = wealth_ax, legend = False, alpha = 0.3, color = 'indianred')
    wealth_ax.axhline(y = start, ls = ':', color = 'k')
    wealth_ax.axhline(y = start * floor, ls = '--', color = 'r')
    wealth_ax.set_ylim(top = y_max)
    
    terminal_wealth.plot(kind = 'hist', ax = hist_ax, bins = 50, ec = 'w', fc = 'indianred', orientation = 'horizontal')
    hist_ax.axhline(y = start, ls = ':', color = 'k')
    hist_ax.axhline(y = tw_mean, ls = ':', color = 'blue')
    hist_ax.axhline(y = tw_median, ls = ':', color = 'purple')
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy = (.7, .9), xycoords = 'axes fraction', fontsize = 24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy = (.7, .85), xycoords = 'axes fraction', fontsize = 24)
    if (floor > 0.01):
        hist_ax.axhline(y = start * floor, ls = '--', color = 'red', linewidth = 3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fail * 100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy = (.7, .7), xycoords = 'axes fraction', fontsize = 24)

## Liabilities / Funding Ratio / Cox Ingestors Ross Model (CIR)

def discount(t, r):
    """
    Compute the price of a pure discount bond 
    that pays a dollar at time t given interest rate r
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    """
    # antes
    # return (1. + r) ** (-t)
    discounts = pd.DataFrame([(r + 1.) ** -i for i in t])
    discounts.index = t
    return discounts
    
def pv(flows, r):
    """
    Computes the present value of a sequence cash flows given by the time (as an index) and amounts
    l: pd.Series indexed by time and values are the amount of each liability
    returns the present value of the sequence
    r can be a scalar, Series or DataFrame with the number of rows matching the num of rows in flows
    """
    # antes l después flows
    # dates = l.index
    # discounts = discount(dates, r)
    # return (discounts * l).sum()
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis = 'rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of some assets given liabilities and interest rate
    """
    # return assets / pv(liabilities, r)
    return pv(assets, r) / pv(liabilities, r)

def inst_to_ann(r):
    """
    Converts short rate to an annualized rate
    """
    #return np.exp(r)-1
    return np.expm1(r)

def ann_to_inst(r):
    """
    Converts annualized to a short rate
    """
    return np.log1p(r)

def cir1(n_years = 10,
         n_scenarios = 1,
         a = 0.05,
         b = 0.03,
         sigma = 0.05,
         steps_per_year = 12,
         r_0 = None):
    """
    Implements CIR model for interest rates
    a : rate of mean reversion
    b : mean of the interest rate
    sigma : standard dev of the interest rate
    r_0 : ann rate at t_0
    """
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    # For small interest rates not very different ann and short
    dt = 1 / steps_per_year
    
    num_steps = int(n_years * steps_per_year) + 1
    shock = np.random.normal(0, scale = np.sqrt(dt), size = (num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    for step in range(1, num_steps):
        r_t = rates[step - 1]
        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
        rates[step] = abs(r_t + d_r_t)
    return pd.DataFrame(data = inst_to_ann(rates), index = range(num_steps))


def cir(n_years = 10,
        n_scenarios = 1,
        a = 0.05,
        b = 0.03,
        sigma = 0.05,
        steps_per_year = 12,
        r_0 = None):
    """
    Generates a random interest rate evolution over time using CIR model
    b and r_0 are assumed to be the annualized rates, not the short rate
    and the returned values are the annualized rates as well
    Prices are zero cupon bonds.
    """
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    # For small interest rates not very different ann and short
    dt = 1 / steps_per_year
    num_steps = int(n_years * steps_per_year) + 1  #Because n_years might be a float
    
    shock = np.random.normal(0, scale = np.sqrt(dt), size = (num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    ## For price generation
    h = math.sqrt(a ** 2 + 2 * sigma ** 2)
    prices = np.empty_like(shock)
    ####
    
    def price(ttm, r):
        _A = ((2 * h * math.exp((h + a) * ttm / 2)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))) ** (2 * a * b / sigma ** 2)
        _B = (2 * (math.exp(h * ttm) - 1)) / (2 * h + (h + a) * (math.exp(h * ttm) - 1))
        _P = _A * np.exp(-_B  * r)
        return _P
    prices[0] = price(n_years, r_0)
    ####
    
    for step in range(1, num_steps):
        r_t = rates[step - 1]
        d_r_t = a * (b - r_t) * dt + sigma * np.sqrt(r_t) * shock[step]
        rates[step] = abs(r_t + d_r_t)
        # Generate prices at time as well
        prices[step] = price(n_years - step * dt, rates[step])
        
    rates = pd.DataFrame(data = inst_to_ann(rates), index = range(num_steps))
    ### For prices
    prices = pd.DataFrame(data = prices, index = range(num_steps))
    ###
    return rates, prices

## GPD Duration Matching

def bond_cash_flows(maturity,
                    principal = 100.,
                    coupon_rate = .03,
                    coupons_per_year = 12):
    """
    Returns a series of cash flows generated by a bond
    Indexed by a coupon number
    """
    n_coupons = round(maturity * coupons_per_year)
    coupon_amt = principal * coupon_rate / coupons_per_year
    coupon_times = np.arange(1, n_coupons + 1)
    cash_flows = pd.Series(data = coupon_amt, index = coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows

def bond_price(maturity,
               principal = 100.,
               coupon_rate = .03,
               coupons_per_year = 12,
               discount_rate = 0.03):
    """
    Price a bond based on bond parameters maturity, principal, coupon rate and coupons per year
    and the prevailing discount rate
    Computes the price of a bond that pays regular coupons until maturity
    al which time principal and final coupon is returned
    This is not designed to be efficient, ilustrate underlying principle behind bond pricing
    If discount rate is a DF, then this is assumed to be the rate of each coupon date
    and the bond value is computed over time.
    i.e. The index of discount_rate DF is assumed to be the coupon number.
    """
    # New Version
    # cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
    # return pv(cash_flows, discount_rate / coupons_per_year)
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index = pricing_dates, columns = discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity - t / coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    else: #Base Case ... Single time period
        if maturity <= 0:
            return principal + principal * coupon_rate / coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate / coupons_per_year)                    

def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows
    """
    discounted_flows = discount(flows.index, discount_rate) * flows
    weights = discounted_flows / discounted_flows.sum()
    return np.average(flows.index, weights = weights)

def match_durations(cf_t, cf_s, cf_l, discount_rate):
    """
    Returns the weight W in cf_s that, along with (1-W) in cf_l have an effective
    duration that matches cf_t
    """
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t) / (d_l - d_s)

def bond_total_return(monthly_prices,
                      principal,
                      coupon_rate,
                      coupons_per_year):
    """
    Computes the total return of a bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    """
    coupons = pd.DataFrame(data = 0, index = monthly_prices.index, columns = monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12 / coupons_per_year, t_max, int(coupons_per_year * t_max / 12), dtype = int)
    coupons.iloc[pay_date] = principal * coupon_rate / coupons_per_year
    total_returns = (monthly_prices + coupons) / monthly_prices.shift() - 1
    return total_returns.dropna()

## Risk Budgeting Strategies

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a backtest of allocation between a two set of returns
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios
    allocator is a function that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested on GHP) as a T x 1 DataFrame.
    Returns a T x N DataFrame of resulting N portfolio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError('r1 and r2 need to be the same shape.')
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError('Allocator returned wights that dont match r1.')
    r_mix = weights * r1 + (1. - weights) * r2
    return r_mix

# Allocator Functions

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
        each columns is an scenario
        each row is the price of timestep
        w1 is the weights in the first portfolio
    Returns an T x N DataFrame of PSP Weights
    """
    return pd.DataFrame(data = w1, index = r1.index, columns = r1.columns)

def glidepath_allocator(r1, r2, start_glide = 1., end_glide = 0.):
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data = np.linspace(start_glide, end_glide, num = n_points))
    paths = pd.concat([path] * n_col, axis = 1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m = 3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP withot going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP.
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP.
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError('PSP and ZC Prices must have the same shape.')
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index = psp_r.index, columns = psp_r.columns)
    for step in range(n_steps):
        floor_value = floor * zc_prices.iloc[step] ## PV of floor assuming today's rates and flat YC
        cushion = (account_value - floor_value) / account_value
        psp_w = (m * cushion).clip(0, 1) # Same as applying min and max
        ghp_w = 1. - psp_w
        psp_alloc = account_value * psp_w
        ghp_alloc = account_value * ghp_w
        # Recompute the new account value at the end of this step
        account_value = psp_alloc * (1. + psp_r.iloc[step]) + ghp_alloc * (1. + ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m = 3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP withot going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP.
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP.
    """
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index = psp_r.index, columns = psp_r.columns)
    for step in range(n_steps):
        floor_value = (1. - maxdd) * peak_value ### floor is based on Prev Peak
        cushion = (account_value - floor_value) / account_value
        psp_w = (m * cushion).clip(0, 1) # Same as applying min and max
        ghp_w = 1. - psp_w
        psp_alloc = account_value * psp_w
        ghp_alloc = account_value * ghp_w
        # Recompute the new account value at the end of this step
        account_value = psp_alloc * (1. + psp_r.iloc[step]) + ghp_alloc * (1. + ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

def terminal_values(rets):
    """
    Returns the final values of dollar at the end of the return period
    """
    return (rets + 1.).prod()

def terminal_stats(rets, floor = 0.8, cap = np.inf, name = 'Stats'):
    """
    Produce a summary statistics on the terminal values per invested dollar
    across a range of N scenarios.
    rets is a T x N DataFrame of returns, where T is the time-step (we asume rets is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name
    """
    terminal_wealth = (rets + 1.).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    # p : probabilities
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = breach.mean() if reach.sum() > 0 else np.nan
    # e : expected
    e_short = (floor - terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap - terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        'mean' : terminal_wealth.mean(),
        'std' : terminal_wealth.std(),
        'p_breach' : p_breach,
        'e_short' : e_short,
        'p_reach' : p_reach,
        'e_surplus' : e_surplus
    }, orient = 'index', columns = [name])
    return sum_stats


# Carga Datos

def get_ind_returns():
    """
    Carga y da formato a Ken French 30 Industries Portfolios Value Weighted Monthly Returns
    """
    ind = pd.read_csv('./csv/ind30_m_vw_rets.csv',
                      header = 0,
                      index_col = 0) / 100.
    ind.index = pd.to_datetime(ind.index, format = '%Y%m').to_period('1M')
    ind.columns = ind.columns.str.strip()
    return ind

# Copy paste -  Pésimo desde programación

def get_ind_nfirms():
    """
    Nro Firms
    """
    ind = pd.read_csv('./csv/ind30_m_nfirms.csv',
                      header = 0,
                      index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = '%Y%m').to_period('1M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    """
    Sizes
    """
    ind = pd.read_csv('./csv/ind30_m_size.csv',
                      header = 0,
                      index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = '%Y%m').to_period('1M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_total_market_index_returns(wealth = False):
    """
    Total Market Index Return
    """
    ind_mktcap = get_ind_nfirms() * get_ind_size()
    total_mktcap = ind_mktcap.sum(axis = 1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis = 0)
    total_mkt_return = (ind_capweight * get_ind_returns()).sum(axis = 1)
    if wealth:
        return drawdown(total_mkt_return).Wealth
    #Ojo rever esto y wealth
    else:
        return total_mkt_return


def get_yf(tickers,
           intervals = ['1d', '1wk', '1mo'],
           period = 'max',
           column = 'Adj Close'):
    """
    Carga datos dato un ticker en days, week, month.
    Devuelve un diccionario keys = ['d', 'w', 'm'] de pandas con retornos
    """
    keys = ['d', 'w', 'm']
    returns = {t[1] : yf.download(tickers,
                               interval = t,
                               period = period)[column].pct_change().dropna() for t in intervals}
    return returns

def get_cer():
    url = r'https://apis.datos.gob.ar/series/api/series/' #'?ids=94.2_CD_D_0_0_10&limit=5000&format=json'
    payload = {'ids' : '94.2_CD_D_0_0_10',
              'format' : 'json',
              'limit' : 5000,
              'start_date' : '2010-01-01'}
    cer = r.get(url = url, params = payload)
    cer = pd.DataFrame.from_dict(cer.json()['data'])
    #cer.json()['meta']
    cer.columns = ['Fecha', 'Cer']    
    cer['Fecha'] = pd.to_datetime(cer['Fecha'])
    cer.set_index('Fecha', inplace = True)
    return cer


# Graficadores

def log_ret(serie,
            n = 200,
            k = 30,
            figsize = (12, 7),
            log = True):
    """
    Grafica Rolling %PNL
    serie : pd.Series - datos a analizar
    n : int - longitud media %PNL y Media Móvil 
    k : int - longitud desvío std móvil
    vol : bool - volatilidad logarítmica o artimética def True
    """
    #n = 72
    log_ret = (serie / serie.shift(n) - 1.).dropna()
    log_ret2 = (serie / serie.rolling(n).mean() - 1).loc[log_ret.index]
    
    #Ploteo
    fig, (ax0, ax1, ax11, ax2) = plt.subplots(4, 1, sharex=True, figsize = figsize)

    serie.loc[log_ret.index].plot(ax = ax0, label = f'XAUT Prices', color = 'indianred')

    log_ret.plot(ax = ax1, color = 'k', lw = .5, label = f'XAUT Rolling %PNL {n} Days')
    ax1.fill_between(log_ret.index, 0., log_ret, where = log_ret > 0., color = 'g')
    ax1.fill_between(log_ret.index, 0., log_ret, where = log_ret < 0., color = 'r')

    log_ret2.plot(ax = ax11, color = 'k', lw = .5)
    ax11.fill_between(log_ret2.index, 0., log_ret2, where = log_ret2 > 0., color = 'g')
    ax11.fill_between(log_ret2.index, 0., log_ret2, where = log_ret2 < 0., color = 'r')

    # ax2 = ax.twinx()
    # ax2.plot(xaut.pct_change().rolling(100).std(), '--k', lw = 1.)
    #k = 24
    #Volatilidad
    volatilidad = np.log(serie/ serie.shift(1)) if log else serie.pct_change()
    (volatilidad.rolling(k).std().loc[log_ret.index] * (252 ** .5)).plot(ax = ax2, style = '--k', lw = 1., label = f'Std Dev Rolling {k} Days')
    
    # Axis
    ax0.grid(axis = 'y')
    ax0.legend()
    ax1.grid(axis = 'y')
    ax1.legend(loc = 'upper left')
    
    fig.subplots_adjust(hspace = 0)

## Varios Estocásticos

def anualidad(A, n, i):
    """
    Devuelve valor presente de una anualidad constante
    A: cuota fija a pagar por período
    n: número de períodos
    i: interés fijo de cada período
    """
    result = A * (1. - (1. + i) ** -n) * (1. + i) / i
    return result

def vp_cuotas(cuota, i, nro_cuotas):
    """
    Devuelve valor presente de una cuota constante
    cuota: cuota fija a pagar por período
    nro_cuotas: número de cuotas
    i: interés fijo de cada período
    """
    vp = cuota * (1. - (1. + i) ** - nro_cuotas) / (1. - (1. + i) ** -1.)
    vp2 = cuota * (1. - (1. + i) ** - nro_cuotas) / i
    return vp, vp2

def vp_estocastico(i, scale, cuotas, n = 20):
    if scale is not None:
        i_periodos = np.random.normal(loc = i, scale = scale, size = (cuotas.shape[0] - 1, n))
        i_periodos = np.vstack((np.ones(n), 
                                i_periodos)).cumprod(axis = 0) ** -1
    else:
        n = 1
        i_periodos = np.repeat(a = i, repeats = (cuotas.shape[0] * n)).reshape(cuotas.shape[0], n)
        pows = np.arange(0, - cuotas.shape[0], -1)
        i_periodos = np.power(i_periodos.T, pows).T
    return np.dot(cuotas, i_periodos)