"""
Creado 17-01-2021
"""

import requests
import pandas as pd, numpy as np
import finnhub
from datetime import datetime

def get_cryptocompare(crypto,
                      token,
                      limite = 100,
                      convertdt = True,
                      moneda = 'USD',
                      compresion = 'd'):
    """
    s: str - símbolo
    token : str - token
    limite : int - 100 default
    convertdt : convierte a datetime
    return: bool - dataframe
    moneda : str - moneda 'USD' default
    """
    
    url = r'https://min-api.cryptocompare.com/data/v2/' #histoday'
    
    compr = {'d' : 'histoday',
             'h' : 'histohour',
             'm' : 'histominute'}
    
    payload = {'fsym' : crypto,
               'tsym' : moneda,
               'limit' : limite,
               'token' : token
        }
    
    json = requests.get(url + compr[compresion],
                        params = payload).json()
    
    if json['Response'] == 'Error':
        raise Exception("Símbolo {0} no encontrado.".format(s))
    elif json['Response'] == 'Success':
        result = pd.DataFrame.from_dict(json['Data']['Data'])
        if convertdt:
            result['datetime'] = pd.to_datetime(result.time,
                                                unit = 's',
                                                utc = True)
        return result
    else:
        return json

    
def from_finnhub(ticker,
                 finnhub_key,
                t1,
                t2 = 'now',
                candle = 'crypto'):
    """
    """
    t1 = to_timestamp(t1)
    t2 = int(datetime.now().timestamp()) if 'now' else to_timestamp(t2)
    client = finnhub.Client(api_key = finnhub_key)
    f_dict = {'crypto' : client.crypto_candles,
             'forex' : client.forex_candles,
             'stock' : client.stock_candles}
    #client.forex_candles('OANDA:XAU_USD', '60', to_timestamp(t1), int(pd.Timestamp.now().timestamp()))
    resp = f_dict[candle](ticker, 'D', t1, t2)
    #client.crypto_candles
    columns_d = {'c' : 'Close',
                'h' : 'High',
                'l' : 'Low',
                'o' : 'Open',
                's' : 'Status',
                't' : 'Timestamp',
                'v' : 'Volume'}
    data = pd.DataFrame(resp).rename(columns = columns_d)    
    #data.rename(columns = columns_d, inplace = True)
    data['Date'] = pd.to_datetime(data['Timestamp'], unit = 's', utc = True)
    #data.set_index(data['Date'], inplace = True)
    #eth_df = eth.set_index(pd.to_datetime(eth.t, unit = 's'))[['o', 'h', 'l', 'c', 'v']]
    #eth_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    #eth_df

    #data.tail()
    return data.set_index(data['Date'])


def to_timestamp(t):
    t = datetime.fromisoformat(t)
    return int(t.timestamp())

def get_symbol_exchange(ticker,
                        finnhub_key):
    """
    devuelve dict en los distintos exchanges
    """
    finnhub_client = finnhub.Client(api_key = finnhub_key)
    crypto_exchanges = finnhub_client.crypto_exchanges()
    forex_exchanges = finnhub_client.forex_exchanges()
    #symbols = finnhub_client.crypto_symbols('BITFINEX')
    #len(symbols)
    # Gold Cryptos : PAXG - PMGT - DGX - XAUT - 
    # USD DAI USDT
    d = {}
    for exchange in exchanges:
        d[exchange] = np.array([ticker in e['symbol'] for e in finnhub_client.crypto_symbols(exchange)]).any()
        todos = [e for e in finnhub_client.crypto_symbols(exchange) if ticker in e['symbol']]
    cont_exchanges = [e[0] for e in d.items() if e[1]]
    #symbols = finnhub_client.crypto_symbols('BITFINEX')
    #mask = np.array(['XAUT' in e['symbol'] for e in symbols])
    #np.array([e for e in symbols])[mask]
    data = np.concatenate([finnhub_client.crypto_symbols(e) for e in crypto_exchanges])
    return cont_exchanges, todos

class Symbols:
    """
    Clase genera objeto para chequear con finnhub
    """
    def __init__(self, finnhub_key):
        """
        Inicializa objeto Symbols para chequear
        finnhub_key : str
        """
        self.symbols_file = 'data_exchange'
        self.finnhub_client = finnhub.Client(api_key = finnhub_key)
        self.crypto_exchanges = self.finnhub_client.crypto_exchanges()
        self.forex_exchanges = self.finnhub_client.forex_exchanges()
        self.all_symbols = self.load_symbols()
    
    def load_symbols(self):
        try:
            all_symbols = np.load(file = self.symbols_file, allow_pickle = True)
        except FileNotFoundError:
            #print('file error')
            all_symbols = self.update_symbols()
            all_symbols.dump(self.symbols_file)
        return all_symbols
    
    def update_symbols(self):
        return np.concatenate([self.finnhub_client.crypto_symbols(e) for e in self.crypto_exchanges] + [self.finnhub_client.crypto_symbols(e) for e in self.forex_exchanges])
    
    def save_symbols(self):
        self.update_symbols().dump(self.symbols_file)
    
    def check_symbol(self, string):
        """
        devuelve symbolos a partir de string
        """
        return [e['symbol'] for e in self.all_symbols if string in e['symbol']]
    
    def get_data(self, symbol):
        """
        devuelve df con datos de symbol
        """
        