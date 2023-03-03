import pandas as pd
import numpy as np
import math
from bcb import sgs,currency
from fredapi import Fred
from datetime import date,datetime
from dateutil.relativedelta import relativedelta
from scipy.optimize import curve_fit
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.models import NBEATSModel
import os
import time
from azure.datalake.store import multithread
from setup import logfile_name,adlsFileSystemClient,upload_file_to_directory
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from retry import retry

# Função que devolve o error e concatena no arquivo de log
@retry(tries = 5,delay = 1)
def error(e):
    log = pd.read_csv(logfile_name)
    log = pd.concat([log,pd.DataFrame({'time':[datetime.now()],'output':['erro'],'error':[repr(e)]})])
    log.to_csv(logfile_name,index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_name,
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)

@retry(tries = 5,delay = 1)
def success(name,output):
    time = datetime.now()
    time_str = str(time).replace('.','-').replace(':','-').replace(' ','-')
    file_name = f'{name}_{time_str}.csv'
    output.index.name = 'date'
    output.to_csv(file_name)
    upload_file_to_directory(file_name,f'DataLakeRiscoECompliance/PrevisionData/Variables/{name}/AI')
    log = pd.read_csv(logfile_name)
    log = pd.concat([log,pd.DataFrame({'time':[time],'output':[file_name],'error':['no errors']})])
    log.to_csv(logfile_name,index = False)
    records = pd.read_csv('records.csv')
    records = pd.concat([records,pd.DataFrame({'file_name':[file_name],'origin':'AI'})])
    records.to_csv('records.csv',index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_name,
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    multithread.ADLUploader(adlsFileSystemClient, lpath='records.csv',
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)

def upload_file_to_directory(file_name,directory):
    multithread.ADLUploader(adlsFileSystemClient, lpath=file_name,
        rpath=f'{directory}/{file_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    time.sleep(1)
    os.remove(file_name)

# Função que converte a variação mensal do IPCA em IPCA absoluto (A série de ipca deve começar em janeiro de 2000)
def absolute(serie):
    valor_atual = 1598.41
    yield valor_atual
    for valor in serie[:-1]:
        valor = valor / 100
        valor_atual += (valor_atual * valor)
        yield valor_atual

# Função que puxa as séries temporais usadas pra prever o IPCA
@retry(TimeoutError,tries = 5,delay = 1)
def get_indicators_ipca(start_date):
    dados = {'selic':432,'emprego':28763,'producao':21859,'comercio':1455,'energia':1406,'IPCA_change':433}
    try:
        dataframe = sgs.get(dados,start = start_date)
    except:
        raise TimeoutError('Erro de conexão com o Banco Central')
    dataframe = dataframe.resample('m').mean()
    dataframe['indice'] = [valor for valor in absolute(dataframe['IPCA_change'].values)]
    del(dataframe['IPCA_change'])
    dataframe = dataframe.dropna()
    return dataframe

# Função que separa os dados em treino e teste
def train_test_split(xdata,ydata,horizonte):
    meses = horizonte * 12
    y_train = ydata.iloc[:-meses]
    x_train = xdata.iloc[-meses - len(y_train):-meses]
    return x_train,y_train

class Scaler:
    def __init__(self,maxes = None):
        if maxes != None:
            self.maxes = np.array(maxes)
    def fit(self,series):
        values = series.values()
        self.maxes = np.array([values[:,i].max() for i in range(values.shape[1])]) * 1.5
    def transform(self,series):
        cols = series.columns
        values = series.values()
        index = series.time_index
        return TimeSeries.from_times_and_values(index,values / self.maxes,columns = cols)
    def fit_transform(self,series):
        values = series.values()
        self.maxes = np.array([values[:,i].max() for i in range(values.shape[1])]) * 1.5
        cols = series.columns
        index = series.time_index
        return TimeSeries.from_times_and_values(index,values / self.maxes,columns = cols)
    def inverse_transform(self,series):
        cols = series.columns
        values = series.values()
        index = series.time_index
        return TimeSeries.from_times_and_values(index,values * self.maxes,columns = cols)

# Função que puxa os dados usados pra prever o câmbio
@retry(TimeoutError,tries = 5,delay = 1)
def get_indicators_cambio(start_date):
    dados = {'selic':432,'ipca':13522,'pib':1208,'emprego':28763}
    try:
        dataframe = sgs.get(dados,start = start_date)
    except:
        raise TimeoutError('Erro de conexão com o Banco Central')
    cy = currency.get('USD', start = start_date,end = str(date.today()))
    dataframe['cambio'] = cy['USD']
    try:
        api_key = '5beeb88b7a5cdd7d4fd8b976e138b52e'
        fred = Fred(api_key = api_key)
        gdp_eua = fred.get_series(series_id = 'GDPC1',observation_start = start_date)
        cpi = fred.get_series(series_id = 'USACPIALLMINMEI',observation_start = f'{int(start_date[:4]) - 1}{start_date[4:]}')
        inflacion_rate = pd.Series(data = (cpi.values[12:] - cpi.values[:-12]) / cpi.values[:-12],index = cpi.iloc[12:].index)
        employment = fred.get_series(series_id = 'CE16OV',observation_start = start_date)
        interest = fred.get_series(series_id = 'INTDSRUSM193N',observation_start = start_date)
    except:
        raise TimeoutError('Erro de conexão com o Federal Reserve')
    dataframe['pib_eua'] = gdp_eua
    dataframe['cpi'] = inflacion_rate
    dataframe['employment'] = employment
    dataframe['interest_rates'] = interest
    dataframe = dataframe.fillna(method = 'ffill')
    dataframe = dataframe.resample('m').mean()
    return dataframe.iloc[:-2]
    
# Função para realizar a captura de dados para cálculo da taxa Selic
@retry(TimeoutError,tries = 5,delay = 1)
def get_indicators_cdi(start_date):
    dados = {'cdi':4391}
    try:
        dataframe = sgs.get(dados,start = start_date)
    except:
        raise TimeoutError('Erro de conexão com o Banco Central')
    dataframe = dataframe.resample('m').mean()
    return dataframe.iloc[:-1]

@retry(TimeoutError,tries = 5,delay = 1)
def get_indicators_gsf():
    multithread.ADLDownloader(adlsFileSystemClient, lpath='gsf.csv', 
        rpath='DataLakeRiscoECompliance/DadosEnergiaCCEE/gsf.csv', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)
    df = pd.read_csv('gsf.csv',index_col = 'index',parse_dates = True)
    os.remove('gsf.csv')
    df['gsf'] = df['Geração'] / df['Garantia Física']
    return df

class Forest:
    def __init__(self,df):
        df = df.copy()
        self.last_date = df.index[-1]
        df['mes'] = df.index.month
        df['quarter'] = df.index.quarter
        ger,gf = df['Geração'].values,df['Garantia Física'].values
        x = df[['mes','quarter']].values
        self.model_ger = RandomForestRegressor(max_depth=10).fit(x,ger)
        self.model_gf = RandomForestRegressor(max_depth=10).fit(x,gf)
    def predict(self,n):
        date_range = pd.date_range(start = self.last_date + relativedelta(months = 1),periods = n,freq = 'MS')
        df = pd.DataFrame(index = date_range)
        df['mes'] = df.index.month
        df['quarter'] = df.index.quarter
        x_fut = df.values
        df['prediction'] = self.model_ger.predict(x_fut) / self.model_gf.predict(x_fut)
        return df['prediction'].values

# Função otilizada para aproximar a curva de câmbio
def simple_square(x,a):
    return (x ** 2) * a

def square(x,a,b,c):
    return ((x ** 2) * a) + (x * b) + c

def linear(x,a,b):
    return (x * a) + b

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def weight(points,expo):
    def ajust(i):
        return sigmoid(i/points) ** expo
    return ajust

def simple_model_predict(serie,projection_points):
    ma = serie.rolling(6).mean().dropna()
    values = ma.values
    last_dif = values[-1] - values[-2]
    line = LinearRegression().fit(np.arange(len(values)).reshape(-1,1),values).predict(np.arange(len(values),len(values) + projection_points).reshape(-1,1)) - values[-1]
    line_derivada = np.cumsum(np.array([last_dif] * projection_points))
    ajuster = weight(10,10)
    final = [(line_derivada[i] * (1 - ajuster(i))) + (line[i] * ajuster(i)) for i in range(projection_points)]
    prediction = final + values[-1]
    micro_diferenca = serie.values[-1] - prediction[0]
    prediction_final = np.array([prediction[i] + (micro_diferenca * (1 - (i / (len(prediction) - 1)))) for i in range(len(prediction))])
    return prediction_final

# Função que prevê o IPCA
def predict_ipca(test = False):
    global run_status
    try:
        # Obtendo os dados
        steps = 60
        df = get_indicators_ipca('2000-01-01')
        ipca = df[['indice']].copy()
        df = df.drop(['indice'],axis = 1)
        main_data,extra_data = TimeSeries.from_dataframe(ipca),TimeSeries.from_dataframe(df)
        scaler_x = Scaler([39.75, 64721022.0, 168.89999999999998, 195.45000000000002, 66381.0])
        scaler_y = Scaler([9683.869411191226])
        main_data_t,extra_data_t = scaler_y.transform(main_data),scaler_x.transform(extra_data)
        model = BlockRNNModel.load('ipcamodel.pt')
        if test:
            prediction_t = model.predict(n = steps,series = main_data_t[:-steps],past_covariates = extra_data_t[:-steps])
            prediction = scaler_y.inverse_transform(prediction_t).values().ravel()
            last = ipca.values.ravel()[- 1 - steps]
            difference = prediction[0] - last
            prediction_final = pd.Series(prediction - difference).rolling(6,1).mean().values
            pred_df = ipca.copy()
            pred_df['prediction'] = ([None] * (len(df) - steps)) + list(prediction_final)
            pred_df['res'] = (pred_df['indice'] - pred_df['prediction'])
            return pred_df
        prediction_t = model.predict(n = steps,series = main_data_t,past_covariates = extra_data_t)
        prediction = scaler_y.inverse_transform(prediction_t).values().ravel()
        last = ipca.values.ravel()[-1]
        difference = prediction[0] - last
        prediction_final = pd.Series(prediction - difference).rolling(6,1).mean().values
        std = 93.34171767852158
        pred_df = pd.DataFrame({'prediction':prediction_final},index = prediction_t.time_index.to_period('M'))
        pred_df['std'] = std
        # Salvando no Log
        success('INFLACAO',pred_df)
        global data_for_plotting
        data_for_plotting = pd.concat([ipca,pred_df[['prediction']].set_index(prediction_t.time_index)])
        run_status = 'O forecast foi gerado e enviado com sucesso para a nuvem'
    except Exception as e:
        error(e)
        run_status = e

def predict_cambio(test = False):
    global run_status
    try:
        # Puxando os dados de câmbio
        steps = 60
        df = get_indicators_cambio('2000-01-01')
        cambio = df[['cambio']].copy()
        df = df.drop(['cambio'],axis = 1)
        main_data,extra_data = TimeSeries.from_dataframe(cambio),TimeSeries.from_dataframe(df)
        scaler_x = Scaler([39.75, 25.86, 13496898298664.924, 64721022.0, 30281.2425, 0.13589636947176387, 238866.0, 9.375])
        scaler_y = Scaler([8.475706451612904])
        main_data_t,extra_data_t = scaler_y.transform(main_data),scaler_x.transform(extra_data)
        model = NBEATSModel.load('cambiomodel.pt')
        if test:
            prediction_t = model.predict(n = steps,series = main_data_t[:-steps],past_covariates = extra_data_t[:-steps])
            prediction = scaler_y.inverse_transform(prediction_t).values().ravel()
            last = cambio.values.ravel()[- 1 - steps]
            difference = prediction[0] - last
            prediction_final = pd.Series(prediction - difference).rolling(6,1).mean().values
            pred_df = cambio.copy()
            pred_df['prediction'] = ([None] * (len(df) - steps)) + list(prediction_final)
            pred_df['res'] = (pred_df['cambio'] - pred_df['prediction'])
            return pred_df
        prediction_t = model.predict(n = steps,series = main_data_t,past_covariates = extra_data_t)
        prediction = scaler_y.inverse_transform(prediction_t).values().ravel()
        last = cambio.values.ravel()[-1]
        difference = prediction[0] - last
        prediction_final = pd.Series(prediction - difference).rolling(6,1).mean().values
        std = 0.9427955605464439
        pred_df = pd.DataFrame({'prediction':prediction_final},index = prediction_t.time_index.to_period('M'))
        pred_df['std'] = std
        # Salvando no Log
        success('CAMBIO',pred_df)
        global data_for_plotting
        data_for_plotting = pd.concat([cambio,pred_df[['prediction']].set_index(prediction_t.time_index)])
        run_status = 'O forecast foi gerado e enviado com sucesso para a nuvem'
    except Exception as e:
        error(e)
        run_status = e

def predict_cdi(test = False):
    global run_status
    try:
        # Puxando e plotando os dados de IPCA
        df = get_indicators_cdi('2000-01-01')
        cdi = df['cdi']
        # Treinando o modelo de SELIC
        anos = 5
        y_train = cdi.iloc[:-12 * anos]
        # Calculando o Erro
        prediction = simple_model_predict(y_train,12 * anos)
        pred_df = df.copy()
        pred_df['prediction'] = [None for _ in range(len(pred_df) - len(prediction))] + list(prediction)
        pred_df['res'] = (pred_df['cdi'] - pred_df['prediction'])
        if test:
            return pred_df
        pred = pred_df.dropna()
        std = math.sqrt(np.square(np.subtract(pred['cdi'].values,pred['prediction'].values)).mean())
        # Treinando novamente o modelo e calculando o Forecast
        prediction = simple_model_predict(cdi,12 * anos)
        pred_df = pd.DataFrame({'prediction':prediction},
            index = pd.period_range(start = cdi.index[-1] + relativedelta(months = 1),periods = len(prediction),freq = 'M'))
        pred_df['std'] = std
        # Salvando no Log
        success('JUROS',pred_df)
        plot_df = pd.DataFrame({'prediction':pred_df['prediction'].values},
            index = pd.date_range(start = cdi.index[-1] + relativedelta(months = 1),periods = len(prediction),freq = 'M'))
        global data_for_plotting
        data_for_plotting = pd.concat([cdi.copy(),plot_df])
        run_status = 'O forecast foi gerado e enviado com sucesso para a nuvem'
    except Exception as e:
        error(e)
        run_status = e

def predict_gsf(test = False):
    global run_status
    try:
        df = get_indicators_gsf()
        gsf = df[['gsf']]
        df = df.drop('gsf',axis = 1)
        # Treinando o modelo de SELIC
        anos = 3
        y_train = df.iloc[:-anos * 12]
        model = Forest(y_train)
        # Calculando o Erro
        prediction = model.predict(12 * anos)
        pred_df = gsf.copy()
        pred_df['prediction'] = [None for _ in range(len(pred_df) - len(prediction))] + list(prediction)
        pred_df['res'] = (pred_df['gsf'] - pred_df['prediction'])
        if test:
            return pred_df
        pred = pred_df.dropna()
        std = math.sqrt(np.square(np.subtract(pred['gsf'].values,pred['prediction'].values)).mean())
        # Treinando novamente o modelo e calculando o Forecast
        model = Forest(df)
        prediction = model.predict(12 * anos)
        pred_df = pd.DataFrame({'prediction':prediction},
            index = pd.period_range(start = gsf.index[-1] + relativedelta(months = 1),periods = len(prediction),freq = 'M'))
        pred_df['std'] = std
        # Salvando no Log
        success('GSF',pred_df)
        plot_df = pd.DataFrame({'prediction':pred_df['prediction'].values},
            index = pd.date_range(start = gsf.index[-1] + relativedelta(months = 1),periods = len(prediction),freq = 'M'))
        global data_for_plotting
        data_for_plotting = pd.concat([gsf.copy(),plot_df])
        run_status = 'O forecast foi gerado e enviado com sucesso para a nuvem'
    except Exception as e:
        error(e)
        run_status = e