from setup import adlsFileSystemClient,multithread
import pandas as pd
import os
import datetime
import montecarlo
from dateutil.relativedelta import relativedelta
import numpy as np
import random

file_links = ["DataLakeRiscoECompliance/ETL/projecao_de_investiamento/DCF - RP'22 8&04 - Valores.xlsx","09 Debt vs Setembro 2022 - RP - Budget 2023.xlsx"]

def download_files(filenames = file_links):
    for file in filenames:
        try:
            multithread.ADLDownloader(adlsFileSystemClient, lpath=os.path.join('datalake_files',file.split('/')[-1]), 
                    rpath=file, nthreads=64, 
                    overwrite=True, buffersize=4194304, blocksize=4194304)
        except Exception as e:
            print(e)

def upload_file(risco,tipo):
    file_name = f'{tipo}_{risco}.csv'
    directory = f'DataLakeRiscoECompliance/RISCOS'
    multithread.ADLUploader(adlsFileSystemClient, lpath=file_name,
        rpath=f'{directory}/{risco}/{file_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    os.remove(file_name)

def read_cash_dcf():
    dcf = pd.read_excel("datalake_files/DCF - RP'22 8&04 - Valores.xlsx",sheet_name = 'Budget',header = 5)
    dcf = dcf.loc[~dcf['Unnamed: 1'].isnull()].drop('Unnamed: 0',axis = 1).set_index('Unnamed: 1').T
    dcf.columns.name = None
    dcf = dcf[list(map(lambda x: type(x) == datetime.datetime,dcf.index.values))]
    serie_selic = dcf[['Cash Accumulated Available']].rename({'Cash Accumulated Available':'Cash'},axis = 1) * 100
    serie_selic = serie_selic.set_index(pd.DatetimeIndex(serie_selic.index))
    return serie_selic[datetime.datetime.today():]

def read_rp():
    rp = pd.read_excel('datalake_files/09 Debt vs Setembro 2022 - RP - Budget 2023.xlsx',sheet_name = 'Daily Calculation prop2019',header = 2)
    rp = rp[['Date','USD','Repayment']].dropna()
    rp = rp.set_index('Date')
    rp = rp[datetime.datetime.today():]
    rp['Period'] = rp.index.to_period('m')
    rp['Period'] = rp['Period'].apply(str)
    return rp

def retrieve_forecast(risco,index = -1):
    montecarlo.find_files(risco)
    try:
        montecarlo.find_datas(-1)
        return montecarlo.main_dataframe
    except Exception as e:
        print(e)

def shuffle(arr):
    new = arr.copy()
    np.random.shuffle(new)
    return new

def risco_juros(selic,dcf):
    size = 10000
    dcf['Date'] = dcf.index.to_period('m')
    dcf['Date'] = dcf['Date'].apply(str)
    risco = dcf.copy()
    std = selic['std'].iloc[0]
    risco = risco.join(selic.set_index('date')[['prediction']],on = 'Date')
    simulation = np.random.normal(size = size,scale = std)
    cen_df = pd.concat([(risco['prediction'] + sim) * risco['Cash'] for sim in simulation],axis = 1,names = list(range(size)))
    return cen_df.apply(lambda x: pd.Series(shuffle(x.values),index = x.index),axis = 1).cumsum()

def risco_cambio(cambio,rp):
    size = 10000
    risco = rp.copy()
    std = cambio['std'].iloc[0]
    risco = risco.join(cambio.set_index('date')[['prediction']],on = 'Period')
    simulation = np.random.normal(size = size,scale = std)
    cen_df = pd.concat([(risco['USD'] - (risco['prediction'] + sim)) * risco['Repayment'] for sim in simulation],axis = 1)
    return cen_df.apply(lambda x: pd.Series(shuffle(x.values),index = x.index),axis = 1).cumsum()

def risco_generico(df):
    size = 10000
    df['date'] = pd.to_datetime(df['date'],format = '%Y-%m')
    df = df.set_index('date')
    std = df['std'].iloc[0]
    simulation = np.random.normal(size = size,scale = std)
    cen_df = pd.concat([df['prediction'] + sim for sim in simulation],axis = 1)
    cen_df.columns = list(range(size))
    return cen_df.apply(lambda x: pd.Series(shuffle(x.values),index = x.index),axis = 1)

def risco_trading():
    size = 10000
    mapa = {'janeiro':1,'fevereiro':2,'março':3,'abril':4,'maio':5,'junho':6,'julho':7,'agosto':8,'setembro':9,'outubro':10,'novembro':11,'dezembro':12}
    df = pd.read_excel('Média Mensal Comercial por Submercado.xlsx',header = 1,index_col = 'Submercado').T.reset_index()
    df['index'] = df['index'].apply(lambda x: datetime(int(x.split()[2]),mapa[x.split()[0]],1))
    df = df.set_index('index')
    df.columns.name = None
    new = df.apply(lambda x: x.mean(),axis = 1).to_frame().rename({0:'energy'},axis = 1).copy()
    new_mean = new['energy'].mean()
    vales = new.apply(lambda x: x['energy'] if new_mean > x['energy'] else None,axis = 1)
    picos = new.apply(lambda x: x['energy'] if new_mean < x['energy'] else None,axis = 1)
    vales_mean = vales.dropna().mean()
    vales_std = vales.dropna().std()
    picos_mean = picos.dropna().mean()
    picos_std = picos.dropna().std()
    def count_holes(serie):
        count = 0
        last = 0
        for valor in serie:
            if np.isnan(valor) and not np.isnan(last):
                count += 1
            last = valor
        return count
    chance_vale = count_holes(vales) / (len(vales.dropna()) - 1)
    chance_pico = 1 - (count_holes(picos) / (len(picos.dropna()) - 1))
    trashold = new['energy'].mean()
    minimo = new['energy'].min()
    def simulate(last,n):
        for _ in range(n):
            rand = random.random()
            if last > trashold:
                if rand < chance_pico:
                    valor = np.random.normal(picos_mean,picos_std)
                else:
                    valor = np.random.normal(vales_mean,vales_std)
            else:
                if rand < chance_vale:
                    valor = np.random.normal(picos_mean,picos_std)
                else:
                    valor = np.random.normal(vales_mean,vales_std)
            last = valor
            yield valor if valor >= minimo else minimo
    last = new['energy'].iloc[-1]
    datetime_index = pd.date_range(start = df.index[-1] + relativedelta(months = 1),periods = 60,freq = 'm')
    sims = []
    for _ in range(size):
        simulator = simulate(last,60)
        sims.append(pd.Series([x for x in simulator]))
    return pd.concat(sims,axis = 1).set_index(datetime_index)

def calculate_cenarios(risco,df_risco = pd.DataFrame()):
    if df_risco.empty:
        df_risco = retrieve_forecast(risco)
    if risco == 'JUROS':
        download_files([file_links[0]])
        dcf = read_cash_dcf()
        cen_df = risco_juros(df_risco,dcf)
    if risco == 'CAMBIO':
        download_files([file_links[1]])
        rp = read_rp()
        cen_df = risco_cambio(df_risco,rp)
    if risco == 'INFLACAO':
        cen_df = risco_generico(df_risco)
    if risco == 'GSF':
        cen_df = risco_generico(df_risco)
    if risco == 'TRADING':
        cen_df = risco_trading()
    return cen_df

def calculate_all():
    for risco in ['JUROS','CAMBIO']:
        cen_df = calculate_cenarios(risco)
        risc_df = pd.DataFrame(index = cen_df.index)
        risc_df['probabilidade_de_prejuizo'] = cen_df.apply(lambda x: sum(x < 0) / len(x),axis = 1)
        risc_df['impacto_mais_provavel'] = cen_df.apply(lambda x: x.mean(),axis = 1)
        risc_df.index.name = 'data'
        cen_df_resumed = pd.concat([cen_df.apply(lambda x: np.percentile(x,i),axis = 1) for i in range(1,100)],axis = 1).rename({i:'{}%'.format(i+1) for i in range(99)},axis = 1)
        risc_df['worst'] = cen_df_resumed['5%']
        risc_df['base'] = cen_df_resumed['50%']
        risc_df['best'] = cen_df_resumed['95%']
        risc_df.to_csv(f'risco_{risco}.csv')
        upload_file(risco,'risco')
        cen_df_resumed.to_csv(f'cenarios_{risco}.csv')
        upload_file(risco,'cenarios')