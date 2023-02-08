from setup import adlsFileSystemClient,multithread
import pandas as pd
import os
import datetime
import montecarlo
from dateutil.relativedelta import relativedelta
import numpy as np

file_links = ["DataLakeRiscoECompliance/ETL/projecao_de_investiamento/DCF - RP'22 8&04 - Valores.xlsx","09 Debt vs Setembro 2022 - RP - Budget 2023.xlsx"]

def download_files(filenames = file_links):
    for file in filenames:
        try:
            multithread.ADLDownloader(adlsFileSystemClient, lpath=os.path.join('datalake_files',file.split('/')[-1]), 
                    rpath=file, nthreads=64, 
                    overwrite=True, buffersize=4194304, blocksize=4194304)
        except Exception as e:
            print(e)

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

def retrieve_forecasts(riscos = ['INFLACAO','CAMBIO','JUROS','GSF'],index = -1):
    main_dataframes = {}
    for risco in riscos:
        montecarlo.find_files(risco)
        try:
            montecarlo.find_datas(-1)
            main_dataframes[montecarlo.main_info['risco']] = montecarlo.main_dataframe
        except:
            pass
        return main_dataframes

def risco_juros(selic,dcf):
    size = 10000
    dcf['Date'] = dcf.index.to_period('m')
    dcf['Date'] = dcf['Date'].apply(str)
    risco = dcf.copy()
    std = selic['std'].iloc[0]
    risco = risco.join(selic.set_index('date')[['prediction']],on = 'Date')
    simulation = np.random.normal(size = size) * std
    cen_df = pd.concat([(risco['prediction'] + sim) * risco['Cash'] for sim in simulation],axis = 1,names = list(range(size)))
    return cen_df.cumsum()

def risco_cambio(cambio,rp):
    size = 10000
    risco = rp.copy()
    std = cambio['std'].iloc[0]
    risco = risco.join(cambio.set_index('date')[['prediction']],on = 'Period')
    simulation = np.random.normal(size = size) * std
    cen_df = pd.concat([(risco['USD'] - (risco['prediction'] + sim)) * risco['Repayment'] for sim in simulation],axis = 1,names = list(range(size)))
    return cen_df.cumsum()

def calculate_cenarios(risco,df_risco = pd.DataFrame()):
    if df_risco.empty:
        df_risco = retrieve_forecasts([risco])
    if risco == 'JUROS':
        download_files([file_links[0]])
        dcf = read_cash_dcf()
        cen_df = risco_juros(df_risco,dcf)
    if risco == 'CAMBIO':
        download_files([file_links[1]])
        rp = read_rp()
        cen_df = risco_cambio(df_risco,rp)
    return cen_df

def simulate(mes,ano,cen_df):
    date_selected = pd.to_datetime(f'{ano}-{mes}',format = '%Y-%m')
