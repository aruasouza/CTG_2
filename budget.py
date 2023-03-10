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

def composed_interest(days,values):
    results = [values[0] + 1]
    for day,valor in zip(days[1:],values[1:]):
        if day == 0:
            results.append(1)
        else:
            results.append(results[-1] * valor + 1)
    return np.array(results) - 1

def read_cash_dcf():
    dcf = pd.read_excel("datalake_files/DCF - RP'22 8&04 - Valores.xlsx",sheet_name = 'Budget',header = 5)
    dcf = dcf.loc[~dcf['Unnamed: 1'].isnull()].drop('Unnamed: 0',axis = 1).set_index('Unnamed: 1').T
    dcf.columns.name = None
    dcf = dcf[list(map(lambda x: type(x) == datetime.datetime,dcf.index.values))]
    serie_selic = dcf[['Cash Accumulated Available']].rename({'Cash Accumulated Available':'Cash'},axis = 1) * 100
    serie_selic = serie_selic.set_index(pd.DatetimeIndex(serie_selic.index))
    arquivo = 'datalake_files/09 Debt vs Setembro 2022 - RP - Budget 2023.xlsx'
    dfs = []
    messages = []
    for i in range(1,11):
        try:
            df = pd.read_excel(arquivo,sheet_name = f'{i}?? Emiss??o CDI',header = 4)
            df = df[df['Filtro'] == 'X'][['Data','Last  CDI','Days','Interest (Month)','Principal Balance']].dropna()
            df['Data'] = pd.to_datetime(df['Data'],format = '%d%m%Y')
            if df['Data'].iloc[0] < datetime.datetime.now():
                df['Last  CDI'] = df['Last  CDI'] / 100
                spread = pd.read_excel(arquivo,sheet_name = '1?? Emiss??o CDI').iloc[2,9] / 100
                df['Spread'] = spread
                df['Taxa Di??ria Estimada'] = ((df['Last  CDI'] + spread + 1) ** (1/252)) - 1
                df['Days Dif'] = df['Days'].diff().fillna(df['Days'].iloc[0]).apply(lambda x: 0 if x < 2 else x)
                df['Taxa Estimada Per??odo'] = ((1 + df['Taxa Di??ria Estimada']) ** df['Days Dif']) - 1
                df['Taxa Acumulada Estimada'] = composed_interest(df['Days Dif'].values,df['Taxa Estimada Per??odo'].values)
                df['Juros Estimados'] = df['Principal Balance'] * df['Taxa Acumulada Estimada']
                df['Per??odo'] = df['Data'].apply(lambda x: str(x.year) + '-' + str(x.month))
                dfs.append(df)
                messages.append(f'{i}?? Emiss??o CDI: Sucesso')
            else:
                messages.append(f'{i}?? Emiss??o CDI: N??o Realizado')
        except ValueError as e:
            print(e)
            messages.append(f'{i}?? Emiss??o CDI: ' + str(e))
    return dfs,serie_selic[datetime.datetime.today():]

def read_rp():
    rp = pd.read_excel('datalake_files/09 Debt vs Setembro 2022 - RP - Budget 2023.xlsx',sheet_name = 'Daily Calculation prop2019',header = 2)
    rp = rp[['Date','USD','Repayment']].dropna()
    rp = rp.set_index('Date')
    rp = rp[datetime.datetime.today():]
    rp['Period'] = rp.index.to_period('m')
    rp['Period'] = rp['Period'].apply(str)
    return rp

def read_deb_ipca():
    arquivo = 'datalake_files/09 Debt vs Setembro 2022 - RP - Budget 2023.xlsx'
    positions = (15,14)
    dfs = []
    for i in range(2):
        emissao = i + 1
        df = pd.read_excel(arquivo,sheet_name = f'{emissao}?? Emiss??o IPCA',header = 6)
        df = df[df['Filtro'] == 'X'][['Date','IPCA','Interest','Monetary Variation',' Principal Balance']]
        df['Date'] = pd.to_datetime(df['Date'],format = '%d%m%Y')
        raw = pd.read_excel(arquivo,sheet_name = f'{emissao}?? Emiss??o IPCA')
        total = raw.iloc[0,positions[i]]
        juros = raw.iloc[4,positions[i]]
        del(raw)
        df['Taxa'] = juros
        df['Capital'] = total
        df['Days Dif'] = df['Date'].diff().apply(lambda x: x.days).fillna(30)
        df = df[df['Days Dif'] <= 1]
        df['Monetary Variation'] = df['Monetary Variation'].fillna(0)
        subtract = 0
        for i in df.index:
            if df.loc[i,'Days Dif'] == 1:
                subtract += df.loc[i,'Interest']
            df.loc[i,'Capital'] -= subtract
        df = df[df['Days Dif'] != 1]
        df['Capital Dif'] = df['Capital'].diff().fillna(0).cumsum()
        df['Per??odo'] = df['Date'].apply(lambda x: str(x.year) + '-' + str(x.month))
        dfs.append(df)
    return dfs

def retrieve_forecast(risco):
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

def sum_series(*args):
    return pd.concat(args,axis = 1).sum(axis = 1)

def risco_juros(selic,dcf):
    size = 10000
    std = selic['std'].iloc[0]
    selic['Per??odo'] = selic['date'].apply(lambda x: x[:5] + str(int(x[5:])))
    date_range = pd.to_datetime(selic['date'],format = '%Y-%m')
    selic = selic.set_index('Per??odo')
    cen_df = pd.concat([(selic['prediction'].rename(i) + pd.Series(np.random.normal(scale = std,size = len(selic)),index = selic.index)) for i in range(size)],axis = 1)
    first_date = date_range[0]

    # Caixa
    budg = dcf[1]
    budg = budg[first_date:].copy()
    budg['Date'] = pd.Series(budg.index).apply(lambda x: str(x.year) + '-' + str(x.month)).values
    budg = budg.set_index('Date')
    cen_caixa = pd.concat([budg['Cash'].rename(i) * (cen_df[i] / 100) for i in cen_df.columns],axis = 1).fillna(0).cumsum().set_index(date_range)

    # Debentures
    dfs = dcf[0]
    for i,df in enumerate(dfs):
        df = df[df['Data'] >= first_date].copy()
        df['Despesa Estimada'] = df['Juros Estimados'].cumsum()
        df['Despesa Real'] = (df['Interest (Month)'] * df['Days Dif'].apply(lambda x: 0 if x == 0 else 1)).cumsum()
        df['Erro'] = (df['Despesa Real'] - df['Despesa Estimada']) / df['Despesa Real']
        dfs[i] = df
    cenarios = []
    for col in cen_df.columns:
        cenario = cen_df[[col]] / 100
        parciais = []
        for df in dfs:
            temp = df.join(cenario,on = 'Per??odo')
            temp['Taxa Anual'] = ((temp[col] + 1) ** 12) - 1 + temp['Spread']
            temp['Taxa Di??ria Estimada'] = ((temp['Taxa Anual'] + 1) ** (1/252)) - 1
            temp['Taxa Estimada Per??odo'] = ((1 + temp['Taxa Di??ria Estimada']) ** temp['Days Dif']) - 1
            temp['Taxa Acumulada Estimada'] = composed_interest(temp['Days Dif'].values,temp['Taxa Estimada Per??odo'].values)
            temp['Juros Estimados'] = temp['Principal Balance'] * temp['Taxa Acumulada Estimada']
            temp['Despesa Estimada'] = temp['Juros Estimados'].cumsum()
            temp['Despesa Corrigida'] = temp['Despesa Estimada'] * (1 - (temp['Erro']))
            temp['Diferen??a Despesa'] = temp['Despesa Real'] - temp['Despesa Corrigida']
            temp = temp[temp['Days Dif'] == 0].drop_duplicates('Per??odo')
            cenario_parcial = pd.Series(temp['Diferen??a Despesa'].values,index = temp['Per??odo'].values)
            parciais.append(cenario_parcial)
        cenarios.append(sum_series(*parciais))
        if len(cenarios) % 100 == 0:
            print(len(cenarios))
    final = pd.concat(cenarios,axis = 1)
    final = pd.DataFrame(index = date_range).join(final.set_index(pd.to_datetime(final.index,format = '%Y-%m'))).fillna(method = 'ffill').fillna(0)
    return final + cen_caixa

def risco_ipca(ipca,dfs):
    size = 10000
    std = ipca['std'].iloc[0]
    ipca['Per??odo'] = ipca['date'].apply(lambda x: x[:5] + str(int(x[5:])))
    date_range = pd.to_datetime(ipca['date'],format = '%Y-%m')
    first_date = date_range[0]
    ipca = ipca.set_index('Per??odo')
    cen_df = pd.concat([(ipca['prediction'].rename(i) + pd.Series(np.random.normal(scale = std,size = len(ipca)),index = ipca.index)) for i in range(size)],axis = 1)
    cenarios = []
    for col in cen_df.columns:
        cenario = cen_df[[col]]
        parciais = []
        for df in dfs:
            temp = df.join(cenario,on = 'Per??odo')
            temp[col] = temp[col].fillna(temp['IPCA'])
            total = temp['Capital'].iloc[0]
            juros = temp['Taxa'].iloc[0]
            juros_semestral = ((1 + (juros / 100)) ** (1/2)) - 1
            ipca_base = (temp['Capital'].iloc[0] / temp[' Principal Balance'].iloc[0]) * temp[col].iloc[0]
            temp['Principal Balance Calculado'] = (total * temp[col] / ipca_base) + (temp['Capital Dif'] * temp[col] / ipca_base)
            temp['Interest Calculado'] = temp['Principal Balance Calculado'] * juros_semestral
            faltante = (total + sum(temp['Capital'].diff().dropna())) * (-1)
            temp['Monetary Variation Calculado'] = temp['Capital'].diff().apply(lambda x: None if x == 0 else x).fillna(method = 'bfill').fillna(faltante) * (-1) * ((temp['IPCA'] / ipca_base) - 1) * temp['Monetary Variation'].apply(lambda x: 1 if x > 0 else 0)
            temp = temp[temp['Date'] >= first_date]
            temp['Despesa Acumulada'] = (temp['Interest'] + temp['Monetary Variation']).cumsum()
            temp['Despesa Acumulada Calculado'] = (temp['Interest Calculado'] + temp['Monetary Variation Calculado']).cumsum()
            temp['Impacto'] = temp['Despesa Acumulada'] - temp['Despesa Acumulada Calculado']
            parciais.append(temp.set_index('Per??odo')['Impacto'])
        cenarios.append(sum_series(*parciais))
        if len(cenarios) % 100 == 0:
            print(len(cenarios))
    final = pd.concat(cenarios,axis = 1)
    final = pd.DataFrame(index = date_range).join(final.set_index(pd.to_datetime(final.index,format = '%Y-%m'))).fillna(method = 'ffill').fillna(0)
    return final

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
    multithread.ADLDownloader(adlsFileSystemClient, lpath='preco_energia.csv', 
        rpath='DataLakeRiscoECompliance/DadosEnergiaCCEE/preco_energia.csv', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)
    df = pd.read_csv('preco_energia.csv',index_col = 'index',parse_dates = True)
    os.remove('preco_energia.csv')
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

def create_trading_info():
    multithread.ADLDownloader(adlsFileSystemClient, lpath='preco_energia.csv', 
        rpath='DataLakeRiscoECompliance/DadosEnergiaCCEE/preco_energia.csv', nthreads=64, 
        overwrite=True, buffersize=4194304, blocksize=4194304)
    df = pd.read_csv('preco_energia.csv',index_col = 'index',parse_dates = True)
    os.remove('preco_energia.csv')
    datetime_index = pd.date_range(start = df.index[-1] + relativedelta(months = 1),periods = 60,freq = 'm')
    df = pd.DataFrame({'std':[0] * len(datetime_index)},index = datetime_index)
    return df

def calculate_cenarios(risco,df_risco = pd.DataFrame()):
    if df_risco.empty:
        df_risco = retrieve_forecast(risco)
    if risco == 'JUROS':
        # download_files([file_links[0]])
        dcf = read_cash_dcf()
        cen_df = risco_juros(df_risco,dcf)
    if risco == 'CAMBIO':
        # download_files([file_links[1]])
        rp = read_rp()
        cen_df = risco_cambio(df_risco,rp)
    if risco == 'INFLACAO':
        dfs = read_deb_ipca()
        cen_df = risco_ipca(df_risco,dfs)
    if risco == 'GSF':
        cen_df = risco_generico(df_risco)
    if risco == 'TRADING':
        cen_df = risco_trading()
    return cen_df

def calculate_all():
    for risco in ['JUROS','CAMBIO','INFLACAO']:
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