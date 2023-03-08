from setup import adlsFileSystemClient,multithread
import pandas as pd
import os
import datetime
import montecarlo
from dateutil.relativedelta import relativedelta
import numpy as np
import random
import json
from bcb import sgs

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

def absolute(serie,base):
    valor_atual = base
    yield valor_atual
    for valor in serie[:-1]:
        valor = valor / 100
        valor_atual += (valor_atual * valor)
        yield valor_atual

def dif_percent(column):
    lista = list(column.diff())[1:] + [None]
    return pd.Series(lista,index = column.index) / column

def ultimo_reajuste(row):
    mes_de_reajuste = row['ReajusteDataBase'].month
    mes_atual = row['Competencia'].month
    ano_atual = row['Competencia'].year
    if mes_atual >= mes_de_reajuste:
        data = pd.Period(f'{ano_atual}-{mes_de_reajuste}')
    else:
        data = pd.Period(f'{ano_atual - 1}-{mes_de_reajuste}')
    return max(data,row['ReajusteDataBase'])

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
            df = pd.read_excel(arquivo,sheet_name = f'{i}ª Emissão CDI',header = 4)
            df = df[df['Filtro'] == 'X'][['Data','Last  CDI','Days','Interest (Month)','Principal Balance']].dropna()
            df['Data'] = pd.to_datetime(df['Data'],format = '%d%m%Y')
            if df['Data'].iloc[0] < datetime.datetime.now():
                df['Last  CDI'] = df['Last  CDI'] / 100
                spread = pd.read_excel(arquivo,sheet_name = '1ª Emissão CDI').iloc[2,9] / 100
                df['Spread'] = spread
                df['Taxa Diária Estimada'] = ((df['Last  CDI'] + spread + 1) ** (1/252)) - 1
                df['Days Dif'] = df['Days'].diff().fillna(df['Days'].iloc[0]).apply(lambda x: 0 if x < 2 else x)
                df['Taxa Estimada Período'] = ((1 + df['Taxa Diária Estimada']) ** df['Days Dif']) - 1
                df['Taxa Acumulada Estimada'] = composed_interest(df['Days Dif'].values,df['Taxa Estimada Período'].values)
                df['Juros Estimados'] = df['Principal Balance'] * df['Taxa Acumulada Estimada']
                df['Período'] = df['Data'].apply(lambda x: str(x.year) + '-' + str(x.month))
                dfs.append(df)
                messages.append(f'{i}ª Emissão CDI: Sucesso')
            else:
                messages.append(f'{i}ª Emissão CDI: Não Realizado')
        except ValueError as e:
            print(e)
            messages.append(f'{i}ª Emissão CDI: ' + str(e))
    return dfs,serie_selic[datetime.datetime.today():].copy()

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
        df = pd.read_excel(arquivo,sheet_name = f'{emissao}ª Emissão IPCA',header = 6)
        df = df[df['Filtro'] == 'X'][['Date','IPCA','Interest','Monetary Variation',' Principal Balance']]
        df['Date'] = pd.to_datetime(df['Date'],format = '%d%m%Y')
        raw = pd.read_excel(arquivo,sheet_name = f'{emissao}ª Emissão IPCA')
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
        df['Período'] = df['Date'].apply(lambda x: str(x.year) + '-' + str(x.month))
        dfs.append(df)

    costs = pd.read_excel("datalake_files/DCF - RP'22 8&04 - Valores.xlsx",sheet_name = 'Budget',header = 5)
    costs = costs.loc[~costs['Unnamed: 1'].isnull()].drop('Unnamed: 0',axis = 1).set_index('Unnamed: 1').T
    costs.columns.name = None
    costs = costs[list(map(lambda x: type(x) == datetime.datetime,costs.index.values))]
    costs = pd.Series(costs['Operational Costs'].values * 100,index = list(map(lambda x: f'{x.year}-{x.month}',costs.index)))

    return dfs,costs

def read_contracts():
    arquivo = 'datalake_files/2021_01_05_12_14_56_wbc.json'
    df = pd.DataFrame.from_records(json.loads(open(arquivo,'r',encoding = 'utf-8').read())['Values'])
    df = df[['ReajusteDataBase','Competencia','QuantSolicitada','Valor','ValorReajustado']].dropna()
    df['ReajusteDataBase'],df['Competencia'] = pd.to_datetime(df['ReajusteDataBase']).apply(lambda x: pd.Period(x,'M')),pd.to_datetime(df['Competencia']).apply(lambda x: pd.Period(x,'M'))
    df['UltimoReajuste'] = df.apply(ultimo_reajuste,axis = 1)
    df['Valor'] = df['Valor'].apply(float)
    df['Total'] = df['QuantSolicitada'] * df['ValorReajustado']
    ipca = sgs.get({'ipca':433},start = '2000-01-01')
    ipca['ipca'] = [valor for valor in absolute(ipca['ipca'].values,1598.41)]
    ipca = ipca.set_index(ipca.index.to_period('M'))
    df = df.join(ipca,on = 'UltimoReajuste')
    df = df.join(ipca,on = 'ReajusteDataBase',rsuffix = '_base')

    # Parte LCA
    lca_anual = pd.read_excel('datalake_files/Projecoes_LCA.xlsx','Base_Anual',header = 11).dropna().set_index('Unnamed: 0').T['IPCA - IBGE (% a.a.)']
    lca_anual = pd.Series(lca_anual.values,index = list(map(lambda x: int(str(x)[:4]),lca_anual.index))).loc[2000:]
    lca_anual = pd.Series([valor for valor in absolute(lca_anual.values,1598.41)],index = lca_anual.index,name = 'ipca')
    df_anual = pd.DataFrame(index = pd.period_range(start = '{}-01-01'.format(lca_anual.index[0]),end = '{}-01-01'.format(lca_anual.index[-1]),freq = 'M'))
    df_anual['ano'] = df_anual.index.year
    df_anual = df_anual.join(lca_anual,on = 'ano')
    df_anual['lca_anual'] = list(df_anual['ipca'].rolling(12).mean())[12:] + ([None] * 12)
    lca_anual = df_anual['lca_anual']
    lca_mensal = pd.read_excel('datalake_files/Projecoes_LCA.xlsx','Base_Mensal',header = 8).drop([0,1,2]).set_index('Período')['IPCA']
    first_year = lca_mensal.index.year[0]
    lca_mensal = pd.Series([valor for valor in absolute(lca_mensal.values,lca_anual[f'{first_year}-01'])],index = lca_mensal.index.to_period('M'),name = 'lca_mensal')
    df = df.join(lca_mensal,on = 'UltimoReajuste')
    df = df.join(lca_mensal,on = 'ReajusteDataBase',rsuffix = '_base')
    df = df.join(lca_anual,on = 'UltimoReajuste')
    df = df.join(lca_anual,on = 'ReajusteDataBase',rsuffix = '_base')
    df['lca_mensal'] = df['lca_mensal'].fillna(df['lca_anual'])
    df['lca_mensal_base'] = df['lca_mensal_base'].fillna(df['lca_anual_base'])
    del(df['lca_anual'])
    del(df['lca_anual_base'])
    df['ValorReajustado'] = (df['lca_mensal'] / df['lca_mensal_base']) * df['Valor']
    df['Total'] = df['QuantSolicitada'] * df['ValorReajustado']

    return df.sort_values('Competencia')

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
    selic['Período'] = selic['date'].apply(lambda x: x[:5] + str(int(x[5:])))
    date_range = pd.to_datetime(selic['date'],format = '%Y-%m')
    selic = selic.set_index('Período')
    cen_df = pd.concat([(selic['prediction'].rename(i) + pd.Series(np.random.normal(scale = std,size = len(selic)),index = selic.index).cumsum()) for i in range(size)],axis = 1)
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
            temp = df.join(cenario,on = 'Período')
            temp['Taxa Anual'] = ((temp[col] + 1) ** 12) - 1 + temp['Spread']
            temp['Taxa Diária Estimada'] = ((temp['Taxa Anual'] + 1) ** (1/252)) - 1
            temp['Taxa Estimada Período'] = ((1 + temp['Taxa Diária Estimada']) ** temp['Days Dif']) - 1
            temp['Taxa Acumulada Estimada'] = composed_interest(temp['Days Dif'].values,temp['Taxa Estimada Período'].values)
            temp['Juros Estimados'] = temp['Principal Balance'] * temp['Taxa Acumulada Estimada']
            temp['Despesa Estimada'] = temp['Juros Estimados'].cumsum()
            temp['Despesa Corrigida'] = temp['Despesa Estimada'] * (1 - (temp['Erro']))
            temp['Diferença Despesa'] = temp['Despesa Real'] - temp['Despesa Corrigida']
            temp = temp[temp['Days Dif'] == 0].drop_duplicates('Período')
            cenario_parcial = pd.Series(temp['Diferença Despesa'].values,index = temp['Período'].values)
            parciais.append(cenario_parcial)
        cenarios.append(sum_series(*parciais))
        if len(cenarios) % 100 == 0:
            print(len(cenarios))
    final = pd.concat(cenarios,axis = 1)
    final = pd.DataFrame(index = date_range).join(final.set_index(pd.to_datetime(final.index,format = '%Y-%m'))).fillna(method = 'ffill').fillna(0)
    return final + cen_caixa

def risco_ipca(ipca,files):
    dfs,cost = files
    size = 10000
    std = ipca['std'].iloc[0]
    ipca['Período'] = ipca['date'].apply(lambda x: x[:5] + str(int(x[5:])))
    date_range = pd.to_datetime(ipca['date'],format = '%Y-%m')
    first_date = date_range[0]
    ipca = ipca.set_index('Período')
    cen_df = (pd.concat([pd.Series(np.random.normal(scale = std,size = len(ipca)),index = ipca.index) for _ in range(size)],axis = 1).cumsum().T + ipca['prediction']).T
    
    # despesas
    cen_df_percent = cen_df.apply(dif_percent)
    cen_df_percent['costs'] = cost
    for cen in cen_df.columns:
        cen_df_percent[cen] = cen_df_percent['costs'] * (1 + cen_df_percent[cen])
    cen_df_costs = cen_df_percent.drop('costs',axis = 1).fillna(0).cumsum().set_index(pd.to_datetime(cen_df.index,format = '%Y-%m'))
    
    # Debentures
    cenarios = []
    for col in cen_df.columns:
        cenario = cen_df[[col]]
        parciais = []
        for df in dfs:
            temp = df.join(cenario,on = 'Período')
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
            parciais.append(temp.set_index('Período')['Impacto'])
        cenarios.append(sum_series(*parciais))
        if len(cenarios) % 100 == 0:
            print(len(cenarios))
    final = pd.concat(cenarios,axis = 1)
    final_deb = pd.DataFrame(index = date_range).join(final.set_index(pd.to_datetime(final.index,format = '%Y-%m'))).fillna(method = 'ffill').fillna(0)
    
    final = final_deb + cen_df_costs
    return final

def risco_cambio(cambio,rp):
    size = 10000
    risco = rp.copy()
    std = cambio['std'].iloc[0]
    cambio = cambio.join(risco.set_index('Period')[['USD','Repayment']],on = 'date').fillna(0)
    cen_df = ((pd.concat([pd.Series(np.random.normal(scale = std,size = len(cambio)),index = cambio.index) for _ in range(size)],axis = 1).cumsum().T + cambio['USD'] - cambio['prediction']) * cambio['Repayment']).T
    return cen_df.cumsum().set_index(pd.to_datetime(cambio['date']))

def risco_generico(df):
    size = 10000
    df['date'] = pd.to_datetime(df['date'],format = '%Y-%m')
    df = df.set_index('date')
    std = df['std'].iloc[0]
    simulation = np.random.normal(size = size,scale = std)
    cen_df = pd.concat([df['prediction'] + sim for sim in simulation],axis = 1)
    cen_df.columns = list(range(size))
    return cen_df.apply(lambda x: pd.Series(shuffle(x.values),index = x.index),axis = 1)

def risco_trading(ipca,con):
    size = 10000
    ipca['date'] = ipca['date'].apply(lambda x: pd.Period(x))
    ipca = ipca.set_index('date')
    std = ipca['std'].iloc[0]
    first_date = ipca.index[0]
    last_date = ipca.index[-1]
    con = con[(con['Competencia'] >= first_date) & (con['Competencia'] <= last_date)]
    cen_df = (pd.concat([pd.Series(np.random.normal(scale = std,size = len(ipca)),index = ipca.index) for _ in range(size)],axis = 1).cumsum().T + ipca['prediction']).T
    cenarios = []
    for i in cen_df.columns:
        cen = cen_df[[i]]
        temp = con.copy()
        temp = temp.join(cen,on = 'UltimoReajuste')
        temp = temp.join(cen,on = 'ReajusteDataBase',rsuffix = '_base')
        temp[f'{i}_base'] = temp[f'{i}_base'].fillna(temp['ipca_base'])
        temp[f'{i}'] = temp[f'{i}'].fillna(temp['ipca'])
        temp['ValorReajustadoCalculado'] = (temp[str(i)] / temp[f'{i}_base']) * temp['Valor']
        temp['TotalCalculado'] = temp['QuantSolicitada'] * temp['ValorReajustadoCalculado']
        temp = temp[['TotalCalculado','Competencia','Total']].groupby('Competencia').sum()
        temp['Diferenca'] = temp['TotalCalculado'] - temp['Total']
        cenarios.append(temp['Diferenca'].rename(i))
        if len(cenarios) % 100 == 0:
            print(len(cenarios))
    df = pd.concat(cenarios,axis = 1).cumsum()
    return df.set_index(df.index.to_timestamp())

def cenarios_preco_energia():
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

def return_cenarios_risco(risco):
    size = 1000
    risco_pred = risco
    if risco == 'TRADING':
        risco_pred = 'INFLACAO'
    df_risco = retrieve_forecast(risco_pred)
    if risco == 'JUROS':
        selic = df_risco
        std = 0.024788480554999645
        selic['Período'] = selic['date'].apply(lambda x: pd.Period(x))
        selic = selic.set_index('Período')
        cen_df = (pd.concat([pd.Series(np.random.normal(scale = std,size = len(selic)),index = selic.index) for _ in range(size)],axis = 1).cumsum().T + selic['prediction']).T
    if risco == 'CAMBIO':
        cambio = df_risco
        std = 0.1281024428289958
        cambio['Período'] = cambio['date'].apply(lambda x: pd.Period(x))
        cambio = cambio.set_index('Período')
        cen_df = (pd.concat([pd.Series(np.random.normal(scale = std,size = len(cambio)),index = cambio.index) for _ in range(size)],axis = 1).cumsum().T + cambio['prediction']).T
    if (risco == 'INFLACAO') or (risco == 'TRADING'):
        ipca = df_risco
        ipca['date'] = ipca['date'].apply(lambda x: pd.Period(x))
        ipca = ipca.set_index('date')
        std = 16.80103821134177
        cen_df = (pd.concat([pd.Series(np.random.normal(0,std,len(ipca)),index = ipca.index) for i in range(size)],axis = 1).cumsum().T + ipca['prediction']).T
    if risco == 'GSF':
        gsf = df_risco
        std = 0.136
        gsf['Período'] = gsf['date'].apply(lambda x: pd.Period(x))
        gsf = gsf.set_index('Período')
        cen_df = pd.concat([(gsf['prediction'].rename(i) + pd.Series(np.random.normal(scale = std,size = len(gsf)),index = gsf.index)) for i in range(size)],axis = 1)
    return cen_df.set_index(cen_df.index.to_timestamp())

def calculate_cenarios(risco,df_risco = pd.DataFrame()):
    risco_pred = risco
    if risco == 'TRADING':
        risco_pred = 'INFLACAO'
    if df_risco.empty:
        df_risco = retrieve_forecast(risco_pred)
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
        con = read_contracts()
        cen_df = risco_trading(df_risco,con)
    return cen_df

def calculate_all():
    for risco in ['JUROS','CAMBIO','INFLACAO','TRADING']:
        cen_df = calculate_cenarios(risco)
        risc_df = pd.DataFrame(index = cen_df.index)
        risc_df['probabilidade_de_prejuizo'] = cen_df.apply(lambda x: sum(x < 0) / len(x),axis = 1)
        risc_df.index.name = 'date'
        cen_df_resumed = pd.concat([cen_df.apply(lambda x: np.percentile(x,i),axis = 1) for i in range(1,100)],axis = 1).rename({i:'{}%'.format(i+1) for i in range(99)},axis = 1)
        risc_df['worst'] = cen_df_resumed['5%']
        risc_df['base'] = cen_df_resumed['50%']
        risc_df['best'] = cen_df_resumed['95%']
        risc_df.to_csv(f'risco_{risco}.csv')
        upload_file(risco,'risco')
        cen_df_resumed.to_csv(f'cenarios_{risco}.csv')
        upload_file(risco,'cenarios')