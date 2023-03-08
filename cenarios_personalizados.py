import budget
import pandas as pd

def juros(cen,files):
    dfs,budg = files
    date_range = pd.to_datetime(cen.index,format = '%Y-%m')
    cen = pd.Series(cen.values,list(map(lambda x: x[:5] + str(int(x[5:])),cen.index)))
    first_date = date_range[0]
    budg = budg[first_date:].copy()
    budg['Date'] = pd.Series(budg.index).apply(lambda x: str(x.year) + '-' + str(x.month)).values
    budg = budg.set_index('Date')
    cen_caixa = (budg['Cash'] * (cen / 100)).fillna(0).cumsum()
    cen_caixa = pd.Series(cen_caixa.values,pd.to_datetime(cen_caixa.index))

    for i,df in enumerate(dfs):
        df = df[df['Data'] >= first_date].copy()
        df['Despesa Estimada'] = df['Juros Estimados'].cumsum()
        df['Despesa Real'] = (df['Interest (Month)'] * df['Days Dif'].apply(lambda x: 0 if x == 0 else 1)).cumsum()
        df['Erro'] = (df['Despesa Real'] - df['Despesa Estimada']) / df['Despesa Real']
        dfs[i] = df

    cenario = cen.rename('cen').to_frame()
    parciais = []
    for df in dfs:
        temp = df.join(cenario,on = 'Período')
        temp['Taxa Anual'] = ((temp['cen'] + 1) ** 12) - 1 + temp['Spread']
        temp['Taxa Diária Estimada'] = ((temp['Taxa Anual'] + 1) ** (1/252)) - 1
        temp['Taxa Estimada Período'] = ((1 + temp['Taxa Diária Estimada']) ** temp['Days Dif']) - 1
        temp['Taxa Acumulada Estimada'] = budget.composed_interest(temp['Days Dif'].values,temp['Taxa Estimada Período'].values)
        temp['Juros Estimados'] = temp['Principal Balance'] * temp['Taxa Acumulada Estimada']
        temp['Despesa Estimada'] = temp['Juros Estimados'].cumsum()
        temp['Despesa Corrigida'] = temp['Despesa Estimada'] * (1 - (temp['Erro']))
        temp['Diferença Despesa'] = temp['Despesa Real'] - temp['Despesa Corrigida']
        temp = temp[temp['Days Dif'] == 0].drop_duplicates('Período')
        cenario_parcial = pd.Series(temp['Diferença Despesa'].values,index = temp['Período'].values)
        parciais.append(cenario_parcial)
    cenario = budget.sum_series(*parciais).rename('cen')
    final = pd.DataFrame(index = date_range).join(pd.Series(cenario.values,pd.to_datetime(cenario.index,format = '%Y-%m')).rename('cen')).fillna(method = 'ffill').fillna(0)['cen']
    return final + cen_caixa

def ipca(cen,files):
    dfs,cost = files
    cen = pd.Series(cen.values,list(map(lambda x: x[:5] + str(int(x[5:])),cen.index)))
    date_range = pd.to_datetime(cen.index,format = '%Y-%m')
    first_date = date_range[0]
    cen_df = cen.rename(0).to_frame()
    
    # despesas
    cen_df_percent = cen_df.apply(budget.dif_percent)
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
        cenarios.append(budget.sum_series(*parciais))
    final = pd.concat(cenarios,axis = 1)
    final_deb = pd.DataFrame(index = date_range).join(final.set_index(pd.to_datetime(final.index,format = '%Y-%m'))).fillna(method = 'ffill').fillna(0)
    
    final = final_deb + cen_df_costs
    return final[final.columns[0]]

def cambio(cen,rp):
    risco = rp.copy()
    cambio = cen.rename('cen').to_frame().join(risco.set_index('Period')[['USD','Repayment']]).fillna(0)
    cen_df = (cambio['USD'] - cambio['cen']) * cambio['Repayment']
    return pd.Series(cen_df.cumsum().values,pd.to_datetime(cen.index))

def trading(cen,con):
    cen = pd.Series(cen.values,list(map(lambda x: pd.Period(x),cen.index)))
    first_date = cen.index[0]
    last_date = cen.index[-1]
    con = con[(con['Competencia'] >= first_date) & (con['Competencia'] <= last_date)]
    cen_df = cen.rename(0).to_frame()
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
    df = pd.concat(cenarios,axis = 1).cumsum()
    df = df.set_index(df.index.to_timestamp())
    return df[df.columns[0]].dropna()

def calculate(risco,cenarios):
    if risco == 'JUROS':
        calculator = juros
        files = budget.read_cash_dcf()
    if risco == 'INFLACAO':
        calculator = ipca
        files = budget.read_deb_ipca()
    if risco == 'CAMBIO':
        calculator = cambio
        files = budget.read_rp()
    if risco == 'TRADING':
        calculator = trading
        files = budget.read_contracts()
    for name in ['worst','base','best']:
        cenarios[name] = calculator(cenarios[name],files)
    cenarios['probabilidade_de_prejuizo'] = 0.5
    cenarios = cenarios.set_index(pd.to_datetime(cenarios.index,format = '%Y-%m'))
    cenarios.to_csv(f'risco_{risco}.csv')
    budget.upload_file(risco,'risco')
    return cenarios.drop('probabilidade_de_prejuizo',axis = 1)
