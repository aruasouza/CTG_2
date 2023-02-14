from tkinter import ttk
import tkinter as tk
from ttkthemes import ThemedTk
from tkinter import font,Menu
from tkinter import filedialog
import models
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from azure.datalake.store import multithread
import pandas as pd
from datetime import datetime
import re
from setup import upload_file_to_directory,logfile_name,adlsFileSystemClient
import montecarlo
import budget
import numpy as np

def upload_file(Risco):
    create_upload_window()
    file_path = filedialog.askopenfilename()
    try:
        output = pd.read_csv(file_path)
    except Exception:
        ttk.Label(root,text = 'Erro: O arquivo selecionado não é do formato correto.').place(relx=0.5, rely=0.2, anchor='center')
        return
    if len(output) > 72:
        ttk.Label(root,text = 'Erro: O tamanho do arquivo excede o limite máximo').place(relx=0.5, rely=0.2, anchor='center')
        return
    if list(output.columns) != ['date', 'prediction', 'superior', 'inferior', 'std']:
        ttk.Label(root,text = 'Erro: As colunas do arquivo devem ser (nessa ordem): date, prediction, superior, inferior, std').place(relx=0.5, rely=0.2, anchor='center')
        return
    date_sample = output.loc[0,'date'] 
    if not re.match("^\d{4}-\d{2}$", date_sample):
        ttk.Label(root,text = 'Erro: As datas não estão no formato correto (YYYY-mm). Exemplo: 2020-04').place(relx=0.5, rely=0.2, anchor='center')
        return
    time = datetime.now()
    time_str = str(time).replace('.','-').replace(':','-').replace(' ','-')
    file_name = f'{Risco}_{time_str}.csv'
    # Colocando o output para csv e encaminhando-o para o data lake
    upload_file_to_directory(file_path,f'DataLakeRiscoECompliance/PrevisionData/Variables/{Risco}/Manual',file_name)
    log = pd.read_csv(logfile_name)
    log = pd.concat([log,pd.DataFrame({'time':[time],'output':[file_name],'error':['no errors']})])
    log.to_csv(logfile_name,index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath=logfile_name,
        rpath=f'DataLakeRiscoECompliance/LOG/{logfile_name}', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)    
    records = pd.read_csv('records.csv')
    records = pd.concat([records,pd.DataFrame({'file_name':[file_name],'origin':['Manual']})])
    records.to_csv('records.csv',index = False)
    multithread.ADLUploader(adlsFileSystemClient, lpath='records.csv',
        rpath=f'DataLakeRiscoECompliance/LOG/records.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)
    ttk.Label(root,text = 'O arquivo foi enviado com sucesso para a nuvem.').place(relx=0.5, rely=0.2, anchor='center')

def show_forecast():
    fig = Figure(figsize = (10,7),dpi = 100)
    fig.add_subplot(111).plot(models.data_for_plotting)
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

def show_simulation(cen_df,mes,ano):
    date_selected = datetime(int(ano),int(mes),1)
    cen_df = cen_df[:date_selected]
    fig = Figure(figsize = (5,3),dpi = 100)
    risco = montecarlo.main_info['risco']
    ax = fig.add_subplot(111)
    for cenario in cen_df.columns[:1000]:
        ax.plot(cen_df.index,cen_df[cenario],alpha = 0.1,color = 'red')
    ax.set_title(f'Cenários {risco}')
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
    row = cen_df.iloc[-1]
    pior,melhor = int(np.percentile(row.values,25)),int(np.percentile(row.values,75))
    ttk.Label(root,text = f'Pior cenário (25%): {pior} ou menos.').place(relx=0.5, rely=0.76, anchor='center')
    ttk.Label(root,text = f'Cenário médio (50%): de {pior} a {melhor}.').place(relx=0.5, rely=0.8, anchor='center')
    ttk.Label(root,text = f'Melhor cenário (25%): {melhor} ou mais.').place(relx=0.5, rely=0.84, anchor='center')

def create_done_window():
    terminate_window()
    ttk.Label(root,text = models.run_status).place(relx=0.5, rely=0.4, anchor='center')
    if models.run_status == 'O forecast foi gerado e enviado com sucesso para a nuvem':
        ttk.Button(root,text = 'Visualizar',command = show_forecast).place(relx=0.5,rely=0.6,anchor='center')
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

def thread_ipca():
    models.predict_ipca()
    create_done_window()

def thread_cambio():
    models.predict_cambio()
    create_done_window()

def thread_cdi():
    models.predict_cdi()
    create_done_window()

def thread_gsf():
    models.predict_gsf()
    create_done_window()

def forecast_ipca():
    terminate_window()
    ttk.Label(root,text = 'Aguarde, a função está sendo executada').place(relx=0.5, rely=0.5, anchor='center')
    thread = threading.Thread(target = thread_ipca)
    thread.start()

def forecast_cambio():
    terminate_window()
    ttk.Label(root,text = 'Aguarde, a função está sendo executada').place(relx=0.5, rely=0.5, anchor='center')
    thread = threading.Thread(target = thread_cambio)
    thread.start()

def forecast_cdi():
    terminate_window()
    ttk.Label(root,text = 'Aguarde, a função está sendo executada').place(relx=0.5, rely=0.5, anchor='center')
    thread = threading.Thread(target = thread_cdi)
    thread.start()

def forecast_gsf():
    terminate_window()
    ttk.Label(root,text = 'Aguarde, a função está sendo executada').place(relx=0.5, rely=0.5, anchor='center')
    thread = threading.Thread(target = thread_gsf)
    thread.start()

def simulation_done_window(mes,ano):
    terminate_window()
    cen_df = budget.calculate_cenarios(montecarlo.main_info['risco'],montecarlo.main_dataframe)
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
    ttk.Button(root,text = 'Visualizar',command = lambda: show_simulation(cen_df,mes,ano)).place(relx=0.5,rely=0.5,anchor='center')

def select_mes(ano,df):
    terminate_window()
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
    ttk.Button(root,text = 'Voltar',command = lambda: get_file_names(montecarlo.main_info['risco'])).place(relx=0.1,rely=0.2,anchor='center')
    menu_mes = Menu(root)
    for mes in df[df['ano'] == ano]['mes']:
        menu_mes.add_command(label = mes,command = lambda mes=mes: simulation_done_window(mes,ano))
    ttk.Menubutton(root,text = 'Selecionar Mês',menu = menu_mes).place(relx=0.5, rely=0.5, anchor='center')

def select_ano(index):
    terminate_window()
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
    ttk.Button(root,text = 'Voltar',command = lambda: get_file_names(montecarlo.main_info['risco'])).place(relx=0.1,rely=0.2,anchor='center')
    montecarlo.find_datas(index)
    menu_ano = Menu(root)
    df = montecarlo.main_dataframe.copy()
    df['ano'] = df['date'].apply(lambda x: x.split('-')[0])
    df['mes'] = df['date'].apply(lambda x: x.split('-')[1])
    for ano in df['ano'].unique():
        menu_ano.add_command(label = ano,command = lambda ano=ano: select_mes(ano,df))
    ttk.Menubutton(root,text = 'Selecionar Ano',menu = menu_ano).place(relx=0.5, rely=0.5, anchor='center')

def get_file_names(risco):
    terminate_window()
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
    montecarlo.find_files(risco)
    info = montecarlo.main_info
    menu = Menu(root)
    for i in range(info['size'] - 1,-1,-1):
        menu.add_command(label = info['full_strings'][i],command = lambda i=i: select_ano(i))
    ttk.Menubutton(root,text = 'Selecionar Arquivo',menu = menu).place(relx=0.5, rely=0.5, anchor='center')

def do_trading():
    pass

def create_forecast_window():
    terminate_window()
    ttk.Button(root, text="Inflação", command=forecast_ipca).place(relx=0.5, rely=0.3, anchor='center')
    ttk.Button(root, text="Câmbio", command=forecast_cambio).place(relx=0.5, rely=0.4, anchor='center')
    ttk.Button(root, text="Juros", command=forecast_cdi).place(relx=0.5, rely=0.5, anchor='center')
    ttk.Button(root, text="GSF", command=forecast_gsf).place(relx=0.5, rely=0.6, anchor='center')
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

def create_upload_window():
    terminate_window()
    ttk.Button(root, text="Inflação", command=lambda: upload_file('INFLACAO')).place(relx=0.5, rely=0.3, anchor='center')
    ttk.Button(root, text="Câmbio", command=lambda: upload_file('CAMBIO')).place(relx=0.5, rely=0.4, anchor='center')
    ttk.Button(root, text="Juros", command=lambda: upload_file('JUROS')).place(relx=0.5, rely=0.5, anchor='center')
    ttk.Button(root, text="GSF", command=lambda: upload_file('GSF')).place(relx=0.5, rely=0.6, anchor='center')
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

def create_simulador_window():
    terminate_window()
    ttk.Button(root, text="Inflação", command=lambda: get_file_names('INFLACAO')).place(relx=0.5, rely=0.3, anchor='center')
    ttk.Button(root, text="Câmbio", command=lambda: get_file_names('CAMBIO')).place(relx=0.5, rely=0.4, anchor='center')
    ttk.Button(root, text="Juros", command=lambda: get_file_names('JUROS')).place(relx=0.5, rely=0.5, anchor='center')
    ttk.Button(root, text="GSF", command=lambda: get_file_names('GSF')).place(relx=0.5, rely=0.6, anchor='center')
    ttk.Button(root, text="Trading", command=do_trading).place(relx=0.5, rely=0.7, anchor='center')
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

def create_main_window():
    terminate_window()
    ttk.Button(root, text="Gerar Forecasts", command=create_forecast_window).place(relx=0.5, rely=0.3, anchor='center')
    ttk.Button(root, text="Subir Forecast Personalizado", command=create_upload_window).place(relx=0.5, rely=0.5, anchor='center')
    ttk.Button(root, text="Simular Cenários", command=create_simulador_window).place(relx=0.5, rely=0.7, anchor='center')

def terminate_window():
    for element in root.winfo_children():
        element.destroy()

root = ThemedTk(theme="adapta")
root.geometry("800x600")
style = ttk.Style()
custom_font = font.Font(family="Montserrat", size=15)
style.configure("TButton", font = custom_font)

create_main_window()

root.mainloop()