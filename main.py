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
import time

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

def show_simulation():
    fig = Figure(figsize = (10,7),dpi = 100)
    fig.add_subplot(111).hist(montecarlo.simulation)
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

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

def thread_selic():
    models.predict_selic()
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

def forecast_selic():
    terminate_window()
    ttk.Label(root,text = 'Aguarde, a função está sendo executada').place(relx=0.5, rely=0.5, anchor='center')
    thread = threading.Thread(target = thread_selic)
    thread.start()

def simulation_done_window(ano,mes):
    terminate_window()
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
    montecarlo.simulate(ano,mes)
    ttk.Button(root,text = 'Visualizar',command = show_simulation).place(relx=0.5,rely=0.5,anchor='center')

def select_mes(ano):
    terminate_window()
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')
    ttk.Button(root,text = 'Voltar',command = lambda: get_file_names(montecarlo.main_info['risco'])).place(relx=0.1,rely=0.2,anchor='center')
    df = montecarlo.main_dataframe
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
    df = montecarlo.main_dataframe
    for ano in df['ano'].unique():
        menu_ano.add_command(label = ano,command = lambda ano=ano: select_mes(ano))
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

def create_forecast_window():
    terminate_window()
    ttk.Button(root, text="Inflação", command=forecast_ipca).place(relx=0.5, rely=0.3, anchor='center')
    ttk.Button(root, text="Câmbio", command=forecast_cambio).place(relx=0.5, rely=0.5, anchor='center')
    ttk.Button(root, text="Juros", command=forecast_selic).place(relx=0.5, rely=0.7, anchor='center')
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

def create_upload_window():
    terminate_window()
    ttk.Button(root, text="Inflação", command=lambda: upload_file('INFLACAO')).place(relx=0.5, rely=0.3, anchor='center')
    ttk.Button(root, text="Câmbio", command=lambda: upload_file('CAMBIO')).place(relx=0.5, rely=0.5, anchor='center')
    ttk.Button(root, text="Juros", command=lambda: upload_file('JUROS')).place(relx=0.5, rely=0.7, anchor='center')
    ttk.Button(root,text = 'Menu',command = create_main_window).place(relx=0.1,rely=0.1,anchor='center')

def create_simulador_window():
    terminate_window()
    ttk.Button(root, text="Inflação", command=lambda: get_file_names('INFLACAO')).place(relx=0.5, rely=0.3, anchor='center')
    ttk.Button(root, text="Câmbio", command=lambda: get_file_names('CAMBIO')).place(relx=0.5, rely=0.5, anchor='center')
    ttk.Button(root, text="Juros", command=lambda: get_file_names('JUROS')).place(relx=0.5, rely=0.7, anchor='center')
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