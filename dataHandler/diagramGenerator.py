"""####---------------------------------------------------------<           SETUPS          >---------------------------------------------------------####""" 

#--> Bibliotecas
#-> Minhas
from CSVmodifier import *
from addresses import *

#-> Para o csv
import pandas as pd

#-> Gráficos
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#-> Estilo do gráfico
from matplotlib import style
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#-> Calculo dos gráficos
from scipy.optimize import curve_fit
from operator import itemgetter, attrgetter
from sklearn.metrics import r2_score
from scipy import integrate

#-> Calculos Normais
import numpy as np


#--> Funções necessárias
#-> Calculos

def criar_multiplicador_formatter(multiplicador):
    def formatter(valor, posicao):
        # 'multiplicador' é "lembrado" pela função aninhada
        return f'{valor * multiplicador:.3f}'
    return formatter

def gaussCalc(x, a, x0, sigma): 
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

def exponentialCalc(x, a, b, c):
    return a * np.exp(b * x) + c

def linearCalc(x, a, b):
    return (a*x) + b

def powerLawCalc(x, a, b):
    return pow(a * x, -b)

def fitLinear(positionX, dfDeviation):

    # Acha as posições mais adequadas para o fit
    popt, pcov = curve_fit(linearCalc,positionX, dfDeviation)
    a,b = popt
    positionY = linearCalc(positionX, a, b)

    # Valor do adequamento
    fitness = round(r2_score(dfDeviation, positionY), 2)

    # Imprimi as características do fit
    label = "Linear "
    print(label)
    print("Angular Coefficient: ", round(a,2))
    print("Linear Coefficient: ", round(b,2))


    return positionY, fitness, label

def fitExponential(positionX, dfDeviation):
    
    # Acha as posições mais adequadas para o fit
    popt, pcov = curve_fit(exponentialCalc,positionX, dfDeviation, p0 = (32.68, 0.11, 0), maxfev=50000)
    a,b, c= popt
    positionY = exponentialCalc(positionX, a, b, c)

    # Valor do adequamento
    fitness = round(r2_score(dfDeviation, exponentialCalc(positionX, a, b, c)), 2)

    # Imprimi as características do fit
    label = "Exponential "
    print(label)
    print("a: ", a)
    print("b: ", round(b,2))
    print("c: ", round(c,2))

    return positionY, fitness, label

def fitPowerLaw(positionX, dfDeviation):

    # Acha as posições mais adequadas para o fit    
    popt, pcov = curve_fit(powerLawCalc,positionX, dfDeviation, p0 = (1,1))
    a,b = popt
    positionY = powerLawCalc(positionX, a, b)

    # Valor do adequamento
    fitness = round(r2_score(dfDeviation, positionY), 2)

    # Imprimi as características do fit
    label = "Power law "
    print(label)
    print("Angular Coefficient: ", round(a,2))
    print("b: ", round(b,2))

    return positionY, fitness, label

"""####---------------------------------------------------------<           ESTILO          >---------------------------------------------------------####""" 
#-> Geral

# Nomeia os eixos X e Y, além disso coloca um titulo para a figure e o gráfico
def nominationAxes(
        fig, axes, # ESSENCIAIS
        axisX : str, axisY : str, # CARACTERÍSTICAS
        withoutAxisX: bool = False, withoutAxisY:bool = False # PARÂMETROS
    ) -> None: 

    if withoutAxisX:
        axisX = ""
    
    if withoutAxisY:
        axisY = ""

    axes.set_xlabel(axisX, fontsize = 25)
    axes.set_ylabel(axisY, fontsize = 25)



# Define o tamanho do eixo X e Y
def xAndYSlim(
        axes, # ESSENCIAL
        parameterX, parameterY, startX : int = 0, startY : int = 0, # CARACTERÍSTICAS
        xYes : bool = True, yYes: bool = True, powYes : bool = True # PARÂMETROS
    ) -> None:

    finalY = parameterY
    finalX = parameterX

    # Caso os eixos estejam em potência de 10
    if powYes:
        if pow(10,len(str(parameterX))) / parameterX != 10:
            finalX = pow(10,len(str(parameterX)))

        if pow(10,len(str(parameterY))) / parameterY != 10:
            
            finalY = pow(10,len(str(parameterY)))
    else:
        finalY = finalY + (finalY / 5)

    if xYes:
        axes.set_xlim(startX,finalX)

    if yYes: 
        axes.set_ylim(startY,finalY)



# Arruma alguns estilos
def diagramStyle(
        axes, # ESSENCIAL
        axisX, axisY, Xsize: int = 20, Ysize : int = 20, angle : int = 0, # CARACTERÍSTICA
        boxplot : bool = False, all : bool = False, bTime : bool = False, science :bool = False, withoutValX : bool = False, IndexMult : int = 0 # PARÂMETROS
    ) -> None:

    # Definimos o tamanho, e o angulo dos eixos
    plt.xticks(fontsize = Xsize, rotation = angle)
    plt.yticks(fontsize = Xsize)
    
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    if all == False:
        axes.tick_params(left = False, bottom = False)
        axes.set_ylim([min(axisY)*0.99, max(axisY)*1.01])
    else:
        axes.tick_params(axis='both', which='both', direction='in')


    # Corrige o xLabel
    if boxplot == False and bTime == False:
        labelXWithouBoxplot(axes, axisX)

    if boxplot == True and bTime == False:
        labelXWithBoxplot(axes, axisX, withoutValX)

    # Corrige o yLabel
    labely(axes, axisY, boxplot, bTime)

    # Coloca notação cientifica
    if science == True:
        axes.yaxis.set_major_formatter(ticker.FuncFormatter(criar_multiplicador_formatter(pow(10, IndexMult))))



# Salva o gráfico
def saveDiagram(
        whereToSave : str, # LOCALIZAÇÃO
        networkName : str, type : str, figure : str# CARACTERÍSTICA
    ) -> None:
    
    # Caso seja necessário salvar em png apenas coloque como False
    pdf : bool = True

    if pdf:

        # "saveAddres" é o caminho completo até a pasta que as figuras serão salvas
        # "whereToSave" é o endereço da pasta em que o gráfico será salvo para aquele tipo de regra
        # "figure" é qual informação tá sendo passada, se é a riqueza, o fitness ou a ligação
        # "type" é uma nomenclatura simplificada para saber quais regras foram aplicadas

        plt.savefig(f"{saveAddress}{whereToSave}/{figure}{networkName}{type}.pdf", 
            transparent=None, dpi=150, format="pdf",
            metadata=None, bbox_inches=None, pad_inches=0.1,
            facecolor='auto', edgecolor='auto', backend=None
        )

    else:
        plt.savefig(f"{saveAddress}{whereToSave}/{figure}{networkName}{type}.png")



#-> Eixo X
def labelXWithouBoxplot(axes, axisX) -> None:
    ticksX = np.arange(0,max(axisX), max(axisX)/4)
    ticksX = np.append(ticksX, max(axisX))
    axes.set_xticks(ticksX) 


def labelXWithBoxplot(axes, axisX, withoutValX) -> None:

    axisXSize = range(1, len(axisX))
    numticks = [0]
    null = ['']
    for i in axisXSize:
        numticks.append(i)
        if withoutValX:
            null.append('')

    axes.set_xticks(numticks)

    
    

    start = axisX[0]
    ticksX = [round(start, 0)]
    for i in axisXSize:
        if start<100:
            sum = 10
            maxi = 4
        else:
            sum = 100
            maxi = 8

        if axisX[i] >= start + sum or i == max(axisXSize):
            if i!= max(axisXSize) and i+maxi > max(axisXSize):
                ticksX.append('')
            else:
                start = axisX[i]
                ticksX.append(round(start,0))
        else:
            ticksX.append('')
    
    
    
    
    if withoutValX:
        axes.set_xticklabels(null)

    axes.set_xticklabels(ticksX)

    

#-> Eixo Y
def labely(
        axes ,axisY, # ESSENCIAL 
        boxplot, bTime # PARÂMETROS
    ) -> None:
    i = 3
    areaY = []
    while len(areaY) != 4:
        if boxplot == True:
            areaY = np.round(np.arange(min(axisY), max(axisY), (max(axisY)/i)), decimals=2)
        else:
            areaY = np.arange(min(axisY), max(axisY), (max(axisY)/i))
        i = i +1


    if max(axisY)*0.9 < areaY[len(areaY) -1]:
        areaY = areaY[:-1]
    ticksY = np.append(areaY, max(axisY))
    
    axes.set_yticks(ticksY) 

    
"""####---------------------------------------------------------<    GERADORES DE GRÁFICO    >---------------------------------------------------------####""" 

#-> Mini geradores ==========================================================================================================================================================

## Função individualGraph
# Vai gerar gráficos separados para serem juntados no allGraphs
def individualGraph(
        axes, # ESSENCIAL
        networkAddress : str, networkName : str, table : str, # LOCALIZAÇÃO
        color : str, symbol : str, opacity : float, size:float, # CARACTERÍSTICAS
        labelY_N : bool = False # PARÂMETROS
    ) -> None:

    # Confere se a tabela passada existe, se não existir para o programa
    if existingFile(networkAddress, table) == False:
        return None
    
    # Abre o arquivo
    df = pd.read_csv(f"{dataAddress}{networkAddress}/{table}.csv")

    
    # Definição de variáveis
    Wealth = df['din']
    Probability = df['quant']
    Probability = Probability/(sum(Probability*100))

    # Cria a imagem daquele gráfico
    axes.scatter(Wealth, Probability, marker = symbol, edgecolors= color, facecolor = 'none', alpha = opacity, s = size)
    if labelY_N == True:
        axes.scatter(0,0,marker = '.', label = f'{networkName}', color=color, s=50)
        
    
    diagramStyle(axes, Wealth, Probability, all=True)
    # Plotamos a legenda
    axes.legend()

## Função lorenzIndividualGraph
# Vai gerar gráficos separados para serem juntados no lorenzGraph
def lorenzIndividualGraph(
        axes, # ESSENCIAL
        networkAddress : str, networkName : str, table : str, # LOCALIZAÇÃO
        color : str, giniReturn, lineStyle : str = "-", # CARACTERÍSTICAS
        labelY_N : bool = False # PARAMETRO  
    ) -> list[float]:

    # Abrimos o arquivo e ordemas em ordem crescente com base no dinheiro
    
    if existingFile(networkAddress, table) == False:
    
        return None
    
    df = pd.read_csv(f"{dataAddress}{networkAddress}/{table}.csv")
    df = df.sort_values(by = 'din')

    # Definimos nossas variaveis
    equalityLine = []
    Wealth = df['din'].values.tolist()
    Quantity = df['quant'].values.tolist()
    totalWealth = []
    for i in range(len(Wealth)):
        totalWealth.append(Wealth[i] * Quantity[i])
    totalWealth = sum(totalWealth)

    # Criamos a linha perfeita
    for i in np.arange(0, 1 + (1/len(Wealth)), 1/len(Wealth)):
        equalityLine.append(i)

    # Definimos a lista população e colocamos seus valores dentro dela
    Population = [0]
    sumQuantity = 0
    for indexQuant in Quantity:
        sumQuantity += indexQuant / sumQuantity(Quantity)
        Population.append(sumQuantity)

    # Definimos a lista de acumulação da populção(income) e colocamos seus valores dentro dela
    income = [0]
    sumWealth = 0
    for i, indexWealth in enumerate(Wealth):
        sumWealth += (indexWealth * Quantity[i]) / totalWealth
        income.append(sumWealth)

    # Fazemos o calculo para descobrir a area abaixo da linha
    gini = 0.5 - integrate.simpson(income, Population)
    giniReturn.append(f"Gine da {networkName}: {round(gini/0.5, 3)}")

    # Geramos a imagem daquele gráfico
    if labelY_N == True:
        axes.plot(Population, income, color = color, linestyle = lineStyle, lw = 2.2, label = f'{networkName}',alpha = 0.6)
    else:
        axes.plot(Population, income, color = color, linestyle = lineStyle,alpha = 1)


    # Retornamos a linha perfeita
    return equalityLine


#-> Gráficos de Riq x Prob ==========================================================================================================================================================

## Função moneyGrafico
# Serve para criar os gráficos de Probability x Quantity de Riqueza
def wealthGraph(
        networkAddress : str, whereToSave : str, networkName : str, type : str, table : str, # LOCALIZAÇÃO
        gauss : bool = True, zoomOrNo : bool = False, initialState :bool = False # DEFINIÇÃO
    ) -> None:  


    # Confere se o Arquivo passado existe
    if existingFile(networkAddress, table) == False:
        return None
    
    # Abre a tabela que vai ser usada para geração do gráfico e ordena em ordem crescente de Riqueza
    df = pd.read_csv(f"{dataAddress}{networkAddress}/{table}.csv")
    df = df.sort_values(by = "din")
    if initialState:
        filt = df[df['din'] < 10]
    else:
        filt = df[df['din'] < 1]

    percentualZero = round(filt['quant'].sum()/sum(df['quant']), 2)

    if initialState: 
        df = df[10:]
    else:
        df = df[1::]
    
    

    # Define a riqueza e a probabilidade dos nós
    Wealth = df['din'].values.tolist()
    Probability = df['quant']/pow(10,6)
    Probability = Probability.values.tolist()

    ValMult = max(Probability)
    indexMult = 0

    while ValMult < 1:
        ValMult = ValMult * 10
        indexMult = indexMult + 1

    

    # Se a quisermos adicionar uma gaussiana utilizaremos desse trecho para gerar ela
    if gauss == True:
        bounds = ([0.0, 0.0, 0.0],  # Limites inferiores: amplitude, mu, sigma
          [np.inf, np.inf, np.inf]) # Limites superiores
        xi = np.linspace(min(Wealth), max(Wealth), len(Wealth))
        popt,pcov = curve_fit(gaussCalc, Wealth, Probability, p0=(max(Probability), 0, 1000), bounds=bounds)


    

    # Configuramos o tamanho do gráfico
    fig, axes = plt.subplots(figsize=(14,7))
    axes.scatter(Wealth, Probability , s= 10, label = 'Data', color="black")

    if gauss == True:
        axes.plot(xi,gaussCalc(xi, *popt),  c = "tab:red", label = 'Gaussian Fit')

    # Definimos até q valor o eixo X(Wealth) e o eixo Y(Probability) devem ir

    xAndYSlim(axes, max(Wealth), max(Probability), powYes=False)

    diagramStyle(axes, Wealth,Probability, 19, 19, science=True, IndexMult=indexMult)

    # Nomeamos os eixos
    axisYnameIni = r'$\text{Probability}(x10^{-'
    axisYnameFin = r'})$'
    nominationAxes(fig, axes, 'Wealth', f'{axisYnameIni}{indexMult}{axisYnameFin}')
    axes.legend(fontsize = 15)
    
    #Imprimimos o nome da rede quando estiver pronta, e caso precisarmos também o gaussianFit
    print(" ")
    print(f"{table}{networkName}_{type}")
    print('Porcentagem de 0s: ', percentualZero)
    if gauss == True:
        print("Media: ", round(popt[1], 2))
        print("Desvio: ", round(popt[2],2 ))
        print('Coeficiente de Correlação: ', round(r2_score(Probability, gaussCalc(Wealth, *popt)), 3))

    saveDiagram(whereToSave, networkName, type, "Riq")

## Função allGraphs
# Gera todos os gráficos que precisamos em um só para uma melhor comparação
def allGraph(
        networkAddress : str, whereToSave : str, type : str, table: str, # LOCALIZAÇÃO
        logScale : bool = True,secondGraphs : bool = False, networkAddress2 : str = "nada" # PARÂMETROS
    ) -> None:       
    
    # Definimos o tamanho
    fig, axes = plt.subplots(figsize=(10,7))

    # Qual o tamanho dos ticks
    plt.xticks(size = 13)
    plt.yticks(size = 13)

    networkNames = ["Barabási", "Scale-Free", "Random", "Square Lattice", "Waxman"]
    networksColor = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

    #Geramos os gráficos
    for i in range(5):
        individualGraph(axes, f"{networkAddress}{networks[i]}", networkNames[i], "Money", networksColor[i], '.', 0.5, 35, True)
        if secondGraphs:
                individualGraph(axes, f"{networkAddress2}{networks[i]}", networkNames[i], "Money", networksColor[i], 'h', 1, 50)

    # Definimos o tamanho da legenda
    axes.legend(fontsize=9)

    # Se quisermos o gráfico em escala log x log vamos abrir esse if para ele definir
    if logScale == True:
        axes.set_yscale('log')
        axes.set_xscale('log')

    # Nomeamos os eixos e colocamos um titulo
    
    nominationAxes(fig,axes, 'Wealth', 'Probability')
    axes.set_xlim(5)
    axes.set_ylim([6.8 * pow(10,-9), 2 * pow(10,-4)])
    
    # Salvamos e Mostramos o gráfico
    saveDiagram(whereToSave, 'Todas', type, "Riq")

    print(f"AllGraficos feito!!\n")

## Função lorenzGraph
def lorenzGraph(
        networkAddress: str, whereToSave : str, type : str, table : str, tabelaEscrita : str | None, # LOCALIZAÇÃO
        secondGini: bool = False, networkAddress2 : str = "Nada"
    ) -> None:

    # Definimos o tamanho do gráfico
    fig, axes = plt.subplots(figsize=(10,7))
    giniReturn = []
    # Criamos as imagens de cada Rede
    networkNames = ["Barabási", "Scale-Free", "Random", "Square Lattice", "Waxman"]
    networksColor = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

    for i in range(5):
        equalityLine = lorenzIndividualGraph(axes, f'{networkAddress}{networks[i]}', networkNames[i], table, networksColor[i], giniReturn, labelY_N = True)
        if secondGini:
            lorenzIndividualGraph(axes, f'{networkAddress2}{networks[i]}', networkNames[i], table, networksColor[i], giniReturn, '-.', labelY_N = False)

    axes.plot(equalityLine, equalityLine, color = "tab:blue", linestyle = '--' )
    diagramStyle(axes, equalityLine, equalityLine, all=False)


    # ao mesmo tempo na ultima imagem criamos uma linha perfeita, para gerarmos um linha que vai de 0 até 1

    # Criamos a linha que vai de 0 até 1
    

    # Colocamos a legenda
    axes.legend()
    axes.set_xlim(-0.01,1.01)
    axes.set_ylim(-0.01,1.01)

    giniValues = ' '

    for gini in giniReturn:
        giniValues = giniValues + gini + '\n'

    print(giniValues)

    if tabelaEscrita != None:
        abreArquivoCSV(whereToSave,tabelaEscrita, giniValues, ler_ou_escrever='w')

    # Nomeamos os eixos
    nominationAxes(fig, axes, 'Proportion of population','Proportion of wealth')
    

    # Salvamos e mostramos oq foi feito
    saveDiagram(whereToSave,'Todas',type,"Gini")

#-> FitLig ==========================================================================================================================================================

# Gera um boxplot de gráficos de Ligações ou Fitness x Riqueza
def ligOrFitGraph(
        networkAddress : str, whereToSave: str, networkName : str, type : str, table : str, # LOCALIZAÇÃO
        percentualOfRichestNodes : float = 0.1, initialState : bool = False, ligOrFit : str = 'lig',
        logScale : bool = False# DEFINIÇÃO
    ) -> None:

    ### ----------\Configuramos o gráfico/---------- ###

    if initialState:
        fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, figsize=(21,7))
    else:
        fig, ax1 = plt.subplots(figsize=(21,7))

    
        

    ### ----------\                      /---------- ###

    
    # Conferimos se a tabela existe
    if existingFile(networkAddress, table) == False:
        return None
    
    # Abre a tabela que vai ser usada para geração do gráfico e ordena em ordem decrescente de Riqueza
    df = pd.read_csv(f"{dataAddress}{networkAddress}/{table}.csv")
    df = df.nlargest(int(len(df)*percentualOfRichestNodes), 'din')

    
    

    # Impedimos as vírgulas
    if ligOrFit == 'fit':
        df[ligOrFit] = df[ligOrFit] * 100

    dfDeviation = df.groupby(ligOrFit)['din'].median()

    if initialState:
        df2 = pd.read_csv(f"{dataAddress}{networkAddress}/{table}_bef.csv")
        df2 = df2[df2['id'].isin(df['id'])]
        if ligOrFit == 'fit':
            df2[ligOrFit] = df2[ligOrFit] * 100
        sns.boxplot(x = ligOrFit, y = 'din', data = df2.round(0), fliersize = 3,fill = False, ax=ax2, color= "orange")



    
    xAndYSlim(ax1,max(df[ligOrFit]), max(df['din']), startY=min(df['din']) -500,powYes=False)

    
    positionX = range(len(dfDeviation))

    if ligOrFit == 'fit':
        #try:
        #    positionY, fitness, label = fitexponencial(positionX, dfDeviation)
        #except:
            positionY, fitness, label = fitLinear(positionX, dfDeviation)
            

    if ligOrFit == 'lig':
        try:
            
            positionY, fitness, label = fitPowerLaw(positionX, dfDeviation)
        except:
            
            positionY, fitness, label = fitExponential(positionX, dfDeviation)
    
    print(f"Coeficiente R² da {networkName}, {ligOrFit}: {fitness}")
    
    df['Labels'] = "Data"

    ax1.plot(positionX, positionY, color = 'red', label = f"{label}fit")
    sns.boxplot(x = ligOrFit, y = 'din',hue = "Labels", data = df.round(0), fliersize = 3,fill = False, ax=ax1, palette=["black"])
    


    if initialState:
        plt.subplot(2,1,2)
        diagramStyle(ax2, dfDeviation.index, df2['din'], boxplot=True, withoutValX=True, angle = 45)
        plt.subplot(2,1,1)
        plt.subplots_adjust(hspace=0.35)
    diagramStyle(ax1, dfDeviation.index, df['din'], boxplot=True, angle=45)
    plt.subplots_adjust(bottom=0.2)

    # Nomeamos os eixos
    wealthAxesName = "Wealth"

    if ligOrFit == "lig":
        AxesYName = "Degree"
    else:
        AxesYName = "Fitness"


    if initialState:
        nominationAxes(fig, ax1, AxesYName, wealthAxesName, withoutAxisX=True)
        nominationAxes(fig, ax2, AxesYName, wealthAxesName)
    else:
        nominationAxes(fig, ax1, AxesYName, wealthAxesName)


    if initialState == False:
        plt.legend(fontsize = 15)
    saveDiagram(whereToSave, type, table)




# Gráficos de Fit x Lig
def FitLigGraph(networkAddress:str, whereToSave: str, networkName:str, type:str, tableFit:str, tableLig:str, percentualOfRichestNodes : float = 1) -> None:

    ### ----------\Configuramos o gráfico/---------- ###

    fig, axes = plt.subplots(figsize=(21,7))


    # nomination de Eixos e Titulo
    nominationAxes(fig, axes, 'Degree', 'Fitness')

    # Definimos o tamanho dos tick do eixo X
    labelSize = 3
    axes.tick_params(axis='x', labelsize=labelSize)

    ### ----------\                      /---------- ###

    if existingFile(networkAddress, tableFit) == False:
        return None
    
    if existingFile(networkAddress, tableLig) == False:
        return None
    
    dfFit = pd.read_csv(f"{dataAddress}{networkAddress}/{tableFit}.csv")
    dfLig = pd.read_csv(f"{dataAddress}{networkAddress}/{tableLig}.csv")
    dfFit = dfFit.sort_values(by = ['id', 'din'], ignore_index=True)
    dfLig = dfLig.sort_values(by = ['id', 'din'], ignore_index=True)



    dfFit['fit'] = dfFit['fit'] * 100
    dfFit['fit'] = dfFit['fit'].astype(int)
    dfFit['lig'] = dfLig['lig']

    dfFit = dfFit.nlargest(int(len(dfFit)*percentualOfRichestNodes), 'din')

    dfDeviation = dfLig.groupby('lig')['din'].median()


    
    sns.boxplot(y = 'fit', x = 'lig', data = dfFit, fliersize = 3, fill = False, palette = ["black"], legend="False")
    diagramStyle(axes, dfDeviation.index, dfFit['fit'], boxplot=True, angle=45)
    plt.subplots_adjust(bottom=0.2)

    saveDiagram(whereToSave, networkName, type, 'FL')

#-> Desvios ==========================================================================================================================================================

def deviationGraph(
        networkAddress : str, whereToSave: str, networkName : str, type : str, table : str, # LOCALIZAÇÃO
        percentualOfRichestNodes : float = 0.1, ligOrFit : str = 'lig'# DEFINIÇÃO
        ) -> None:

    if existingFile(networkAddress, table) == False:
        return None
    
    df = pd.read_csv(f"{dataAddress}{networkAddress}/{table}.csv")
    df = df.nlargest(int(len(df)*percentualOfRichestNodes), 'din')

    if ligOrFit == 'fit':
        df[ligOrFit] = df[ligOrFit] * 100

    dfDeviation = round(df.groupby(ligOrFit)['din'].std(),2)
    dfDeviation = dfDeviation.fillna(0)

    Y = dfDeviation.reset_index(drop=True)

    fig, axes = plt.subplots(figsize=(21,7))

    axes.scatter(dfDeviation.index, Y, s= 25, label = 'Data')

    if ligOrFit == 'lig':
        nominationAxes(fig, axes, 'Degree', 'Wealth')
    else:
        nominationAxes(fig, axes, 'Fitness', 'Wealth')

    diagramStyle(axes, dfDeviation.index, Y, boxplot=False)

    saveDiagram(whereToSave, networkName, type, f'D{table}')


#-> By Time ==========================================================================================================================================================

# Gera um boxplot da porcentagem de riqueza dos nós que não são os mais ricos
def byTimeGrafico(
        networkAddress : str, whereToSave: str, networkName : str, type : str, table : str, # LOCALIZAÇÃO
    ) -> None:

    # Configuramos o tamanho do gráfico
    fig, axes = plt.subplots(figsize=(14,7))

    # Abre a tabela que vai ser usada para geração do gráfico
    
    if existingFile(networkAddress, table) == False:
        return None
    
    df = pd.read_csv(f"{dataAddress}{networkAddress}/{table}.csv")

    df['porcentagem'] = round(df['porcentagem']/0.9, 3)

    # Cria um boxplot de tempo x probabilidade
    sns.boxplot(x = 'tempo', y = 'porcentagem', data = df, gap = .3, width=1, fliersize = 3, palette=["black"])



    # Area dos ticks no eixo x
    axes.set_xticks(range(0, 101, 25)) 
    axes.set_yticks(np.arange(0.5,1, 0.1)) 

    # Espaçamentos dos ticks no eixo x
    #axes.set_xticklabels([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) 

    # Nomeamos os eixos e estilizamos
    diagramStyle(axes, df["tempo"], df['porcentagem'], bTime = True)
    nominationAxes(fig, axes, 'Ticks', 'Income concentration')
    
    # Chamamos a função saveDiagram para salvar oq foi feito
    saveDiagram(whereToSave,networkName, f"byTime_{type}", table)

    print(f"Gráfico bTime {networkName} feito!!\n")





