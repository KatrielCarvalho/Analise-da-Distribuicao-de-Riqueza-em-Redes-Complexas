"""####---------------------------------------------------------<           SETUPS          >---------------------------------------------------------####""" 

#--> Bibliotecas
#-> Minhas
from CSVmodifier import *
from diagramGenerator import *


#--> Variaveis

#--> Tabelas
# Utilizado para fazer a limpezza das tabelas do tick final
def cleaner():
    for network in networks:
        clean(f'{destinyByTime}{network}', 'Money§', 100)

# Utilizado para fazer a limpeza das tabelas de ticks especificos + o tick final
def cleanerOfAll():
    cleaner()
    for network in networks:
        for i in range(100):
                clean(f'{destinyByTime}{network}', f'Money§{i}_', 1000, 10)

# Junta todas as tabelas que foram geradas em uma só, para facilitar o acesso das informações
def mixer(initialState: bool):
    for network in networks:

        mixTables(f'{destiny}{network}','Fit','Fit§', 100, 'fit')
        mixTables(f'{destiny}{network}','Money','Money§', 100)

        if network == networks[0] or network == networks[1]:
            mixTables(f'{destiny}{network}','Lig','Lig§', 100, 'lig')

        if initialState:
            mixTables(f'{destiny}{network}','Fit_bef','Fitness_bef', 100, 'fit')
            if network == networks[0] or network == networks[1]:
                mixTables(f'{destiny}{network}','Lig_bef','Lig_bef', 100, 'lig')

# Adequa a tabela para funcionar no padrão byTime
def byTimeSetup():
    for network in networks:
        byTimeCSV(f'{destinyByTime}{network}', 'Money', 1000, 10)
        print(f'{network} FINALIZADO')


"""####---------------------------------------------------------<    GERADORES DE GRÁFICO    >---------------------------------------------------------####""" 

# Gera gráficos de Riq x Prob
def wealthGraphGenerator(zoomOrNo : bool = False, initialState:bool = False) -> None:

    for network in networks:
            
            if network == networks[0]:
                wealthGraph(f'{destiny}{network}', saveDestiny, 'Barabási-Albert', type, 'Money', zoomOrNo= zoomOrNo, initialState=initialState)
                continue
            
                
            wealthGraph(f'{destiny}{network}', saveDestiny, network[2::], type, 'Money', zoomOrNo= zoomOrNo, initialState=initialState)

# Gera Gráficos de Fit x Riq
def fitGraphGenerator(percentage : float, initialState:bool) -> None:
    for network in networks:
        if network == networks[0]:
            ligOrFitGraph(f'{destiny}{network}', saveDestiny,'Barabási-Albert', type, 'Fit', percentage, ligOrFit='fit', logScale=False, initialState=initialState)
            continue
        ligOrFitGraph(f'{destiny}{network}', saveDestiny, network[2::], type, 'Fit', percentage, ligOrFit='fit', logScale=False, initialState=initialState)

# Gera Gráficos de Lig x Riq
def ligGraphGenerator(percentage : float, initialState:bool) -> None:

    ligOrFitGraph(f'{destiny}{networks[0]}', saveDestiny,'Barabási-Albert', type, 'Lig', percentage, initialState=initialState)

    ligOrFitGraph(f'{destiny}{networks[1]}', saveDestiny,'Scale-Free', type, 'Lig', percentage, initialState=initialState)

# Gera Gráficos de Fit x Lig
def fitLigGraphGenerator() -> None:
    FitLigGraph(f'{destiny}{networks[0]}', saveDestiny, 'Barabási-Albert', type, 'Fit', 'Lig', 0.1)
    FitLigGraph(f'{destiny}{networks[1]}', saveDestiny, 'Scale-Free', type, 'Fit', 'Lig', 0.1)

def deviationFitGraphGenerator() -> None:
    for network in networks:
        if network == networks[0]:
            deviationGraph(f'{destiny}{network}', saveDestiny,'Barabási-Albert', type, 'Fit', 0.1, ligOrFit='fit')
            continue
        deviationGraph(f'{destiny}{network}', saveDestiny, network[2::], type, 'Fit', 0.1, ligOrFit='fit')

def deviationLigGraphGenerator() -> None:
    deviationGraph(f'{destiny}{networks[0]}', saveDestiny,'Barabási-Albert', type, 'Lig', 0.1, ligOrFit='lig')
    deviationGraph(f'{destiny}{networks[1]}', saveDestiny, 'Scale-Free', type, 'Lig', 0.1, ligOrFit='lig')

#Gera Gráficos de Tick x 10% Mais ricos
def byTimeGraficos():
    for network in networks:
        if network == networks[0]:
                byTimeGrafico(f"{destinyByTime}{network}", saveDestiny, 'Barabási-Albert', type, 'Money')
                continue
        byTimeGrafico(f"{destinyByTime}{network}", saveDestiny, network[2::], type, 'Money')


    
if __name__ == '__main__':
    
    """####-------------------<          SETUPS            >-------------------####""" 

    #cleaner()
    #cleanerOfAll()
    #mixer(True)
    #byTimeSetup()
    
    """####-------------------<    GERADORES DE GRÁFICO    >-------------------####""" 
    #--> Gráficos Básicos
    #wealthGraphGenerator(initialState=False)
    #fitGraphGenerator(0.1, False) 
    #ligGraphGenerator(0.1, False)
    #fitLigGraphGenerator()
    #allGraph(destiny,saveDestiny, type, 'Money', secondGraphs=False, networkAddress2=destiny2)

    #--> Com parametro especfico
    #byTimeGraficos()
    #lorenzGraph(destiny,saveDestiny,f'Gini{type}', 'Money', None, False, destiny2)

    #--> Desvios
    #deviationFitGraphGenerator()
    #deviationLigGraphGenerator()

    pass
    