"""####---------------------------------------------------------<           SETUPS          >---------------------------------------------------------####""" 
#-> Minhas
from addresses import *

#-> Para o csv
import pandas as pd

#-> Funções necessárias
# Confere se um arquivo existe
def existingFile(networkAddress : str, table : str):
    try:
        df = pd.read_csv(f"{dataAddress}{networkAddress}/{table}.csv")
        return True
    except FileNotFoundError:
        print(f'Arquivo não encontrado: {dataAddress}{networkAddress}/{table}.csv')
        return False
    
# Abre o arquivo CSV e decide o que fazer com a informação do arquivo
def openCSVFile(
    networkAddress : str, tabela : str, #INDENTIFICADOR DA TABELA
    lines : list[str] | None = None, #LINHAS DA TABELA
    delete : bool = False, writeOrRead : str = 'r' #CONFERIDORES
) -> list[str] | None:
    
    # Abrimos o arquivo csv seja para leitura 'r' ou escrita 'w'
    with open(f"{dataAddress}{networkAddress}/{tabela}.csv", writeOrRead) as CSVFile:

        # Caso seja para leitura retornamos as linhas
        if writeOrRead == 'r':
            return CSVFile.readlines()
        
        
        # Caso escrita abrimos a funcao que transcreve por cima do arquivo anterior
        if writeOrRead == 'w':
            escritaDoCSV(networkAddress, tabela,lines, CSVFile, delete)   

# Vai transcrever o que estava no arquivo antigo
def escritaDoCSV(
    networkAddress : str, tabela : str, #INDENTIFICADOR DA TABELA
    lines : list[str], #LINHAS DA TABELA
    CSVFile, #ACESSO AO ARQUIVO
    delete: bool = False, #CONFERIDORES
    ) -> None:

    # Caso ele seja usado para deletar ele irá chamar a função deleteLines
    if delete:
        deleteLines(lines, CSVFile)
        return None
    
    # Transcreve o que foi passado para ele transcrever, ou seja, linhas
    CSVFile.write(lines)

# Deleta as linhas em excesso
def deleteLines(lines : list[str], CSVFile):
    CSVFile.write('din,quant,color,pen down?\n')
    for i, line in enumerate(lines):
        # Parametro pra reescrever todas as linhas sem as 16 primeiras linhas
        if i not in [0,1, 2,3 ,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
            CSVFile.write(line)
    
"""####---------------------------------------------------------<         MANIPULAÇÃO        >---------------------------------------------------------####""" 

#-> Limpeza   
# Limpa coisas que são excesso
def clean(networkAddress : str, table : str, tableNumber : int, increaser: int = 1) -> None:

    
    
    for i in range(0, tableNumber, increaser):

        # Serve para acessar a tabela de número i
        tempTable = table + f'{i}'

        # Criamos uma lista chamada linhas q vai armazernar todas as linhas do arquivo antigo
        lines : list[str] = []

        # Le o Arquivo
        # E salvamos todas as linhas do arquivo antigo na lista criada
        lines = openCSVFile(networkAddress, tempTable)

        if lines[0] == 'din,quant,color,pen down?\n':
            print('Arquivo já está limpo')
            return None

        # Escreve o Arquivo
        # Apagamos as linhas q n precisamos
        openCSVFile(networkAddress, tempTable, lines, True, 'w')

    print('Limpeza realizada com sucesso')

#-> Mesclagem
# Cria uma tabela principal com o dado de todas as outras
def mixTables(networkAddress : str, table1 : str, table2 : str, tableNumber : int, ligFitOrNo : str = 'quant') -> None:
    if existingFile(networkAddress, table1) == False:
        return None
    
    if len(pd.read_csv(f'{dataAddress}{networkAddress}/{table1}.csv') > 1):
        print("Arquivo já mesclado!!")
        return None
    else:
        print("Mesclando Arquivo:")
    

    # J para repetir o código 100 vezes(por ser 100 amostras)
    for j in range(tableNumber):
        
        # Pegamos onde vai armazenar as 100 amostras e salvamos oq já está lá dentro em df
        df = pd.read_csv(f'{dataAddress}{networkAddress}/{table1}.csv')

        # Pegamos a amostra j e salvamos oq está lá dentro em df2
        df2 = pd.read_csv(f'{dataAddress}{networkAddress}/{table2}{j}.csv')
        df2 = round(df2, 2)
        
        # Transformamos df e df2 em int

        # Quais serão as variaveis a ser definidas
        Wealth = [];Wealth2 = []
        Degree = [];Degree2 = []
        Ids = [];Ids2 = []
        nodeQuantity = [];nodeQuantity2 = []

        # Definimos se vamos armazenar Riqueza x Quantidade(se lig == False) de Nós ou em Riqueza x Ligações(se lig == True)
        definition(df, df2, j, Wealth, Wealth2, Degree, Degree2, Ids, Ids2, nodeQuantity, nodeQuantity2, ligFitOrNo)

        # message vai ser o cabeçalho
        # A função addValues além de retornar um cabeçalho para message também vai adicionar as valores da tabela2 na tabela1
        message = addValues(df, ligFitOrNo, Wealth, Wealth2, Degree, Degree2, Ids, Ids2, nodeQuantity, nodeQuantity2, j)


        # Vai definir message como a message completa a ser transcrita na tabela1
        for i in range(len(Wealth)):
            message += (fillMessage(i, ligFitOrNo, Wealth, Degree, Ids, nodeQuantity))
        
        # Vai trancrever a message na tabela 1
        openCSVFile(networkAddress, table1, message, writeOrRead='w')
    

    print(table1, ' mesclado com sucesso')

# Cria uma tabela principal que se adequa aos padrões da regra byTime
def byTimeCSV(networkAddress: str, table: str, tableNumber : int, increaser :int = 1) -> None:

    # Criamos as litas Percentage e Time
    Percentage = []
    Time = []
    
    # Executamos todas as tabelas e seus tempos
    for i in range (0, tableNumber, increaser):
        for j in range(0, 100):
            mostRich(networkAddress, f"Money§{j}_{i}", i, Percentage , Time)

    # Executamos todos os tempos que foram o último
    for i in range(100):
        mostRich(networkAddress, f"Money§{i}", tableNumber, Percentage, Time)

    # Processo de transcrição na tabela
    message = "porcentagem,tempo\n"
    for i in range(len(Time)):
                message += (fillMessage(i, False, Percentage, Time, Time, Time))

    openCSVFile(networkAddress, table, message, writeOrRead='w')

#-> Analise
# Serve para pegar os 10% dos nós mais ricos, informação utilizada no byTimeCSV
def mostRich(networkAddress : str, table : str, time : int, Percentage, Time) -> None:

    # Variavel soma para somarmos os valores que pegamos
    wealth = 0 

    # Abrimos a tabela desejada para fazer a modificação
    if existingFile(networkAddress, table) == False:
        return None
    dft = pd.read_csv(f"{dataAddress}{networkAddress}/{table}.csv")
    dft = dft.sort_values(by = "din", ascending=False)

    # Definimos a quantidade de Nos e a quantia de wealth
    Quantity = dft["quant"].to_list()
    Wealth = dft["din"].to_list()

    # Criamos um counter e a quantidade para pegarmos os valores atuais
    quantity = 0
    counter = 0
    
    # Definimos quantos Nós vamos pegar
    mostRichPercentage = sum(Quantity) * 0.1

    # Enquanto quantity for menor que os 10% adicionamos wealth e quantity e aumentamos o counter em um
    while(quantity < mostRichPercentage):
        wealth+= Wealth[counter]*Quantity[counter]
        quantity += Quantity[counter]
        counter += 1
        
    # Caso quanity ultrapasse arrumamos para o valor se adequar
    if quantity > mostRichPercentage:
        wealth-= Wealth[counter-1] * (quantity - mostRichPercentage)

    # Transformamos em uma probabilidade de qual a riqueza dos nós que não são os 10% mais ricos
    Percentage.append(1 - (wealth/pow(10, 7)))

    # Também pegamos o tempo em que aquilo foi executado
    Time.append(time)

#-> Funções necessárias para a mesclaTabela

# Serve para definirmos as variabeis em mesclaTabela
def definition(df, df2, j, Wealth, Wealth2, Degree, Degree2, Ids, Ids2, nodeQuantity, nodeQuantity2, ligFitOrNo: str) -> None:
    if ligFitOrNo == 'fit' or ligFitOrNo == 'lig':
        # Os que não possuem número são os do table1
        Wealth.extend(df['din'].values.tolist())
        Degree.extend(df[ligFitOrNo].values.tolist())
        Ids.extend(df['id'].values.tolist())

        # Os que possuem número são os do table2, vale ressaltar que Ids2 armazena a qual amostra pertence aquele id
        Wealth2.extend(df2['din'].values.tolist())   
        Degree2.extend(df2[ligFitOrNo])
        
        Ids2.extend(df2['idt'].values.tolist())
        Ids2 = [(Idt * 100) + j for Idt in Ids2]

        return None

    # Os que não possuem número são os do table1
    Wealth.extend(df['din'].values.tolist())
    nodeQuantity.extend(df['quant'].values.tolist())

    

    # Os que possuem número são os do table2
    Wealth2.extend(df2['din'].values.tolist())    
    nodeQuantity2.extend(df2['quant'].values.tolist())
## Função fillMessage 
# Serve para adicionarmos o valor a tabela que conterá todas amostras
def fillMessage(ind : int, ligFitOrNo : str, Wealth, Degree, Ids, nodeQuantity) -> str:
        
    # Se ligFitOrNo == true -> Riqueza x Ligações
    # Se ligFitOrNo == false -> Quantidade x Riqueza
    if ligFitOrNo == 'fit' or ligFitOrNo == 'lig':
        return f'{Wealth[ind]},{Degree[ind]},{Ids[ind]}\n'
    
    return f'{Wealth[ind]},{nodeQuantity[ind]}\n'
    
## Função definition


## Função addValues
# Vai adicionar o valor
def addValues(df, ligFitOrNo: str, Wealth, Wealth2, Degree, Degree2, Ids, Ids2, nodeQuantity, nodeQuantity2, j)-> str:

    # Confere se entrará no Riqueza x Ligação ou não
    if ligFitOrNo == 'lig' or ligFitOrNo == 'fit':
        return addToDegreeFitness(ligFitOrNo, Wealth, Wealth2, Degree, Degree2, Ids, Ids2, j)
    
    return addToQuantity(df, Wealth, Wealth2, nodeQuantity, nodeQuantity2)

# Adiciona os valores com base na regra da tabela de ligações x riqueza
def addToDegreeFitness(ligOrFit : str, Wealth, Wealth2, Degree, Degree2, Ids, Ids2, j):
    # Apenas adiciona os novos valores aos valores antigos
    # Já que não tem possibilidade de esse valor já ter sido adicionado
    Wealth.extend(Wealth2)
    Degree.extend(Degree2)
    Ids2 = [element + ((j+1)*100000) for element in Ids2]
    Ids.extend(Ids2)

    # Isso serve para reescrever o inicio e assim não ter problemas quando o código passar para próxima amostra
    return f'din,{ligOrFit},id\n'

# Adiciona os valores com base na regra da tabela de riqueza  x probabilidade
def addToQuantity(df, Wealth, Wealth2, nodeQuantity, nodeQuantity2):
    # Se ele não entrar ele fará o seguinte:
    # Vai pegar o valor e a numeração de cada valor da tabela2
    for possibleExistentValueIndex, possibleExistentValue in enumerate(Wealth2):
        count = 0

        # temp confere se existe algum valor da tabela1 com aquele valor
        temp = df[(df['din'] == possibleExistentValue)]

        # Se o tamanho da quantidade de vértices com a riqueza daquele respectivo valor foi diferente de 0 ele entra
        if(nodeQuantity2[possibleExistentValueIndex] != 0):
            # Confere se existem valores iguais
            equalValue(temp, count, possibleExistentValue, possibleExistentValueIndex, Wealth, nodeQuantity, nodeQuantity2)

    # Isso serve para reescrever o inicio e assim não ter problemas quando o código passar para próxima amostra
    return 'din,quant\n'

# Busca por valores iguais dada duas tabelas
def equalValue(temp, count, possibleExistentValue, possibleExistentValueIndex, Wealth, nodeQuantity, nodeQuantity2):
    if temp.index.size > 0:
        # Se existe ele vai chegar até a posição aonde fica aquele valor na tabela1
        while Wealth[count] != possibleExistentValue:
            count = count + 1
        
        # E soma a quantia que possui esse determinado valor da tabela1 com o da tabela2

        nodeQuantity[count] = nodeQuantity[count] + nodeQuantity2[possibleExistentValueIndex]

    else:
        # Se não existe só adiciona aquele valor a lista
        Wealth.append(possibleExistentValue)

        # e a quantia que possui ele
        nodeQuantity.append(nodeQuantity2[possibleExistentValueIndex])


