# ANÁLISE DA DISTRIBUIÇÃO DE RIQUEZA EM REDES COMPLEXAS

 > Este trabalho propõe a investigação da concentração de renda em uma população utilizando a modelagem baseada em agentes [1]. Na modelagem baseada em agentes _(Agent Based Model)_, o sistema é modelado como uma coleção de entidades de tomada de decisão chamadas agentes. A modelagem baseada em agentes oferece diversos benefícios em comparação com outras técnicas de modelagem porque permite capturar fenômenos emergentes; fornece uma descrição natural de certos tipos de sistemas; e é flexível permitindo que os agentes apresentem comportamento complexo, incluindo aprendizagem e adaptação.

## Pré-Requisitos 

 - Deve ser usada a versão _3.13_, ou maior, do Python.
 - Para executar a simulação, arquivo .nlogox que se encontra em dataCollector, deve-se baixar a versão mais recente do [Netlogo](https://www.netlogo.org/).
 - O projeto funciona no Linux, Mac e Windows.

## Instalação do Python 

### #1 - Clone o repositório

```bash
git clone git@github.com:KatrielCarvalho/Analise-da-Distribuicao-de-Riqueza-em-Redes-Complexas.git
cd Analise-da-Distribuicao-de-Riqueza-em-Redes-Complexas
```

### #2  - Crie um Virtual Enviroment(venv)

```bash
python3 -m venv venv
```

### #3 - Ative o venv

_**Ativação do venv no Linux/Mac:**_
```bash
source venv/bin/activate
```

_**Ativação do venv no Windows:**_
```bash
venv\Scripts\activate
```

### #4 - Instale as dependências

```bash
pip install -r requirements.txt
```

### #5 - Crie os endereços necessários

Deve-se criar um arquivo chamado **addresses.py** com as seguintes variáveis:
```
universalAddress: str  = # Endereço raíz do PC até a pasta do projeto
dataAddress: str =  # Endereço da pasta onde fica os dados
saveAddress: str = # Endereço onde será salvo os gráficos


destiny : str = # Destino 1 de onde os dados específicos estão
destiny2 : str = # Destino 2 de onde os dados específicos estão (Usado para comparação)
destinyByTime : str = # Destino dos dados salvos por tick
saveDestiny : str = # Destino específico para salvar
type : str = # Nome do tipo de dado que você está tratando (Usado para nomear os gráficos e diferencia-los)

networks : list[str]= ['01.Barabasi', '02.Scale-Free', '03.Aleatória', '04.Quadrada', '05.Waxman'] # Pastas em que se encontram os dados de cada rede
```

## Estrutura do Projeto

```
Analise-da-Distribuicaode-Riqueza-em-Redes-Complexas/
├── dataCollector/              # Pasta onde se localiza o simulador
│   ├── simulator.nlogox        # Simulador 
├── dataHandler/                # Pasta onde se encontra os tratadores dos dados gerados
│   ├── addresses.py            # Arquivo usado para definir os endereços necessários (Deve ser criado)
│   ├── CSVmodifier.py          # Código responsável pela modificação das tabelas CSV
│   ├── diagramGenerator.py     # Código responsável para gerar gráficos
│   └── main.py                 # Local para executaros comandos do CSVmodifier.py e diagramGenerator.py
├── requirements.txt            # Dependências
├── README.md                   # Esse arquivo
└── .gitignore                  # Git ignore
```

## Instituições Apoiadoras
<table>
  <tr>
    <td align="center">
      <a href="https://www.cefetmg.br/" title="cefetmg">
        <img src="IntituicoesLogo/CEFET-MG_completa-negativo.png" width="100px;" alt="Logo do CEFET-MG"/><br>
        <sub>
          <b>CEFET-MG</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://fapemig.br/" title="fapemig">
        <img src="https://api.site.fapemig.br/wp-content/uploads/Trasparente-pequena.png" width="100px;" alt="Logo da FAPEMIG"/><br>
        <sub>
          <b>FAPEMIG</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

## Referências 
[1] B. M. Boghosian. Is Inequality Inevitable? Wealth naturally trickles up in free-market economies, model suggests. Disponível em: https://andyborne.com/math/StorageForFreebies/TinyLibrary/Articles/Is_Inequality_Inevitable.pdf. Acesso em: 03/02/2026.
