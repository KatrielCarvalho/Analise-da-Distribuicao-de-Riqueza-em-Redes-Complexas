# ANÁLISE DA DISTRIBUIÇÃO DE RIQUEZA EM REDES COMPLEXAS

## Pré-Requisitos

 - Deve ser usada a versão 3.13, ou maior, do Python.
 - Para executar a simulação, arquivo .nlogox que se encontra em dataCollector, deve-se baixar a versão mais recente do [Netlogo](https://www.netlogo.org/).
 - O projeto funciona no Linux, Mac e Windows.

## Python Setup 

### 1) Clone o repositório

```bash
git clone git@github.com:KatrielCarvalho/Analise-da-Distribuicao-de-Riqueza-em-Redes-Complexas.git
cd Analise-da-Distribuicao-de-Riqueza-em-Redes-Complexas
```

### 2) Crie um Virtual Enviroment

```bash
python3 -m venv venv
```

### 3) Ative o Venv

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 4) Instale as dependências

```bash
pip install -r requirements.txt
```

### 5) Endereços

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