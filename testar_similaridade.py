import pandas as pd

# Carregue os dois arquivos csv
file1 = pd.read_csv('AprvCredito.csv')
file2 = pd.read_csv('AprvCredito1.csv')

# Compare se os dois dataframes são iguais
if file1.equals(file2):
    print("Os dois arquivos CSV são iguais")
else:
    print("Os dois arquivos CSV não são iguais")