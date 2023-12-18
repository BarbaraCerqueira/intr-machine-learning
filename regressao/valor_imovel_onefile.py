import numpy as np
import pandas as pd
import scipy.stats

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from sklearn.svm import SVR
from sklearn.metrics import make_scorer


train = pd.read_csv('regressao/conjunto_de_treinamento.csv')
test = pd.read_csv('regressao/conjunto_de_teste.csv')

# Salvar os Ids para auxiliar na criação do arquivo que será enviado ao Kaggle
id = test['Id']

# Criando um DF novo com o train e o test
imovel_df = pd.concat([train, test], ignore_index=True)

# Para auxiliar no momento separar o DF credito_df em train e test
split_index = len(train)


# Criando os intervalos e rótulos para as novas categorias de quartos
intervalos = [0, 3, 6, 9]
rotulos = ['0_to_3', '4_to_6', '7_to_9']
imovel_df['quartos'] = pd.cut(imovel_df['quartos'], bins=intervalos, labels=rotulos, include_lowest=True)

# Criando os intervalos e rótulos para as novas categorias de suites
intervalos = [0, 1, 4, 6]
rotulos = ['0_to_1', '2_to_4', '5_to_6']
imovel_df['suites'] = pd.cut(imovel_df['suites'], bins=intervalos, labels=rotulos, include_lowest=True)

# Criando os intervalos e rótulos para as novas categorias de vagas
intervalos = [0, 1, 3, 6, 8, 30]
rotulos = ['0_to_1', '2_to_3', '4_to_6', '7_to_8', 'more_than_9']
imovel_df['vagas'] = pd.cut(imovel_df['vagas'], bins=intervalos, labels=rotulos, include_lowest=True)

imovel_df = imovel_df.drop(columns='diferenciais')

imovel_df['area_util'] = np.log1p(imovel_df['area_util'])  # log1p = log(x + 1); pra evitar algum log(0), que e indefinido
imovel_df['area_extra'] = np.log1p(imovel_df['area_extra'])  

categorical_variables = ['tipo', 'bairro', 'tipo_vendedor', 'quartos', 'suites', 'vagas',
       'churrasqueira', 'estacionamento', 'piscina', 'playground', 'quadra', 's_festas', 
       's_jogos', 's_ginastica', 'sauna', 'vista_mar']

# drop_first tira a primeira coluna, de forma que se as outras são zero, ela é a 'selecionada'
for variable in categorical_variables:
    imovel_df = pd.get_dummies(imovel_df, columns=[variable], prefix=[variable], drop_first = True)

normalize_data = ['area_util', 'area_extra']
scaler = StandardScaler() 

# Aplicar a normalização aos dados
imovel_df[normalize_data] = scaler.fit_transform(imovel_df[normalize_data])

imovel_df = imovel_df.drop(columns='Id')

train = imovel_df[:split_index].copy()
test = imovel_df[split_index:].copy()

X_train = train.drop('preco', axis = 1)
y_train = train['preco']
X_test = test.drop('preco', axis = 1)

# log transformation
y_train_log = np.log(y_train)

# Função para calcular RMSPE
def rmspe(y_true, y_pred):
    pct_error = (y_true - y_pred) / y_true
    pct_error[pct_error == np.inf] = 0  # Trata casos em que y_true é zero
    rmspe_score = np.sqrt(np.mean(pct_error**2))
    return rmspe_score

# Cria um scorer personalizado usando a função rmspe
rmspe_scorer = make_scorer(rmspe, greater_is_better=False)

best_params = {'C': 1, 'epsilon': 0.1, 'gamma': 0.1}

# Treinando o modelo com os melhores hiperparâmetros no conjunto de treinamento completo
best_hr = SVR(**best_params)
best_hr.fit(X_train, y_train_log)

# Fazer previsões no conjunto de testes
best_hr_pred = best_hr.predict(X_test)
best_hr_pred = np.exp(best_hr_pred)

kaggle = pd.DataFrame({'Id': id, 'preco': best_hr_pred})
kaggle.to_csv('ValorImovel.csv', index = False)
print(kaggle)