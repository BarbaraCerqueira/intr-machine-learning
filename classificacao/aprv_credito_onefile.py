import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv('classificacao\conjunto_de_treinamento.csv')
test = pd.read_csv('classificacao\conjunto_de_teste.csv')

# Salvar id_solicitante para auxiliar na criação do arquivo que será enviado ao Kaggle
id_soliciante = test['id_solicitante']

# Criando um DF novo com o train e o test
credito_df = pd.concat([train, test], ignore_index=True)

# Para auxiliar no momento separar o DF credito_df em train e test
split_index = len(train)


#### Data Preparation ###

credito_df['tipo_residencia'] = credito_df['tipo_residencia'].fillna(1)

avg_months_res = credito_df.groupby('tipo_residencia')['meses_na_residencia'].mean()
credito_df['meses_na_residencia'] = credito_df['meses_na_residencia'].fillna(credito_df['tipo_residencia'].map(avg_months_res))

credito_df['ocupacao'] = credito_df['ocupacao'].fillna(2)

credito_df = credito_df.drop(columns='profissao')

credito_df = credito_df.drop(columns=['profissao_companheiro', 'grau_instrucao_companheiro'])

credito_df = credito_df.drop(columns=['estado_onde_trabalha'])

credito_df.loc[credito_df['sexo'] == ' ', 'sexo'] = 'N'
credito_df.loc[credito_df['estado_onde_nasceu'] == ' ', 'estado_onde_nasceu'] = 'XX'


#### Feature Engineering ###

state_to_region = {
    'AC': 'Norte',
    'AL': 'Nordeste',
    'AP': 'Norte',
    'AM': 'Norte',
    'BA': 'Nordeste',
    'CE': 'Nordeste',
    'DF': 'Centro-Oeste',
    'ES': 'Sudeste',
    'GO': 'Centro-Oeste',
    'MA': 'Nordeste',
    'MT': 'Centro-Oeste',
    'MS': 'Centro-Oeste',
    'MG': 'Sudeste',
    'PA': 'Norte',
    'PB': 'Nordeste',
    'PR': 'Sul',
    'PE': 'Nordeste',
    'PI': 'Nordeste',
    'RJ': 'Sudeste',
    'RN': 'Nordeste',
    'RS': 'Sul',
    'RO': 'Norte',
    'RR': 'Norte',
    'SC': 'Sul',
    'SP': 'Sudeste',
    'SE': 'Nordeste',
    'TO': 'Norte',
    'XX': 'Unknown'
}

credito_df['estado_onde_nasceu'] = credito_df['estado_onde_nasceu'].map(state_to_region)
credito_df['estado_onde_reside'] = credito_df['estado_onde_reside'].map(state_to_region)
credito_df = credito_df.drop(columns=['local_onde_reside', 'local_onde_trabalha'])


### Categorical variables codification ###

categorical_variables = ['produto_solicitado', 'dia_vencimento', 'forma_envio_solicitacao', 'tipo_endereco', 'sexo', 'estado_civil',
                        'grau_instrucao', 'nacionalidade', 'estado_onde_nasceu', 'estado_onde_reside', 
                        'possui_telefone_residencial', 'codigo_area_telefone_residencial', 'tipo_residencia', 'possui_telefone_celular',
                        'possui_email', 'possui_cartao_visa', 'possui_cartao_mastercard', 'possui_cartao_diners', 'possui_cartao_amex', 
                        'possui_outros_cartoes', 'possui_carro', 'vinculo_formal_com_empresa', 'possui_telefone_trabalho', 
                        'codigo_area_telefone_trabalho', 'ocupacao']

# drop_first tira a primeira coluna, de forma que se as outras são zero, ela é a 'selecionada'
for variable in categorical_variables:
    credito_df = pd.get_dummies(credito_df, columns=[variable], prefix=[variable], drop_first = True)


### Applying Scaler ###

normalize_data = ['idade', 'qtde_dependentes', 'meses_na_residencia', 'renda_mensal_regular', 'renda_extra', 
                'qtde_contas_bancarias', 'qtde_contas_bancarias_especiais', 'valor_patrimonio_pessoal', 
                'meses_no_trabalho']
scaler = MinMaxScaler() 

# Aplicar a normalização aos dados
credito_df[normalize_data] = scaler.fit_transform(credito_df[normalize_data])


### Modelling ###

credito_df = credito_df.drop(columns='id_solicitante')
train = credito_df[:split_index].copy()
test = credito_df[split_index:].copy()
train['inadimplente'] = train['inadimplente'].astype(int)

X_train = train.drop('inadimplente', axis = 1)
y_train = train['inadimplente']
X_test = test.drop('inadimplente', axis = 1)

# gradient booster params
best_params = {'learning_rate': 0.1, 
               'max_depth': 3, 
               'min_samples_split': 10,
               'min_samples_leaf': 1,
               'n_estimators': 200, 
               'subsample': 0.8}

# Treinando o modelo com os melhores hiperparâmetros no conjunto de treinamento completo
best_gb = GradientBoostingClassifier(**best_params)
best_gb.fit(X_train, y_train)

# Fazer previsões no conjunto de testes
best_gb_pred = best_gb.predict(X_test)


### Generating output file ###

kaggle = pd.DataFrame({'id_solicitante': id_soliciante, 'inadimplente': best_gb_pred})
kaggle.to_csv('AprvCredito.csv', index = False)
print(kaggle)