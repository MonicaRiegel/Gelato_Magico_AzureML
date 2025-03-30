# %% 

# %% 
#  Importar bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn

# %% 
# Configurações iniciais
print("Iniciando notebook para Azure Machine Learning...")

# %% 
# Etapa 1: Carregar os dados
print("Carregando os dados...")
# Substitua "vendas_sorvetes.csv" pelo caminho correto do arquivo no Azure ML
df = pd.read_csv("../Coletando_dados/dados/vendas_sorvetes.csv")

# %% 
# Preparar os dados
X = df[["Temperatura (°C)"]]
y = df["Vendas (Unidades)"]

# %% 
# Dividir os dados em treino e teste
print("Dividindo os dados...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% 
# Etapa 2: Treinar o modelo
print("Treinando o modelo...")
model = LinearRegression()
model.fit(X_train, y_train)

# %% 
# Fazer previsões
print("Realizando previsões...")
y_pred = model.predict(X_test)

# %% 
# Avaliar o modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Absoluto Médio (MAE): {mae}")
print(f"Erro Quadrático Médio (MSE): {mse}")

# %% 
# Etapa 3: Integrar com MLflow para registro do modelo
print("Registrando modelo no MLflow...")
mlflow.set_experiment("Previsão de Vendas de Sorvete")
with mlflow.start_run():
    mlflow.log_param("modelo", "Regressão Linear")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "modelo_regressao_linear")

# Mensagem final
print("Notebook concluído com sucesso!")
