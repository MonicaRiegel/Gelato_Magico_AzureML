import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carregar dados
df = pd.read_csv("vendas_sorvetes.csv")
X = df[["Temperatura (°C)"]]
y = df["Vendas (Unidades)"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
print("Erro Absoluto Médio:", mean_absolute_error(y_test, y_pred))
print("Erro Quadrático Médio:", mean_squared_error(y_test, y_pred))

# Salvar o modelo usando MLflow
import mlflow
mlflow.set_experiment("Previsão de Vendas de Sorvete")
with mlflow.start_run():
    mlflow.log_param("modelo", "Regressão Linear")
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.sklearn.log_model(model, "modelo_regressao_linear")