import random
import pandas as pd
from datetime import datetime, timedelta

# Configurar valores iniciais
start_date = datetime(2025, 1, 1)
rows = 100

# Gerar dados mais realistas com variação sazonal
data = []
for i in range(rows):
    date = start_date + timedelta(days=i)
    month = date.month

    # Definir intervalos de temperatura conforme as estações do ano no RS
    if month in [6, 7, 8]:  # Inverno (junho, julho, agosto)
        temperature = random.randint(5, 18)  # Mínimo 5°C, máximo 18°C
    elif month in [12, 1, 2]:  # Verão (dezembro, janeiro, fevereiro)
        temperature = random.randint(25, 40)  # Verão quente
    elif month in [3, 4, 5]:  # Outono (março, abril, maio)
        temperature = random.randint(15, 25)
    else:  # Primavera (setembro, outubro, novembro)
        temperature = random.randint(18, 30)

    # Vendas baseadas na temperatura
    sales = int(random.gauss(temperature * 10, 15))  # Média proporcional à temperatura
    sales = max(50, min(sales, 300))  # Limitar vendas entre 50 e 300 unidades
    data.append([date.strftime("%d/%m/%Y"), sales, temperature])

# Criar DataFrame com pandas
df = pd.DataFrame(data, columns=["Data", "Vendas (Unidades)", "Temperatura (°C)"])

# Exibir a tabela e salvar em csv
print(df)
df.to_csv("vendas_sorvetes.csv", index=False)