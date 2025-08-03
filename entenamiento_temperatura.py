import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import joblib

# 1. Leer el CSV
ruta = "temperatura_mensual_papallacta_2025.csv"
df = pd.read_csv(ruta)

# 2. Limpiar datos nulos (-999.0)
df = df.replace(-999.0, np.nan)
df = df.dropna()

# 3. Crear índice de fechas correcto
fechas = pd.to_datetime(df[['YEAR', 'Month']].assign(DAY=1))  # Primer día de cada mes
temperaturas = df['Temperature_C'].astype(float)
serie = pd.Series(temperaturas.values, index=fechas)

# 4. División en entrenamiento y prueba (85%)
train_size = int(len(serie) * 0.85)
train, test = serie[:train_size], serie[train_size:]

# 5. Entrenar modelo SARIMA (puedes ajustar los parámetros si deseas)
modelo = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
resultado = modelo.fit(disp=False)

# 6. Predicción sobre el conjunto de prueba
pred = resultado.predict(start=test.index[0], end=test.index[-1])
rmse = np.sqrt(mean_squared_error(test, pred))
print(f"✅ RMSE: {rmse:.2f}")

# 7. Gráfica de resultados
plt.figure(figsize=(12, 5))
plt.plot(train, label='Entrenamiento')
plt.plot(test, label='Prueba', color='orange')
plt.plot(pred, label='Predicción', linestyle='--', color='red')
plt.title("Predicción de Temperatura Mensual (SARIMA)")
plt.xlabel("Fecha")
plt.ylabel("°C")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("prediccion_temperatura_sarima.png")
plt.show()

# 8. Predecir los siguientes 12 meses
forecast = resultado.get_forecast(steps=12)
forecast_index = pd.date_range(start=serie.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')
forecast_values = forecast.predicted_mean

# 9. Guardar modelo SARIMA
joblib.dump(resultado, "modelo_sarima_temperatura.pkl")
print("✅ Modelo guardado como 'modelo_sarima_temperatura.pkl'")

