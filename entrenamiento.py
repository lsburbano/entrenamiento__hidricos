import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# 1. Cargar datos
ruta = "caudal_H34_H36_logico_corregido.xlsx"
df = pd.read_excel(ruta, parse_dates=["Fecha"], index_col="Fecha")

df["caudal_total"] = df["valor_H34"] + df["valor_H36"] + df["valor_H13"]
print("Primeros valores del caudal total:")
print(df["caudal_total"].head())


# 2. Descomposición de la serie temporal
result = seasonal_decompose(df["caudal_total"], model='additive', period=12)
result.plot()
plt.savefig("descomposicion_series.png")

# 3. Verificar estacionariedad
adf_test = adfuller(df["caudal_total"])
print("ADF Statistic:", adf_test[0])
print("p-value:", adf_test[1])

# 4. Transformación logarítmica y diferenciación
serie_log = np.log(df["caudal_total"].clip(lower=1))  # Evita negativos o ceros
serie_diff = serie_log.diff().dropna()

# 5. División en entrenamiento y prueba
split = int(len(serie_diff) * 0.5)
y_train, y_test = serie_diff.iloc[:split], serie_diff.iloc[split:]

# 6. Entrenamiento modelo SARIMAX con hiperparámetros ajustados
model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)
joblib.dump(model_fit, "modelo_caudal.pkl")

# 7. Predicciones sobre test
pred = model_fit.predict(start=y_test.index[0], end=y_test.index[-1], dynamic=False)

# 8. Evaluación del modelo
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
mape = mean_absolute_percentage_error(y_test, pred) * 100
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

# 9. Gráfica entrenamiento vs prueba
plt.figure(figsize=(12, 6))
plt.plot(y_train, label="Entrenamiento")
plt.plot(y_test, label="Prueba", color="orange")
plt.plot(pred, label="Predicción", color="red", linestyle="--")
plt.title("SARIMAX Mejorado - Entrenamiento, Prueba y Predicción")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("evaluacion_mejorada.png")

# 10. Predicción futura
forecast_diff = model_fit.forecast(steps=12)
forecast_log = serie_log.iloc[-1] + forecast_diff.cumsum()
forecast = np.exp(forecast_log)
forecast_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='MS')

# 11. Gráfica de predicción futura
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["caudal_total"], label="Histórico")
plt.plot(forecast_index, forecast, label="Predicción futura", color="green")
plt.title("Predicción futura de caudal mensual")
plt.xlabel("Fecha")
plt.ylabel("Caudal (m³/s)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("prediccion_futura_mejorada.png")
plt.show()
