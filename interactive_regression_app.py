import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# --- 網頁設定 ---
st.set_page_config(layout="wide", page_title="互動式線性迴歸模型")

# 由於圖表標籤改為英文，不再需要特別設定中文字體
plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號
plt.style.use('seaborn-v0_8-whitegrid')

# --- 主標題 ---
st.title("互動式線性迴歸模型展示 (CRISP-DM 流程)")
st.markdown("透過調整左側的參數，觀察數據分佈和模型結果的變化。")

# --- 側邊欄：使用者輸入 ---
st.sidebar.header("1. 調整數據生成參數")
st.sidebar.markdown("請調整 `y = ax + b + noise` 中的參數")

# 使用者可調整的參數
param_a = st.sidebar.slider('斜率 (a)', min_value=0.0, max_value=50.0, value=10.0, step=0.5)
param_b = st.sidebar.slider('截距 (b)', min_value=0.0, max_value=100.0, value=50.0, step=5.0)
param_noise = st.sidebar.slider('噪聲標準差 (noise)', min_value=0.0, max_value=100.0, value=20.0, step=5.0)
param_num_points = st.sidebar.slider('數據點數量', min_value=20, max_value=1000, value=100, step=10)
param_test_size = st.sidebar.slider('測試集比例', min_value=0.1, max_value=0.5, value=0.3, step=0.05)

# --- CRISP-DM 流程 ---

# === 步驟 1: 商業理解 (Business Understanding) ===
with st.expander("第一步：商業理解", expanded=True):
    st.markdown("""
    **情境:** 一家冰淇淋店的老闆想了解「每日溫度」和「冰淇淋銷售額」之間的關係。
    
    **目標:** 建立一個預測模型，根據溫度預測冰淇淋的銷售額，以便更有效地管理庫存和人力。
    """)

# === 步驟 2: 資料理解 (Data Understanding) ===
st.header("第二步：資料理解")

# 2.1 資料收集 (根據使用者參數動態生成)
np.random.seed(42)
temperatures = 15 + 20 * np.random.rand(param_num_points)
sales = param_b + param_a * temperatures + np.random.randn(param_num_points) * param_noise
df = pd.DataFrame({'Temperature': temperatures, 'Sales': sales})

# 佈局
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("生成數據預覽")
    st.dataframe(df.head())
    st.subheader("描述性統計")
    st.dataframe(df.describe())

with col2:
    st.subheader("Temperature vs. Sales Relationship")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Temperature', y='Sales', data=df, ax=ax1)
    ax1.set_title('Data Distribution Scatter Plot', fontsize=16)
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Sales', fontsize=12)
    st.pyplot(fig1)

# === 步驟 3: 資料準備 (Data Preparation) ===
st.header("第三步：資料準備")
X = df[['Temperature']].values
y = df['Sales'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=param_test_size, random_state=42)
st.write(f"資料已分割完成，訓練集大小: `{X_train.shape[0]}` 筆，測試集大小: `{X_test.shape[0]}` 筆。")

# === 步驟 4: 模型建立 (Modeling) ===
st.header("第四步：模型建立")
model = LinearRegression()
model.fit(X_train, y_train)

st.success(f"模型訓練完成！學習到的迴歸方程式為: **Sales = {model.coef_[0]:.2f} * Temperature + {model.intercept_:.2f}**")

# === 步驟 5: 模型評估 (Evaluation) ===
st.header("第五步：模型評估")
y_pred = model.predict(X_test)

# 佈局
col3, col4 = st.columns([1, 2])

with col3:
    st.subheader("模型效能指標")
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    
    st.metric(label="R-squared (R²)", value=f"{r2:.3f}")
    st.metric(label="平均絕對誤差 (MAE)", value=f"{mae:.2f}")
    st.metric(label="均方根誤差 (RMSE)", value=f"{rmse:.2f}")


with col4:
    st.subheader("Model Prediction Evaluation")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(X_test, y_test, color='blue', label='Actual Sales')
    ax2.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
    ax2.set_title('Model Performance on Test Set', fontsize=16)
    ax2.set_xlabel('Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Sales', fontsize=12)
    ax2.legend()
    st.pyplot(fig2)

# === 步驟 6: 部署 (Deployment) ===
st.header("第六步：部署與互動預測")
st.sidebar.header("2. 進行即時預測")
user_temp = st.sidebar.number_input('輸入一個溫度 (°C) 進行預測:', min_value=-10.0, max_value=50.0, value=25.0, step=0.5)

if st.sidebar.button('預測銷售額'):
    predicted_sale = model.predict(np.array([[user_temp]]))
    st.sidebar.info(f"當溫度為 **{user_temp}°C** 時，預測的銷售額約為 **{predicted_sale[0]:.2f}** 元。")


