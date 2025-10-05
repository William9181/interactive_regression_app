# HW1
write python to solve simple linear regression problem, following CRISP-DM steps, 要有prompt and 過程, 不可只有CODE AND RESULT
1. CRISP-DM
2. allow user to modify a in ax+b, noise, number of points 
3. streamlit or flask web, 框架 deployment

# setup
## 步驟 1: 安裝必要的函式庫
您需要安裝 streamlit, pandas, scikit-learn, matplotlib 和 seaborn。
打開您的終端機或命令提示字元，執行以下指令：
```bash
pip install streamlit pandas scikit-learn matplotlib seaborn
```
## 步驟 2: 執行 Streamlit 應用程式
確保 interactive_regression_app.py 檔案在您目前的工作目錄中。

在終端機中，執行以下指令：

```bash
streamlit run interactive_regression_app.py
```

執行後，Streamlit 會在您的預設網頁瀏覽器中自動開啟一個新分頁，您就可以看到並操作這個應用程式了。

## 如何使用

左側側邊欄: 您可以拖動滑桿來調整生成數據的各種參數，包括數據的線性關係、噪聲大小和數據點數量。

即時更新: 每當您調整任何參數，主畫面的圖表、模型結果和評估指標都會自動重新計算並更新。

互動預測: 在側邊欄下方，您可以輸入一個溫度值，然後點擊按鈕來獲得模型對該溫度的銷售額預測。

# Gemini prompt & response
- [Gemini-prompt](https://g.co/gemini/share/6cfc8fc4a45b)

# Streamlit app
已將 APP 部屬至 Streamlit Community Cloud，可從以下網址進入使用
- [Streamlit-app](https://interactiveregressionapp-william.streamlit.app/)
