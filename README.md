◻️Stock Price Prediction with LSTM

このプロジェクトは、LSTM（Long Short-Term Memory）を使ってトヨタ自動車（7203.T）の株価予測を行うPythonアプリケーションです。
Yahoo Finance から株価データを取得し、PyTorch によるディープラーニングモデルで未来の株価を予測します。


◻️ 使用技術

Python 3.x
PyTorch
Yahoo Finance API (yfinance)
NumPy / Pandas / scikit-learn
Matplotlib


◻️ ディレクトリ構成

.
├── stock_predict.py       # メインの実行スクリプト
├── toyota_stock_data.csv  # ダウンロードされた株価データ
├── README.md              # このファイル


◻️ 実行方法

1. 必要なライブラリのインストール
pip install yfinance torch pandas numpy matplotlib scikit-learn

3. スクリプトを実行
python3 stock_predict.py


◻️　モデルの内容

過去80日間の「終値」を使用し、翌日の終値を予測します。
データは MinMaxScaler で 0〜1 の範囲に正規化。
LSTM レイヤーを1層使用。
平均二乗誤差（MSE）を損失関数として採用。


◻️ 学習結果、改善点

2025-5-1
Loss値は4~8%程度。スリムなTransformerやRNNでまだ改善の余地はあり。
損失関数やハイパーパラメータの見直しも必要そう。










