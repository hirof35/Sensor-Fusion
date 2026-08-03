# 2D Sensor Fusion Simulator in Python

Pythonとカルマンフィルター（Kalman Filter）を用いた2D視覚的センサー・フュージョン（センサ統合）シミュレーターです。
異なるノイズ特性を持つ2つのセンサー（例: GPSおよびLiDAR）からの観測データをリアルタイムに統合し、移動体の真の位置を高精度に推定する過程を視覚的に確認できます。
<img width="1248" height="833" alt="スクリーンショット 2026-08-04 070406" src="https://github.com/user-attachments/assets/66ce6fa4-796e-420b-a22e-11e86e9d77e3" />

## 特徴

- **マルチセンサー統合**: 異種ノイズ特性を持つ2つのセンサーの観測ベクトルを結合し、同時に処理
- **カルマンフィルター実装**: 行列演算を用いた線形カルマンフィルターによる状態推定
- **リアルタイムアニメーション**: `Matplotlib.animation` による軌跡とノイズデータの動的描画

## 動作環境

- Python 3.8+
- NumPy
- Matplotlib

## インストール

必要なライブラリをインストールします。

```bash
pip install numpy matplotlib
実行方法スクリプトを実行すると、アニメーションウィンドウが立ち上がります。Bashpython sensor_fusion_sim.py
動作原理本シミュレーターでは、以下の状態ベクトル $x_k$ を推定します。$$x_k = \begin{bmatrix} x & y & v_x & v_y \end{bmatrix}^T$$予測ステップ: 等速運動モデルに従って次時刻の位置および速度を予測します。更新ステップ: 各センサーの誤差共分散 $R_A, R_B$ に基づき、信頼度の高いセンサーの重みを大きくするようカルマンゲイン $K$ を自動調整して状態を更新します。ライセンスMIT License
