import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. シミュレーション設定 ---
np.random.seed(42)
dt = 0.1  # サンプリング周期 [s]
num_steps = 150

# 真の軌跡（等速直線運動 + ゆるやかなカーブ）
t = np.linspace(0, num_steps * dt, num_steps)
true_x = 2 * t + 5 * np.sin(0.5 * t)
true_y = 1.5 * t + 3 * np.cos(0.5 * t)
true_states = np.vstack([true_x, true_y])

# --- 2. センサーノイズの設定 ---
# センサーA（例: GPS系 - X軸にノイズが大きい）
R_a = np.diag([4.0, 1.0])  # 共分散行列 (σx^2=4.0, σy^2=1.0)
noise_a = np.random.multivariate_normal([0, 0], R_a, num_steps).T
z_a = true_states + noise_a

# センサーB（例: LiDAR/Radar系 - Y軸にノイズが大きい）
R_b = np.diag([1.0, 4.0])  # 共分散行列 (σx^2=1.0, σy^2=4.0)
noise_b = np.random.multivariate_normal([0, 0], R_b, num_steps).T
z_b = true_states + noise_b

# --- 3. カルマンフィルターの初期化 ---
# 状態ベクトル x = [pos_x, pos_y, vel_x, vel_y]^T
x = np.array([[z_a[0, 0]], [z_a[1, 0]], [0.0], [0.0]])

# 状態移行行列 A
A = np.array([
    [1, 0, dt,  0],
    [0, 1,  0, dt],
    [0, 0,  1,  0],
    [0, 0,  0,  1]
])

# 観測行列 H（2つのセンサーの観測 [pos_xA, pos_yA, pos_xB, pos_yB]^T を一度に結合）
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

# プロセスノイズ共分散 P, Q
P = np.eye(4) * 10.0
q_val = 0.1
Q = np.eye(4) * q_val

# 結合観測ノイズ共分散 R
R = np.block([
    [R_a,               np.zeros((2, 2))],
    [np.zeros((2, 2)), R_b             ]
])

# 推定結果の保存配列
fused_x = []
fused_y = []

# --- 4. カルマンフィルター ループ ---
for k in range(num_steps):
    # 【予測ステップ (Predict)】
    x = A @ x
    P = A @ P @ A.T + Q
    
    # 観測ベクトルの構築 (センサーAとBの同時観測)
    z_k = np.array([[z_a[0, k]], [z_a[1, k]], [z_b[0, k]], [z_b[1, k]]])
    
    # 【更新ステップ (Update)】
    y = z_k - H @ x                        # 観測残差
    S = H @ P @ H.T + R                    # 残差共分散
    K = P @ H.T @ np.linalg.inv(S)         # カルマンゲイン
    
    x = x + K @ y                          # 状態の更新
    P = (np.eye(4) - K @ H) @ P            # 共分散の更新
    
    fused_x.append(x[0, 0])
    fused_y.append(x[1, 0])

# --- 5. Matplotlibによるアニメーション描画 ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(min(true_x) - 5, max(true_x) + 5)
ax.set_ylim(min(true_y) - 5, max(true_y) + 5)
ax.set_title("2D Sensor Fusion Simulator (Kalman Filter)", fontsize=14)
ax.set_xlabel("X Position [m]")
ax.set_ylabel("Y Position [m]")
ax.grid(True, linestyle="--", alpha=0.6)

# プロット要素の初期化
line_true, = ax.plot([], [], 'k--', label='True Path', linewidth=2)
scat_a = ax.scatter([], [], c='red', alpha=0.5, label='Sensor A (GPS-like)', s=20)
scat_b = ax.scatter([], [], c='blue', alpha=0.5, label='Sensor B (LiDAR-like)', s=20)
line_fused, = ax.plot([], [], 'g-', label='Fused Estimate (EKF/KF)', linewidth=2.5)

ax.legend(loc='upper left')

def init():
    line_true.set_data([], [])
    scat_a.set_offsets(np.empty((0, 2)))
    scat_b.set_offsets(np.empty((0, 2)))
    line_fused.set_data([], [])
    return line_true, scat_a, scat_b, line_fused

def update(frame):
    line_true.set_data(true_x[:frame], true_y[:frame])
    scat_a.set_offsets(np.column_stack((z_a[0, :frame], z_a[1, :frame])))
    scat_b.set_offsets(np.column_stack((z_b[0, :frame], z_b[1, :frame])))
    line_fused.set_data(fused_x[:frame], fused_y[:frame])
    return line_true, scat_a, scat_b, line_fused

ani = FuncAnimation(fig, update, frames=num_steps, init_func=init, interval=50, blit=True)
plt.show()
