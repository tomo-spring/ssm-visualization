import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legfit, legval
import pandas as pd
import japanize_matplotlib

def generate_signal(t, signal_type, freq):
    if signal_type == "正弦波":
        return np.sin(2 * np.pi * freq * t)
    elif signal_type == "矩形波":
        return np.sign(np.sin(2 * np.pi * freq * t))
    elif signal_type == "三角波":
        return 2 * np.arcsin(np.sin(2 * np.pi * freq * t)) / np.pi

def plot_signal(t, signal, title):
    fig, ax = plt.subplots()
    ax.plot(t, signal)
    ax.set(xlabel="時間", ylabel="振幅", title=title)
    return fig

def fourier_analysis(signal, t):
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(t), d=(t[1]-t[0]))
    return fft_vals, freqs

def legendre_analysis(signal, t, degree):
    t_mapped = 2 * t - 1
    return legfit(t_mapped, signal, deg=degree)

def sliding_window_analysis(signal, t, window_size, step_size, analysis_func):
    results = []
    for start_idx in range(0, len(signal) - window_size + 1, step_size):
        end_idx = start_idx + window_size
        window_result = analysis_func(signal[start_idx:end_idx], t[start_idx:end_idx])
        results.append(window_result)
    return np.array(results)

st.title("三角関数とルジャンドル多項式による時系列データの近似と係数の可視化")

# Input settings
st.subheader("入力信号設定")
total_time = st.number_input("総信号の長さ", min_value=100, max_value=10000, value=1000)
window_size = st.number_input("ウィンドウサイズ", min_value=10, max_value=total_time, value=200)
step_size = st.number_input("ステップサイズ", min_value=1, max_value=window_size, value=10)

t_total = np.linspace(0, 1, int(total_time))

signal_type = st.selectbox("信号の種類", ("正弦波", "矩形波", "三角波", "カスタム"))
if signal_type != "カスタム":
    freq = st.slider("周波数", 1, 50, 5)
    signal_total = generate_signal(t_total, signal_type, freq)
else:
    uploaded_file = st.file_uploader("CSVファイルをアップロード（1列のデータ）", type="csv")
    if uploaded_file is not None:
        signal_total = np.loadtxt(uploaded_file)
        total_time = len(signal_total)
        t_total = np.linspace(0, 1, total_time)
    else:
        st.warning("カスタム信号を使用するにはCSVファイルをアップロードしてください。")
        st.stop()

st.subheader("入力信号の概形")
st.pyplot(plot_signal(t_total, signal_total, "入力信号"))

# Approximation settings
st.subheader("係数の調整")
num_components_fourier = st.slider("三角関数の周波数成分数", 1, total_time//2, 10)
max_degree_legendre = st.slider("ルジャンドル多項式の最大次数", 1, 50, 10)

# Fourier approximation
fft_vals, freqs = fourier_analysis(signal_total, t_total)
indices_fourier = np.argsort(np.abs(fft_vals))[-num_components_fourier:]
fft_filtered = np.zeros_like(fft_vals)
fft_filtered[indices_fourier] = fft_vals[indices_fourier]
signal_approx_fourier = np.fft.ifft(fft_filtered).real

# Legendre approximation
coeffs_legendre = legendre_analysis(signal_total, t_total, max_degree_legendre)
signal_approx_legendre = legval(2 * t_total - 1, coeffs_legendre)

# Comparison plot
st.subheader("各手法による近似の概形")
fig_comparison, ax_comparison = plt.subplots()
ax_comparison.plot(t_total, signal_total, label="元の信号")
ax_comparison.plot(t_total, signal_approx_fourier, label="三角関数近似")
ax_comparison.plot(t_total, signal_approx_legendre, label="ルジャンドル多項式近似")
ax_comparison.set(xlabel="時間", ylabel="振幅", title="三角関数とルジャンドル多項式による入力信号の近似比較")
ax_comparison.legend()
st.pyplot(fig_comparison)

# Error calculation
st.subheader("各手法における誤差")
error_fourier = np.mean((signal_total - signal_approx_fourier)**2)
error_legendre = np.mean((signal_total - signal_approx_legendre)**2)
error_data = {
    "近似手法": ["三角関数による近似", "ルジャンドル多項式による近似"],
    "平均二乗誤差": [f"{error_fourier:.6f}", f"{error_legendre:.6f}"]
}
st.table(pd.DataFrame(error_data).set_index("近似手法"))

# Fourier coefficient plot
st.subheader("三角関数近似の各周波数成分の係数")
min_freq, max_freq = st.slider(
    "表示する周波数範囲",
    float(freqs[1]),
    float(freqs[total_time//2 - 1]),
    (float(freqs[1]), float(freqs[total_time//4]))
)
indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]

fig_fourier, (ax_cos, ax_sin) = plt.subplots(2, 1, figsize=(8, 6))
ax_cos.stem(freqs[indices], np.real(fft_vals)[indices])
ax_sin.stem(freqs[indices], np.imag(fft_vals)[indices])
ax_cos.set(ylabel="余弦波係数")
ax_sin.set(xlabel="周波数", ylabel="正弦波係数")
fig_fourier.suptitle("三角関数近似における正弦波と余弦波成分の係数")
st.pyplot(fig_fourier)

# Legendre coefficient plot
st.subheader("ルジャンドル多項式近似の各次数成分の係数")
num_coeffs_display = st.slider(
    "表示する次数範囲",
    1,
    len(coeffs_legendre),
    min(10, len(coeffs_legendre))
)
fig_legendre, ax_legendre = plt.subplots()
ax_legendre.stem(range(num_coeffs_display), coeffs_legendre[:num_coeffs_display])
ax_legendre.set(xlabel="次数", ylabel="係数", title="ルジャンドル多項式近似における各次数成分の係数")
st.pyplot(fig_legendre)

# Time-varying Fourier analysis
st.header("フーリエ係数の時間変化")
num_components = st.slider("表示するフーリエ成分の数", 1, window_size//2, 10)

def fourier_analysis_window(signal, t):
    fft_vals = np.fft.fft(signal)
    indices = np.argsort(np.abs(fft_vals))[-num_components:]
    return np.abs(fft_vals[indices])

coeffs_array = sliding_window_analysis(signal_total, t_total, window_size, step_size, fourier_analysis_window)
time_steps = np.arange(0, len(coeffs_array) * step_size, step_size)

# 3D plot for Fourier coefficients
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set(xlabel='周波数', ylabel='時間（サンプル）', zlabel='係数の大きさ', title='フーリエ係数の時間変化')

for i, coeffs in enumerate(coeffs_array):
    xs = freqs[indices_fourier]
    ys = np.full_like(xs, time_steps[i])
    zs = np.zeros_like(xs)
    dx = (xs.max() - xs.min()) / num_components * 0.8
    dy = step_size * 0.8
    dz = coeffs
    ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, alpha=0.7)

st.pyplot(fig)

# Time-varying Legendre analysis
st.header("ルジャンドル多項式の係数の時間変化")

def legendre_analysis_window(signal, t):
    t_mapped = 2 * t - 1
    return legfit(t_mapped, signal, deg=max_degree_legendre)

coeffs_legendre_array = sliding_window_analysis(signal_total, t_total, window_size, step_size, legendre_analysis_window)
time_steps_legendre = np.arange(window_size//2, total_time - window_size//2 + 1, step_size)

# 3D plot for Legendre coefficients
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set(xlabel='次数', ylabel='時間', zlabel='係数の大きさ', title='ルジャンドル多項式の係数の時間変化')

for i, coeffs in enumerate(coeffs_legendre_array):
    xs = np.arange(max_degree_legendre + 1)
    ys = np.full_like(xs, time_steps_legendre[i])
    zs = np.zeros_like(xs)
    dx = np.ones_like(xs) * 0.5
    dy = (step_size / total_time) * 0.5
    dz = coeffs
    ax.bar3d(xs, ys, zs, dx, dy, dz, shade=True, alpha=0.7)

st.pyplot(fig)

# Colormap for Fourier coefficients
st.subheader("三角関数近似の各周波数成分の係数の時間変化のカラーマップ表示")
fig_fourier_cmap, ax_fourier_cmap = plt.subplots(figsize=(10, 6))
c = ax_fourier_cmap.pcolormesh(time_steps_legendre, freqs[indices_fourier], coeffs_array.T, shading='auto', cmap='viridis')
ax_fourier_cmap.set(xlabel="時間", ylabel="周波数", title="三角関数近似における各周波数成分の係数の時間変化カラーマップ")
fig_fourier_cmap.colorbar(c, ax=ax_fourier_cmap, label='係数の大きさ')
st.pyplot(fig_fourier_cmap)

# Colormap for Legendre coefficients
st.subheader("ルジャンドル多項式近似の各次数成分の係数の時間変化のカラーマップ表示")
fig_legendre_cmap, ax_legendre_cmap = plt.subplots(figsize=(10, 6))
c_leg = ax_legendre_cmap.pcolormesh(time_steps_legendre, np.arange(max_degree_legendre + 1), coeffs_legendre_array.T, shading='auto', cmap='plasma')
ax_legendre_cmap.set(xlabel="時間", ylabel="次数", title="ルジャンドル多項式近似における各次数成分の係数の時間変化カラーマップ")
fig_legendre_cmap.colorbar(c_leg, ax=ax_legendre_cmap, label='係数の大きさ')
st.pyplot(fig_legendre_cmap)
