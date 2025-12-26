
import pyfftw
import numpy as np
import torch
import matplotlib.pyplot as plt


def Exampel01():
  # Создаём тестовый сигнал
  signal = np.array([1, 2, 3, 4, 3, 2, 1, 0], dtype=np.float32)

  # FFT (прямое преобразование)
  fft_result = pyfftw.interfaces.numpy_fft.fft(signal)
  print("FFT результат:", fft_result)

  # IFFT (обратное преобразование)
  ifft_result = pyfftw.interfaces.numpy_fft.ifft(fft_result)
  print("IFFT результат:", ifft_result.real)  # .real для удаления шума

  # Проверка: должно совпадать с исходным сигналом
  print("Исходный сигнал:", signal)

def Exampel02():

  # Матрица 8x8
  matrix = np.array([
    [1, 2, 3, 4, 3, 2, 1, 0],
    [0, 1, 2, 3, 4, 3, 2, 1],
    [1, 0, 1, 2, 3, 2, 1, 0],
    [2, 1, 0, 1, 2, 3, 2, 1],
    [3, 2, 1, 0, 1, 2, 3, 2],
    [2, 3, 2, 1, 0, 1, 2, 3],
    [1, 2, 3, 2, 1, 0, 1, 2],
    [0, 1, 2, 3, 2, 1, 0, 1]
  ], dtype=np.float32)

  # 2D FFT
  fft_2d = pyfftw.interfaces.numpy_fft.fft2(matrix)
  print("2D FFT shape:", fft_2d.shape)

  # 2D IFFT
  ifft_2d = pyfftw.interfaces.numpy_fft.ifft2(fft_2d)
  print("Восстановленная матрица:\n", ifft_2d.real)

def Example03():

  # Генерируем сигнал: смесь двух синусоид
  fs = 1000  # Частота дискретизации
  t = np.linspace(0, 1, fs, dtype=np.float32)
  signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)

  # FFT
  fft_result = pyfftw.interfaces.numpy_fft.fft(signal)
  magnitude = np.abs(fft_result)

  # Частоты
  frequencies = np.fft.fftfreq(len(signal), 1 / fs)

  # Построение спектра (только положительные частоты)
  plt.figure(figsize=(10, 4))
  plt.plot(frequencies[:fs // 2], magnitude[:fs // 2])
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Magnitude')
  plt.title('Спектр сигнала')
  plt.grid()
  plt.show()

  # IFFT (восстановление)
  recovered_signal = pyfftw.interfaces.numpy_fft.ifft(fft_result).real
  print("MAE:", np.mean(np.abs(signal - recovered_signal)))

def Example04():

  # Создаём сигнал на CPU
  signal_np = np.array([1, 2, 3, 4, 3, 2, 1, 0], dtype=np.float32)

  # Переводим на GPU
  signal = torch.from_numpy(signal_np).to('cuda')  # На GPU
  print(f"Signal device: {signal.device}")

  # FFT (прямое преобразование)
  fft_result = torch.fft.fft(signal)
  print("FFT результат:", fft_result)

  # IFFT (обратное преобразование)
  ifft_result = torch.fft.ifft(fft_result)
  print("IFFT результат:", ifft_result.real)

  # Проверка (должно совпадать)
  print("Исходный сигнал:", signal.cpu().numpy())

def Example05():

  # Генерируем сигнал: смесь синусоид
  fs = 1000
  t = torch.linspace(0, 1, fs, device='cuda')
  signal = torch.sin(2 * np.pi * 50 * t) + 0.5 * torch.sin(2 * np.pi * 150 * t)

  # FFT на GPU
  fft_result = torch.fft.fft(signal)
  magnitude = torch.abs(fft_result)

  # Переводим в numpy для визуализации
  magnitude_np = magnitude.cpu().numpy()
  frequencies = np.fft.fftfreq(len(signal), 1 / fs)

  # График
  plt.figure(figsize=(10, 4))
  plt.plot(frequencies[:fs // 2], magnitude_np[:fs // 2])
  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Magnitude')
  plt.title('Спектр сигнала (GPU вычисления)')
  plt.grid()
  plt.show()

  # IFFT
  recovered = torch.fft.ifft(fft_result).real
  print("MAE:", torch.mean(torch.abs(signal - recovered)))

if __name__ == '__main__':
  Example05()
