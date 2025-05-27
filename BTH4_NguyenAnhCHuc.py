import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

class VibrationSignalProcessor:
    def __init__(self, data_path, fs=1000, order=5):
        self.df = pd.read_csv(data_path)
        self.fs = fs
        self.order = order
        self.N = len(self.df)
        self.t = np.arange(self.N) / fs
        self.axis_labels = ['X', 'Y', 'Z']
        if self.df.shape[1] < 3:
            raise ValueError("Dữ liệu phải có ít nhất 3 cột tương ứng X, Y, Z")

    def lowpass_filter(self, data, cutoff):
        nyq = 0.5 * self.fs
        norm_cutoff = cutoff / nyq
        b, a = butter(self.order, norm_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def plot_signal(self, samples=2000):
        plt.figure(figsize=(15, 8))
        for i in range(3):
            signal = self.df.iloc[:, i].values
            plt.subplot(3, 1, i + 1)
            plt.plot(self.t[:samples], signal[:samples])
            plt.title(f"Tín hiệu gốc trục {self.axis_labels[i]} (2000 mẫu đầu)")
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_spectrum(self):
        plt.figure(figsize=(15, 8))
        for i in range(3):
            signal = self.df.iloc[:, i].values
            yf = np.abs(fft(signal))[:self.N // 2]
            xf = fftfreq(self.N, 1 / self.fs)[:self.N // 2]
            plt.subplot(3, 1, i + 1)
            plt.plot(xf, yf)
            plt.title(f'Phổ tần số tín hiệu trục {self.axis_labels[i]}')
            plt.xlabel('Tần số (Hz)')
            plt.ylabel('Biên độ')
            plt.grid()
            plt.xlim(0, self.fs / 2)
        plt.tight_layout()
        plt.show()

    def compare_cutoffs(self, cutoff_values=None, samples=2000, axis_index=0):
        if cutoff_values is None:
            cutoff_values = [20, 50, 80, 120, 200]
        signal = self.df.iloc[:, axis_index].values
        plt.figure(figsize=(15, 8))
        plt.plot(self.t[:samples], signal[:samples], label='Tín hiệu gốc', alpha=0.3)
        for cutoff in cutoff_values:
            filtered = self.lowpass_filter(signal, cutoff)
            plt.plot(self.t[:samples], filtered[:samples], label=f'Cutoff = {cutoff} Hz')
        plt.title(f'So sánh tín hiệu lọc với các cutoff khác nhau - Trục {self.axis_labels[axis_index]}')
        plt.xlabel('Thời gian (s)')
        plt.ylabel('Biên độ')
        plt.legend()
        plt.grid()
        plt.show()

    def filter_all_axes(self, cutoff=80):
        filtered_data = {}
        for i in range(3):
            signal = self.df.iloc[:, i].values
            filtered_data[self.axis_labels[i]] = self.lowpass_filter(signal, cutoff)
        return pd.DataFrame(filtered_data)

    def compare_before_after_filter(self, cutoff=80, samples=2000):
        filtered_df = self.filter_all_axes(cutoff)
        plt.figure(figsize=(15, 10))
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(self.t[:samples], self.df.iloc[:samples, i], label='Tín hiệu gốc', alpha=0.4)
            plt.plot(self.t[:samples], filtered_df.iloc[:samples, i], label=f'Tín hiệu đã lọc (cutoff={cutoff}Hz)', linewidth=1.5)
            plt.title(f'So sánh trước và sau lọc - Trục {self.axis_labels[i]}')
            plt.xlabel('Thời gian (s)')
            plt.ylabel('Biên độ')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.show()

    def save_filtered(self, output_path, cutoff=80):
        filtered_df = self.filter_all_axes(cutoff)
        filtered_df.to_csv(output_path, index=False)
        print(f"Đã lưu dữ liệu đã lọc vào: {output_path}")

# Ví dụ sử dụng
if __name__ == "__main__":
    data_path = 'E:\code\TimeSeries\BTH4\XYZ_N(1).csv'  # Thay đường dẫn file dữ liệu
    processor = VibrationSignalProcessor(data_path=data_path, fs=1000, order=5)

    # Hiển thị tín hiệu gốc cả 3 trục
    processor.plot_signal()

    # Hiển thị phổ tần số cả 3 trục
    processor.plot_spectrum()

    # So sánh các cutoff trên trục X (có thể thay axis_index=1 hoặc 2 để thử Y hoặc Z)
    processor.compare_cutoffs(axis_index=0)

    # So sánh tín hiệu trước/sau lọc cho cả 3 trục với cutoff = 80 Hz
    processor.compare_before_after_filter(cutoff=80)

    # Lọc và lưu dữ liệu đã lọc ra file CSV mới
    # processor.save_filtered('E:\code\TimeSeries\BTH4\XYZ_N(1).csv', cutoff=80)
