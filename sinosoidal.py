import cv2
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import wave
import pyaudio

PINK = "#e2979c"
RED = "#e7305b"
GREEN = "#00aa00"
BLUE = "#008eff"
YELLOW = "#f7f5dd"
WHITE = '#ffffff'
FONT_NAME = "Times New Roman"

class SinosidalSignal():
    def __init__(self, w: Tk, amplitude=1, frequency=1, phase=0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

        self.amp_slider = Scale(w, from_=0., to=10., resolution=0.1, length=250,
                                orient="horizontal", command=self.change_amp)
        self.freq_slider = Scale(w, from_=0., to=10., resolution=1, length=250,
                                orient="horizontal", command=self.change_freq)
        self.theta_slider = Scale(w, from_=-3.14, to=3.14, resolution=0.01,  length=250,
                                orient="horizontal", command=self.change_theta)
        self.delete_but = Button(w, text="Delete\nFrequency", bg=RED, 
                                font=(FONT_NAME, 15, 'bold'), pady=0)
        self.dash_lab = Label(w, text=40*'-', bg=YELLOW, highlightthickness=0)

        self.amp_slider.set(self.amplitude)
        self.freq_slider.set(self.frequency)
        self.theta_slider.set(self.phase)

    def change_amp(self, val):
        self.amplitude = float(val)
    def change_freq(self, val):
        self.frequency = float(val)
    def change_theta(self, val):
        self.phase = float(val)

    def __repr__(self):
        return f"SinosidalSignal(amplitude={self.amplitude}, frequency={self.frequency}, phase={self.phase})"


class ThePlot():
    def __init__(self, image_width= 500, image_length = 1000, distance_to_pix_ratio_x = 250,
                distance_to_pix_ratio_y = 30, sampling_rate = 50, duration = 20, plot_column_span=3, plot_row_span=15):
        
        self.image_width = image_width
        self.image_length = image_length
        self.distance_to_pix_ratio_x = distance_to_pix_ratio_x
        self.distance_to_pix_ratio_y = distance_to_pix_ratio_y
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.plot_column_span = plot_column_span
        self.plot_row_span = plot_row_span
        self.use_line = 0
        self.original = Canvas(width=self.image_length, height=self.image_width)
        self.with_noise = Canvas(width=self.image_length, height=self.image_width)
        self.filterd = Canvas(width=self.image_length, height=self.image_width)
        self.canvs = [self.original, self.with_noise, self.filterd]

        self.important_amps = []
        self.important_freq = []
        self.important_phases = []

        self.input_t = np.linspace(0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        self.output_image_pix = np.zeros((self.image_width, self.image_length, 3), np.uint8)
        self.output_image_pix2 = np.zeros((self.image_width, self.image_length, 3), np.uint8)
        self.output_image_pix3 = np.zeros((self.image_width, self.image_length, 3), np.uint8)

        self.dtft_t = None
        self.dtft_out_image = np.zeros((500, 600, 3), np.uint8)

    def put_plot(self, img_pix, column=0, row=0, canv_no = 0):
        self.canvs[canv_no].destroy()

        # the canvas in which the main plot is going to be added
        self.canvs[canv_no] = Canvas(width=self.image_length, height=self.image_width)
        pil_image = Image.fromarray(img_pix)
        img = ImageTk.PhotoImage(pil_image) 
        self.canvs[canv_no].create_image(self.image_length/2, self.image_width/2, image=img)
        # canv.pack()
        self.canvs[canv_no].grid(column=column, row=row, rowspan=self.plot_row_span, columnspan=self.plot_column_span)
        self.canvs[canv_no].image = img
        
    def draw_graphlines(self, plot, r1: range= range(-6, int(10*2*np.pi)), r2: range=range(1, 30)):
        for i in r1:
            cv2.line(plot, (int((i)*self.distance_to_pix_ratio_x), 0), 
                     (int((i)*self.distance_to_pix_ratio_x), self.image_width), 
                     (100, 100, 100), 1)
        for i in r2:
            cv2.line(plot, (0, i*self.distance_to_pix_ratio_y + int(self.image_width/2)),
                     (self.image_length, i*self.distance_to_pix_ratio_y + int(self.image_width/2)),
                     (100, 100, 100), 1)
            cv2.line(plot, (0, -i*self.distance_to_pix_ratio_y + int(self.image_width/2)),
                     (self.image_length, -i*self.distance_to_pix_ratio_y + int(self.image_width/2)),
                     (100, 100, 100), 1)
        cv2.line(self.output_image_pix, (self.distance_to_pix_ratio_x, 0),
                (self.distance_to_pix_ratio_x, self.image_width), 
                (255, 255, 255), 2)
        cv2.line(self.output_image_pix, (0, int(self.image_width/2)),
                (self.image_length, int(self.image_width/2)),
                (255, 255, 255), 2)
    
    def calculate_points(self, complex_sino: list[SinosidalSignal], sampling_rate):
        self.draw_graphlines(self.output_image_pix, range(-6, int(self.duration*2*np.pi)))
        
        self.input_t = np.linspace(0, self.duration, int(sampling_rate * self.duration), endpoint=False)
        self.output_data_set = [(t, 0) for t in self.input_t]
        
        for sino in complex_sino:
            self.output_data_set = [(t, y + sino.amplitude * np.cos(2*np.pi * sino.frequency * t + sino.phase)) for t, y in self.output_data_set]

        point = [(t*self.distance_to_pix_ratio_x, -y* self.distance_to_pix_ratio_y + self.image_width/2) for (t, y) in self.output_data_set]
        return point
    
    def plot_lines(self, list_of_graphs: list[list[SinosidalSignal]], sampling_rate):
        complex_sino = list_of_graphs[0]
        point = self.calculate_points(complex_sino, sampling_rate)
        cv2.polylines(self.output_image_pix, [np.array(point, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)

    def plot_points(self, list_of_graphs: list[list[SinosidalSignal]], sampling_rate):
        complex_sino = list_of_graphs[0]
        point = self.calculate_points(complex_sino, sampling_rate)
        for t, y in point:
            cv2.circle(self.output_image_pix, (int(t), int(y)), radius=2, color=(0, 255, 255), thickness=-1)

    def get_audio_data(self, file_path):
        with wave.open(file_path, 'r') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_width = wav_file.getsampwidth()
            self.audio_frame_rate = wav_file.getframerate()
            num_channels = wav_file.getnchannels()

        # Convert frames to a numpy array
        if sample_width == 2:  # 16-bit audio
            self.audio_data = np.frombuffer(frames, dtype=np.int16)
        elif sample_width == 1:  # 8-bit audio
            self.audio_data = np.frombuffer(frames, dtype=np.uint8) - 128  # Center it
        else:
            raise ValueError("Unsupported sample width.")

        if num_channels > 1:
            self.audio_data = np.reshape(self.audio_data, (-1, num_channels))
        
        self.audio_duration = len(self.audio_data) / self.audio_frame_rate
        print(np.max(np.abs(self.audio_data)))

    def plot_sound(self, sound = None, img = None):
        is_out_changed = 0
        is_2_changed = 0
        if type(sound) == type(None):
            sound = self.audio_data
        if type(img) == type(None):
            is_out_changed = 1
            img = self.output_image_pix
        elif img is self.output_image_pix2:
            is_2_changed = 1
        elif img is self.output_image_pix3:
            is_3_changed = 1  

        self.input_t = np.linspace(0, self.audio_duration, num=len(sound))
        self.output_data_set = [(self.input_t[i], sound[i]) for i in range(len(self.input_t))]
        x_ratio = self.image_length/self.audio_duration
        max_amp = np.max(sound) / 100
        point = [(t*x_ratio, -y/max_amp + self.image_width/2) for (t, y) in self.output_data_set]
        
        img = np.zeros((self.image_width, self.image_length, 3), np.uint8)
        cv2.polylines(img, [np.array(point, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=1)
        if is_out_changed:
            self.output_image_pix = img
        elif is_2_changed:
            self.output_image_pix2 = img
        elif is_3_changed:
            self.output_image_pix3 = img

    def estimate_sound(self, wind: Tk, num_of_comp):
        self.input_t = np.linspace(0, self.audio_duration, num=len(self.audio_data))
        print("ps")
        self.estimated_sound_data = sum(self.important_amps[i]*np.cos(2 * np.pi * self.important_freq[i] * self.input_t + self.important_phases[i]) for i in range(num_of_comp))

        self.output_data_set = [(self.input_t[i], self.estimated_sound_data[i]) for i in range(len(self.input_t))]

        # self.estimated_sound_data = (self.estimated_sound_data * 32767 / np.max(np.abs(self.estimated_sound_data))).astype(np.int16)

        x_ratio = self.image_length/self.audio_duration
        max_amp = np.max(self.estimated_sound_data) / 100
        point = [(t*x_ratio, -y/max_amp + self.image_width/2) for (t, y) in self.output_data_set]
        
        self.output_image_pix2 = np.zeros((self.image_width, self.image_length, 3), np.uint8)
        cv2.polylines(self.output_image_pix2, [np.array(point, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=1)
        
        self.put_plot(self.output_image_pix2, row=self.plot_row_span, canv_no=1)

    def generate_sound_wave(self, w: Tk, freqs: str, amps: str):
        w.destroy()
        
        freqs_ = [int(i) for i in freqs.split(',')]
        if amps == '':
            amps_ = [1] * len(freqs)
        else:
            amps_ = amps.split(',')
            amps_ = [int(i) if i != '' else 1 for i in amps_]
            if len(freqs_) > len(amps_):
                amps_.extend([1]*(len(freqs_) - len(amps_)))

        self.audio_duration = 2
        self.input_t = np.linspace(0, self.audio_duration, num=self.audio_duration * 44100)

        self.audio_data = sum(amps_[i]*np.cos(2 * np.pi * freqs_[i] * self.input_t) for i in range(len(freqs_)))

        self.plot_sound()
        self.put_plot(self.output_image_pix)

    def add_noise(self, w: Tk, freq):
        w.destroy()
        # np.max(np.abs(self.audio_data))
        noise = 1000 * np.sin(2 * np.pi * freq * self.input_t)
        self.noisy_audio_data = self.audio_data + noise

        self.plot_sound(self.noisy_audio_data, self.output_image_pix2)
        # self.put_plot(self.output_image_pix)
        self.put_plot(self.output_image_pix2, row=self.plot_row_span, canv_no=1)

    def filter_noise(self, w: Tk, target_freq):
        w.destroy()

        self.audio_duration = 2
        self.input_t = np.linspace(0, self.audio_duration, num=self.audio_duration * 44100)

        N = 101 # filter order
        # normalize the data
        self.noisy_audio_data = self.noisy_audio_data / np.max(np.abs(self.noisy_audio_data)) 

        fc_normalized = target_freq / (44100 / 2)

        n = np.arange(N)
        M = (N - 1)/2
        h = np.sinc(2 * fc_normalized * (n - M))

        window = np.hamming(N)
        h = h * window

        self.filtered_sound = np.convolve(self.noisy_audio_data, h, mode='same')
        print(self.filtered_sound.dtype)

        self.plot_sound(self.filtered_sound, self.output_image_pix3)
        self.put_plot(self.output_image_pix3, row=2*self.plot_row_span, canv_no=2)

    def plot_dft(self, list_of_graphs: list[list[SinosidalSignal]], sampling_rate):
        self.output_image_pix = np.zeros((self.image_width, self.image_length, 3), np.uint8)
        complex_sino = list_of_graphs[0]
        self.input_t = np.linspace(0, self.duration, int(sampling_rate * self.duration), endpoint=False)
        output_data_set = [(t, 0) for t in self.input_t]
        
        for sino in complex_sino:
            output_data_set = [(t, y + sino.amplitude * np.cos(2*np.pi * sino.frequency * t + sino.phase)) for t, y in output_data_set]

        point = [(t*self.distance_to_pix_ratio_x, -y* self.distance_to_pix_ratio_y + self.image_width/2) for (t, y) in output_data_set]
        cv2.polylines(self.output_image_pix, [np.array(point, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=2)

    def sampling_rate_changer(self, value):
        value = int(value)
        threshhold = 50
        if value > threshhold:
            value -= threshhold
            value = value*5 + threshhold
        self.sampling_rate = value
        self.sampling_rate_lab.config(text=f'Sampling Rate: {self.sampling_rate}')
        
    def width_adj_changer(self, value, func= None):
        self.distance_to_pix_ratio_x = int(value)
        if func:
            func()

    def length_adj_changer(self, value, func= None):
        self.distance_to_pix_ratio_y = int(value)
        if func:
            func()

