import cv2
from tkinter import *
from tkinter import  filedialog
from PIL import Image, ImageTk
from sinosoidal import *
import matplotlib.pyplot as mplt
import wave
import pyaudio


PINK = "#e2979c"
RED = "#e7305b"
GREEN = "#00aa00"
BLUE = "#008eff"
YELLOW = "#f7f5dd"
WHITE = '#ffffff'
FONT_NAME = "Times New Roman"

class UI():
    def __init__(self, right_column_start=3, bellow_raw_start=15):
        self.PINK = "#e2979c"
        self.RED = "#e7305b"
        self.GREEN = "#00aa00"
        self.BLUE = "#008eff"
        self.YELLOW = "#f7f5dd"
        self.WHITE = '#ffffff'
        self.FONT_NAME = "Times New Roman"
        self.mode = 0
        self.right_column_start = right_column_start
        self.bellow_raw_start = bellow_raw_start
        self.dft_siganl = [[]]

    def clear_window(self, window: Tk):
        for widget in window.winfo_children():
            widget.grid_remove()

    def welcom_setup(self, wind: Tk, list_of_graphs: list[list[SinosidalSignal]], plt: ThePlot, mode1_update):
        self.wind = wind
        self.list_of_graphs = list_of_graphs
        self.plt = plt
        self.mode1_update = mode1_update

        self.clear_window(wind)
        wind.title('Signal Processor')
        wind.config(padx=100, pady=100, bg=YELLOW)

        welcom_lab = Label(wind, text='Signal & Sound Processor', bg=YELLOW, fg=GREEN,
                           highlightthickness=0, font=(FONT_NAME, 30, 'bold'),
                           padx=100, pady=100)
        welcom_lab.grid(row=0, column=0, columnspan=3)

        signal_pro_button = Button(wind, text="Simple Signal Processor", bg=GREEN,
                                   command=lambda: self.ui_mode1_setup(wind, list_of_graphs, plt, mode1_update),
                                   font=(FONT_NAME, 25, 'bold'))
        signal_pro_button.grid(row=1, column=0)

        dft_pro_button = Button(wind, text="DFT Calculator", bg=PINK,
                                   command=lambda: self.ui_mode2_setup(wind, plt),
                                   font=(FONT_NAME, 25, 'bold'))
        dft_pro_button.grid(row=1, column=1)

        sound_pro_button = Button(wind, text="Sound Processor", bg=RED,
                                   command=lambda: self.ui_mode3_setup(wind, plt),
                                   font=(FONT_NAME, 25, 'bold'))
        sound_pro_button.grid(row=1, column=2)

    def ui_mode3_setup(self, wind: Tk, plt: ThePlot):
        self.mode = 3
        self.clear_window(wind)
        wind.config(padx=20, pady=20)
        wind.title('Sound Processor')

        plt.distance_to_pix_ratio_y = 30
        plt.distance_to_pix_ratio_x = 250
        plt.image_width = 250
        plt.plot_row_span = 5
        plt.put_plot(plt.output_image_pix)

        back_to_main_button = Button(wind, text="Back", bg=RED, width=8,
                                     command=self.back_to_main,
                                     font=(FONT_NAME, 15, 'bold'))
        back_to_main_button.grid(column=self.right_column_start+1, row=0)

        choose_audio_file_button = Button(
            wind, text="Choose Audio File", bg=BLUE,
            command=lambda: self.choose_audio_file(wind, plt),
            font=(FONT_NAME, 15, 'bold')
        )
        choose_audio_file_button.grid(column=self.right_column_start, row=0)

        play_audio_button = Button(
            wind, text="Play sound", bg=GREEN, width=22,
            command=lambda: self.play_audio(wind, plt, plt.audio_data, plt.output_image_pix),
            font=(FONT_NAME, 15, 'bold')
        )
        play_audio_button.grid(column=self.right_column_start, row=1, columnspan=2)

        calc_audio_dft_button = Button(
            wind, text="Calculate DFT", bg=BLUE, width=22,
            command=lambda: self.calc_dft(wind, plt, 44100, plt.audio_duration, plt.audio_data),
            font=(FONT_NAME, 15, 'bold')
        )
        calc_audio_dft_button.grid(column=self.right_column_start, row=2, columnspan=2)

        plot_num_of_component_button = Button(
            wind, text='Plot the first X\nnum of components',
            font=(FONT_NAME, 14, 'bold'), bg=GREEN,
            command=lambda: self.plot_num_of_component(wind, plt)
        )
        plot_num_of_component_button.grid(column=self.right_column_start, row=3)

        self.plot_num_of_component_in = Entry(wind, font=(FONT_NAME, 15), width=8)
        self.plot_num_of_component_in.grid(column=self.right_column_start+1, row=3)

        play_estimated_sound_button = Button(
            wind, text="Play Estimated sound", bg=BLUE, width=22,
            command=lambda: self.play_audio(wind, plt, plt.estimated_sound_data, plt.output_image_pix2),
            font=(FONT_NAME, 15, 'bold')
        )
        play_estimated_sound_button.grid(column=self.right_column_start, row=4, columnspan=2)

        generate_sound_wave_button = Button(
            wind, text="Generate Sound", bg=GREEN, width=22,
            command=lambda: self.generate_sound_wave(plt),
            font=(FONT_NAME, 15, 'bold')
        )
        generate_sound_wave_button.grid(column=self.right_column_start, row=5, columnspan=2)

        add_noise_button = Button(
            wind, text="Add Noise", bg=RED, width=22,
            command=lambda: self.add_noise(plt),
            font=(FONT_NAME, 15, 'bold')
        )
        add_noise_button.grid(column=self.right_column_start, row=6, columnspan=2)

        play_noisy_sound_button = Button(
            wind, text="Play Noisy Sound", bg=BLUE,
            command=lambda: self.play_audio(wind, plt, plt.noisy_audio_data, plt.output_image_pix2), width=22,
            font=(FONT_NAME, 15, 'bold')
        )
        play_noisy_sound_button.grid(column=self.right_column_start, row=7, columnspan=2)

        filter_noise_button = Button(
            wind, text="Filter Noisy Sound", bg=GREEN,
            command=lambda: self.filter_noise(plt), width=22,
            font=(FONT_NAME, 15, 'bold')
        )
        filter_noise_button.grid(column=self.right_column_start, row=8, columnspan=2)

        play_filtered_sound_button = Button(
            wind, text="Play Filterd Sound", bg=BLUE,
            command=lambda: self.play_audio(wind, plt, plt.filtered_sound, plt.output_image_pix3), width=22,
            font=(FONT_NAME, 15, 'bold')
        )
        play_filtered_sound_button.grid(column=self.right_column_start, row=9, columnspan=2)

    def filter_noise(self, plt: ThePlot):
        w = Tk()
        w.title('Noise Filter')
        w.config(padx=20, pady=20, bg=YELLOW)

        add_noise_lab = Label(
            w, text='Choose The \nCutoff Frequency:',
            bg=YELLOW, fg=GREEN,
            highlightthickness=0, font=(FONT_NAME, 20, 'bold')
        )
        add_noise_lab.pack()

        freq_in = Entry(w, font=(FONT_NAME, 15), width=20)
        freq_in.pack(pady=20)

        add_button = Button(
            w, text="Generate Sound", bg=GREEN,
            command=lambda: plt.filter_noise(w, int(freq_in.get())),
            font=(FONT_NAME, 15, 'bold')
        )
        add_button.pack()
        
    def add_noise(self, plt: ThePlot):
        w = Tk()
        w.title('Noise Adder')
        w.config(padx=20, pady=20, bg=YELLOW)

        add_noise_lab = Label(
            w, text='Write the noise frequency',
            bg=YELLOW, fg=GREEN,
            highlightthickness=0, font=(FONT_NAME, 20, 'bold')
        )
        add_noise_lab.pack()

        freq_in = Entry(w, font=(FONT_NAME, 15), width=20)
        freq_in.pack(pady=20)

        add_button = Button(
            w, text="Generate Sound", bg=GREEN,
            command=lambda: plt.add_noise(w, int(freq_in.get())),
            font=(FONT_NAME, 15, 'bold')
        )
        add_button.pack()

    def generate_sound_wave(self, plt: ThePlot):
        w = Tk()
        w.title('Sound Wave Generator')
        w.config(padx=20, pady=20, bg=YELLOW)

        add_wave_lab = Label(
            w, text='Add Component\n(f1,f2,f3, ...)\n(A1,A2,A3,...)',
            bg=YELLOW, fg=GREEN,
            highlightthickness=0, font=(FONT_NAME, 20, 'bold')
        )
        add_wave_lab.pack()

        freq_in = Entry(w, font=(FONT_NAME, 15), width=20)
        freq_in.pack(pady=20)

        amp_in = Entry(w, font=(FONT_NAME, 15), width=20)
        amp_in.pack(pady=20)

        generate_button = Button(
            w, text="Generate Sound", bg=GREEN,
            command=lambda: plt.generate_sound_wave(w, freq_in.get(), amp_in.get()),
            font=(FONT_NAME, 15, 'bold')
        )
        generate_button.pack()

    def plot_num_of_component(self, wind: Tk, plt: ThePlot):
        num_of_comp = int(self.plot_num_of_component_in.get())
        if not num_of_comp:
            return
        if num_of_comp > len(plt.important_amps):
            num_of_comp = len(plt.important_amps)

        plt.estimate_sound(wind, num_of_comp)

    def play_audio(self, wind: Tk, plt: ThePlot, sound, img):
        if img is plt.output_image_pix:
            canv_no = 0 
            row = 0
        elif img is plt.output_image_pix2:
            canv_no = 1
            row = plt.plot_row_span
        elif img is plt.output_image_pix3:
            canv_no = 2
            row = 2* plt.plot_row_span
        
        if sound.dtype != np.int16:
            sound = (sound * 32767 / np.max(np.abs(sound))).astype(np.int16)

        print('start')
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,  # 16-bit audio
                channels=1,     # Mono
                rate=44100,     # Sampling rate
                output=True)    # Output mode

        # Play the audio data
        repeat_num = int(10*plt.audio_duration)
        raw_image = np.copy(img) 
        for i in range(repeat_num):
            img = np.copy(raw_image)
            x = i * plt.image_length // repeat_num
            cv2.line(img, (x, 0), (x, 500), (255, 255, 0), 2)

            plt.put_plot(img, row=row, canv_no=canv_no)
            wind.update()

            start = i*len(sound)//repeat_num
            end = (i+1)*len(sound)//repeat_num
            stream.write(sound[start:end].tobytes())  # Convert NumPy array to bytes and play
            
        # Close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        print('stop')

    def choose_audio_file(self, wind: Tk, plt: ThePlot):
        file_path = filedialog.askopenfilename(
            title="Select the text file",
            filetypes=[("Text Files", "*.wav"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        
        plt.get_audio_data(file_path)
        plt.plot_sound()
        # cv2.imshow('plot sound', plt.output_image_pix)
        plt.put_plot(plt.output_image_pix)

    def ui_mode2_setup(self, wind: Tk, plt: ThePlot):
        self.mode = 2
        self.clear_window(wind)
        wind.config(padx=20, pady=20)
        wind.title('DFT Calculator')

        self.dft_siganl[0].append(SinosidalSignal(wind, 2., 1.))
        self.dft_siganl[0].append(SinosidalSignal(wind, 0.5, 4.))
        self.zoom_equation_slider(wind, plt)
        self.width_adj_slider.config(command=lambda value: plt.width_adj_changer(value, lambda: self.mode2_update(wind, plt)))
        self.length_adj_slider.config(command=lambda value: plt.length_adj_changer(value, lambda: self.mode2_update(wind, plt)))

        calc_dft_button = Button(wind, text="Calculate DFT", bg=GREEN, width=15,
                                 command=lambda: self.estimate_dft(wind, plt),
                                 font=(FONT_NAME, 20, 'bold'))
        calc_dft_button.grid(row=self.bellow_raw_start, column=2)

        back_to_main_button = Button(wind, text="Back To Main", bg=RED,
                                     command=self.back_to_main, width=15,
                                     font=(FONT_NAME, 20, 'bold'))
        back_to_main_button.grid(row=self.bellow_raw_start+2, column=2)

    def estimate_dft(self, wind: Tk, plt: ThePlot):
        file_path = filedialog.askopenfilename(
            title="Select the text file",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]  # Filter file types
        )

        if not file_path:
            return

        with open(file_path, 'r') as f:
            lines = f.readlines()

        y = list(map(float, lines[1].split()))
        sampling_rate = int(float(lines[3]))
        duration = int(float(lines[5]))

        T = 1 / sampling_rate
        L = len(y)
        Y = np.fft.fft(y)
        frequencies = np.fft.fftfreq(sampling_rate*duration, T)
        positive_freqs = frequencies[:L // 2]
        psitive_Y = Y[:L // 2]
        positive_magnitude = 2 * np.abs(psitive_Y) / L
        positive_phase = np.angle(psitive_Y)

        indecies = [i for i in range(len(Y[:L // 2])) if positive_magnitude[i] > 0.05]
        important_amps = [round(positive_magnitude[i], 3) for i in indecies]
        important_freq = [round(positive_freqs[i], 3) for i in indecies]
        important_phases = [round(positive_phase[i], 3) for i in indecies]

        self.dft_siganl = [[SinosidalSignal(wind, important_amps[i],
                                            important_freq[i],
                                            important_phases[i])
                                            for i in range(len(indecies))]]

        # plot_lines(self.dft_siganl, 200)
        # update_plot(wind)
        # print(important_amps)
        # print(important_freq)
        # print(important_phases)

        mplt.figure(figsize=(10, 6))
        mplt.stem(positive_freqs, positive_magnitude, 'b', markerfmt=" ", basefmt="-")
        mplt.title('DFT of the Signal')
        mplt.xlabel('Frequency (Hz)')
        mplt.ylabel('Magnitude')
        mplt.grid()
        mplt.figure(figsize=(10, 6))
        mplt.stem(positive_freqs, positive_phase, 'b', markerfmt=" ", basefmt="-")
        mplt.title('DFT of the Signal')
        mplt.xlabel('Frequency (Hz)')
        mplt.ylabel('Phase')
        mplt.grid()
        mplt.show()

        self.mode2_update(wind, plt)

    def mode2_update(self, wind: Tk, plt: ThePlot):
        plt.plot_dft(self.dft_siganl, 50)
        plt.draw_graphlines(plt.output_image_pix)
        plt.put_plot(plt.output_image_pix)

    def back_to_main(self):
        self.mode = 0
        self.plt.image_width = 500
        self.plt.plot_row_span = 15
        self.welcom_setup(self.wind, self.list_of_graphs, self.plt, self.mode1_update)

    def ui_mode1_setup(self, wind: Tk, list_of_graphs: list[list[SinosidalSignal]], plt: ThePlot, mode1_update):
        self.mode = 1
        self.dft_siganl = [[]]
        self.clear_window(wind)
        wind.config(padx=20, pady=20)

        # add frequency and change plot style buttons
        add_freq_but = Button(wind, text="Add Frequency", bg=GREEN,
                            command=lambda: self.add_freq(wind, list_of_graphs, plt), width=20,
                            font=(FONT_NAME, 15, 'bold'))
        self.change_plot_style_but = Button(wind, text="Points", bg=BLUE,
                                       command=lambda: self.change_plot_style(plt), width=8,
                                       font=(FONT_NAME, 15, 'bold'))
        
        add_freq_but.grid(column=self.right_column_start, row=0)
        
        self.change_plot_style_but.grid(column=self.right_column_start+1, row=0)

        # sampling rate modifier
        plt.sampling_rate_lab = Label(wind, text=f'Sampling Rate: {plt.sampling_rate}', bg=YELLOW, 
                            highlightthickness=0, font=(FONT_NAME, 12))
        sampling_rate_slider = Scale(wind, from_=1, to=150, length=500,
                                     orient="horizontal", command=plt.sampling_rate_changer)
        sampling_rate_slider.set(plt.sampling_rate)
        
        plt.sampling_rate_lab.grid(row=self.bellow_raw_start+1, column=0)
        sampling_rate_slider.grid(row=self.bellow_raw_start+1, column=1)

        self.zoom_equation_slider(wind, plt)

        # calcultae DFT button
        calc_dft_but = Button(wind, text="Plot DFT", bg=GREEN,
                            command=lambda: self.calc_dft(wind, plt, plt.sampling_rate, plt.duration, [i[1] for i in plt.output_data_set]), width=10,
                            font=(FONT_NAME, 15, 'bold'))
        calc_dft_but.grid(row=self.bellow_raw_start, column=2)

        rem_dft_but = Button(wind, text="Remove DFT", bg=RED,
                            command=self.rem_dft, width=10,
                            font=(FONT_NAME, 15, 'bold'))
        rem_dft_but.grid(row=self.bellow_raw_start+1, column=2)

        back_to_main_button = Button(wind, text="Back To Main", bg=RED,
                                     command=self.back_to_main, width=10,
                                     font=(FONT_NAME, 15, 'bold'))
        back_to_main_button.grid(row=self.bellow_raw_start+2, column=2)

        self.put_signal_slider(wind, list_of_graphs, plt)
        mode1_update()
    
    def zoom_equation_slider(self, wind: Tk, plt: ThePlot, ):
        # the equation label
        self.equation_lab = Label(wind, text='x(t)', bg=YELLOW, 
                            highlightthickness=0, anchor='w',
                            font=(FONT_NAME, 15))
        self.equation_lab.grid(row=self.bellow_raw_start, column=0, columnspan=self.right_column_start-1)

        # Zoom in X slider
        width_adj_lab = Label(wind, text='Zoom in X', bg=YELLOW, 
                              highlightthickness=0, font=(FONT_NAME, 12))
        self.width_adj_slider = Scale(wind, from_=50, to=500, length=500,
                                     orient="horizontal", command=lambda value: plt.width_adj_changer(value))
        self.width_adj_slider.set(plt.distance_to_pix_ratio_x)
        width_adj_lab.grid(row=self.bellow_raw_start+2, column=0)
        self.width_adj_slider.grid(row=self.bellow_raw_start+2, column=1)

        # Zoom in Y slider
        length_adj_lab = Label(wind, text='Zoom in Y', bg=YELLOW, 
                              highlightthickness=0, font=(FONT_NAME, 12))
        self.length_adj_slider = Scale(wind, from_=1, to=100, length=500,
                                     orient="horizontal", command=lambda value: plt.length_adj_changer(value))
        self.length_adj_slider.set(plt.distance_to_pix_ratio_y)
        length_adj_lab.grid(row=self.bellow_raw_start+3, column=0)
        self.length_adj_slider.grid(row=self.bellow_raw_start+3, column=1)

    def calc_dft(self, wind: Tk, plt: ThePlot, sampling_rate, duration, signal):
        print(sampling_rate)
        print(duration)

        y = [i[1] for i in plt.output_data_set]
        
        T = 1 / sampling_rate
        L = len(y)
        Y = np.fft.fft(y)
        frequencies = np.fft.fftfreq(round(sampling_rate*duration), T)
        positive_freqs = frequencies[:L // 2]
        psitive_Y = Y[:L // 2]
        positive_magnitude = 2 * np.abs(psitive_Y) / L
        positive_phase = np.angle(psitive_Y)

        indecies = [i for i in range(len(Y[:L // 2])) if positive_magnitude[i] > 0.05]
        plt.important_amps = [round(positive_magnitude[i], 3) for i in indecies]
        plt.important_freq = [round(positive_freqs[i], 3) for i in indecies]
        plt.important_phases = [round(positive_phase[i], 3) for i in indecies]

        if self.mode == 1:
            self.dft_siganl = [[SinosidalSignal(wind, plt.important_amps[i],
                                                plt.important_freq[i],
                                                plt.important_phases[i])
                                                for i in range(len(indecies))]]
            print(self.dft_siganl)
        elif self.mode == 3:
            sorted_data = sorted(zip(plt.important_amps, plt.important_freq, plt.important_phases), 
                                key=lambda x: x[0],  # Sort by the amplitude (first element)
                                reverse=True)
            plt.important_amps, plt.important_freq, plt.important_phases = map(list, zip(*sorted_data))
            print(plt.important_freq[:5])
            print(plt.important_amps[:5])

        print(len(plt.important_amps))

        mplt.figure(figsize=(10, 6))
        mplt.stem(positive_freqs, positive_magnitude, 'b', markerfmt=" ", basefmt="-")
        mplt.title('DFT of the Signal')
        mplt.xlabel('Frequency (Hz)')
        mplt.ylabel('Magnitude')
        mplt.grid()
        # mplt.figure(figsize=(10, 6))
        # mplt.stem(positive_freqs, positive_phase, 'b', markerfmt=" ", basefmt="-")
        # mplt.title('DFT of the Signal')
        # mplt.xlabel('Frequency (Hz)')
        # mplt.ylabel('Phase')
        # mplt.grid()
        mplt.show()
        
    def rem_dft(self):
        self.dft_siganl = [[]]
    
    def add_freq(self, wind: Tk, list_of_graphs: list[list[SinosidalSignal]], plt: ThePlot):
        def done_callback(wind: Tk, w2: Tk):
            new_amp = float(A_in.get())
            new_freq = float(f_in.get())
            if theta_in.get() == '':
                new_theta = 0
            else:
                new_theta = float(theta_in.get())
            w2.destroy()

            # add frequency to the list, but first check if it's aleardy there
            is_not_changed = 1
            for sino in list_of_graphs[0]:
                if sino.frequency == new_freq:
                    is_not_changed = 0
                    final_amp = (new_amp**2 + sino.amplitude**2 + 2*new_amp*sino.amplitude*np.cos(sino.phase-new_theta))**0.5
                    if final_amp == 0: #if amplitude is zero, remove the component
                        list_of_graphs[0].remove(sino)
                        break
                    final_theta = np.arctan((new_amp*np.sin(new_theta)+sino.amplitude*np.sin(sino.phase))/(new_amp*np.cos(new_theta)+sino.amplitude*np.cos(sino.phase)))
                    index = list_of_graphs[0].index(sino)
                    new_sino = SinosidalSignal(wind, final_amp, new_freq, final_theta)
                    list_of_graphs[0][index] = new_sino
                    break
            
            if is_not_changed:
                list_of_graphs[0].append(SinosidalSignal(wind, new_amp, new_freq, new_theta))
            # sort the signal by ascending frequencies
            list_of_graphs[0].sort(key=lambda signal: signal.frequency)
            self.put_signal_slider(wind, list_of_graphs, plt)
        
        w2 = Tk()
        w2.title('Add Frequency Component')
        w2.config(padx=50, pady=20, bg=YELLOW)
        equation_lab = Label(w2, text="A * cos(2πf *t + theta)", font=(FONT_NAME, 18, 'bold italic'),
                            fg=GREEN, bg=YELLOW, highlightthickness=0, pady=30)
        equation_lab.grid(column=0, row=0, columnspan=2)
        A_lab = Label(w2, text="A", font=(FONT_NAME, 15), pady=10,
                    bg=YELLOW, highlightthickness=0)
        f_lab = Label(w2, text="f", font=(FONT_NAME, 15), pady=10,
                    bg=YELLOW, highlightthickness=0)
        theta_lab = Label(w2, text="theta", font=(FONT_NAME, 15), pady=10,
                        bg=YELLOW, highlightthickness=0)
        A_in = Entry(w2, font=(FONT_NAME, 18))
        f_in = Entry(w2, font=(FONT_NAME, 18))
        theta_in = Entry(w2, font=(FONT_NAME, 18))

        A_lab.grid(column=0, row=1)
        f_lab.grid(column=0, row=2)
        theta_lab.grid(column=0, row=3)
        A_in.grid(column=1, row=1)
        f_in.grid(column=1, row=2)
        theta_in.grid(column=1, row=3)

        button = Button(w2, text="Confirm", bg=YELLOW, fg="#008800",
                        font=(FONT_NAME, 25, 'bold'), command=lambda: done_callback(wind, w2))
        button.grid(column=0, row=4, columnspan=2)

    def update_ui(self, wind: Tk, list_of_graphs: list[list[SinosidalSignal]], plt: ThePlot):
        self.write_the_equation(list_of_graphs)

    def destroy_widget_by_position(self, root: Tk, row, column):
        # Get all widgets placed in the grid
        widgets = root.grid_slaves(row=row, column=column)
        if widgets: # Destroy the widget if it exists
            widgets[0].destroy()  # Assuming only one widget is in this grid cell

    def write_the_equation(self, list_of_graphs: list[list[SinosidalSignal]]):
        equation_text = 'x(t) = '
        for complex_sino in list_of_graphs:
            threshhold = 75
            for sino in complex_sino:
                equation_text += f'{sino.amplitude:.2f}*cos(2π*{sino.frequency:.2f}*t'
                if not sino.phase == 0:
                    equation_text += f' +{sino.phase:.2f})'
                else:
                    equation_text += ')'
                
                if len(equation_text) > threshhold:
                    equation_text += '\n'
                    threshhold += 75
                equation_text += ' + '
        equation_text = equation_text[:-3]

        self.equation_lab.config(text=equation_text)
        
    def change_plot_style(self, plt: ThePlot):
        plt.use_line = not plt.use_line
        if plt.use_line:
            self.change_plot_style_but.config(text='Line') 
        else:
            self.change_plot_style_but.config(text='Points')

    def delete_button(self, wind: Tk,plt: ThePlot,sino: SinosidalSignal, list_of_graphs: list[list[SinosidalSignal]]):
        sino.amp_slider.destroy()
        sino.freq_slider.destroy()
        sino.theta_slider.destroy()
        sino.delete_but.destroy()
        sino.dash_lab.destroy()
        list_of_graphs[0].remove(sino)
        self.put_signal_slider(wind, list_of_graphs, plt)

    def put_signal_slider(self, wind: Tk, list_of_graphs: list[list[SinosidalSignal]], plt: ThePlot):
        starting_row = 1
        for complex_sino in list_of_graphs:
            for sino in complex_sino:
                sino.delete_but.grid(column=self.right_column_start+1, row=starting_row, rowspan=3)
                sino.delete_but.config(command=lambda s=sino: self.delete_button(wind, plt, s, list_of_graphs))

                sino.amp_slider.grid(column=self.right_column_start, row=starting_row)
                starting_row += 1
                sino.freq_slider.grid(column=self.right_column_start, row=starting_row)
                starting_row += 1
                sino.theta_slider.grid(column=self.right_column_start, row=starting_row)
                starting_row += 1
                sino.dash_lab.grid(column=self.right_column_start, row=starting_row, columnspan=2)
                starting_row += 1
                
