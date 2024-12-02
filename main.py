import cv2

import numpy as np
import time
from sinosoidal import SinosidalSignal, ThePlot

from ui_script import UI, Tk
from ui_script import *

####    MAKE A BUTTON THAT CALCULATE THE DFT
####    FIRST, SHOW THE SAME SIGNAL, THEN PROCEED TO DFT
####    MAKE SURE THE FORMULA YOU USE IS CORRECT use matlab for this
####    Add sound player, and show the sound wave
####    Generate sound using sinosoidal
####    Calculate and show the dft
####    Most 50, 100, 1000 signal components
####    Add noise to signal
####    ADD DFT FOR ALL SIGNALS
####    ADD FILTERS
####    call it a project ☕.

def main_update(w: Tk, ui: UI, list_of_graphs, plt: ThePlot):
    plt.output_image_pix = np.zeros((plt.image_width, plt.image_length, 3), np.uint8)
    if not ui.dft_siganl == [[]]:
        plt.plot_dft(ui.dft_siganl, 100)

    if plt.use_line:
        plt.plot_lines(list_of_graphs, plt.sampling_rate)
    else:
        plt.plot_points(list_of_graphs, plt.sampling_rate)
    

    if ui.mode == 1:
        plt.put_plot(plt.output_image_pix)
        ui.update_ui(w, list_of_graphs, plt)
        w.after(33, lambda: main_update(w, ui, list_of_graphs, plt))
    else:
        plt.canv.destroy()

def main(win):
    plt = ThePlot(sampling_rate=50, image_length=1000)
    ui = UI()
    
    complex_sino1 = []
    list_of_graphs = [complex_sino1]
    #                                (amp, frq, phase)
    complex_sino1.append(SinosidalSignal(win, 2., 1.))
    complex_sino1.append(SinosidalSignal(win, 0.5, 4.))
    # complex_sino1.append(SinosidalSignal(wind, 1., 3, 1.56))
    # complex_sino1.append(SinosidalSignal(wind, 1., 5, 1.56))
    # complex_sino1.append(SinosidalSignal(wind, 1., 6, -1.56))
    # complex_sino1.append(SinosidalSignal(wind, 0.2, 25, 0))

    # plt.plot_lines(list_of_graphs, plt.sampling_rate)
    # cv2.imshow("test", frequency_image)
    # cv2.waitKey(0)
    ui.welcom_setup(win, list_of_graphs, plt, lambda: main_update(win, ui, list_of_graphs, plt))

if __name__ == "__main__":
    start_time = time.time()
    wind = Tk()
    
    main(wind)
    wind.mainloop()
    
    print(f"execution duration: {time.time() - start_time}")