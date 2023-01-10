import spidev
import RPi.GPIO as io
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib import cycler
import time as t

IPython_default = plt.rcParams.copy()

colors = cycler('color',
                ['#06161a', '#5682b5', '#a42f42',
                 '#edaa47', '#88BB44', '#FFBBBB'])
plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
       axisbelow=True, grid=True, prop_cycle=colors)
plt.rc('grid', color='w', linestyle='solid')
plt.rc('xtick', direction='out', color='gray')
plt.rc('ytick', direction='out', color='gray')
plt.rc('patch', edgecolor='#E6E6E6')
plt.rc('lines', linewidth=0.5)


def acquire_samples(bus_rate, total_samples, channel=1, vref=3.3, dual=False):
    """
    Acquire samples using SPI interface with ADC board. Supports dual and single channel modes.
    In dual channel mode, channel selection is not required. Sends three bytes to MCP3202 depending on channel selection:
        - First byte always 1
        - Second byte is different for each channel:
            -> Channel 1: 0x10000000 (2<<6)
            -> Channel 2: 0x11000000 (3<<6)
        - Third byte always 0
    Converts the input data to voltage using reference voltage (vref). Output contains 3 bytes and relevant data is sent over
    2 bytes; last four bits of 2nd byte is combined with the 3rd byte for full 12 bit sample read.

    Input variables:
        bus_rate: determines SPI bus rate, supposed max 36MHz
        total_samples: sets the total number of samples to be read.
        channel: selects the channel
        vref: reference voltage
        dual: sets dual channel acquire
    
    Outputs:
        channel 1 and 2 voltages if dual mode, or single voltage set
        sample_rate: the rate at which samples were acquired
    """
    voltages_1 = []
    voltages_2 = []
    count = 0

    dev = spidev.SpiDev()

    dev.open(0, 0)
    dev.max_speed_hz = bus_rate  # recommended max RPi bus speed 36MHz
    start_time = t.time()

    if dual:
        while count<total_samples:
            output = dev.xfer2([1, 2<<6, 0]) # first channel input
            voltages_1.append(((output[1] & 0x0f)<<8) + output[2])
    
            output = dev.xfer2([1, 3<<6, 0]) # second channel input
            voltages_2.append(((output[1] & 0x0f)<<8) + output[2]) 
            count+=1
    else:
        while count<total_samples:
            output = dev.xfer2([1, channel+1 << 6, 0]) # channel selection
            voltages_1.append(((output[1] & 0x0f)<<8) + output[2])
            count+=1
        
    end_time = t.time()
    sample_rate = total_samples/(end_time-start_time)
    dev.close()

    # convert the raw data w.r.t reference voltage w resolution of 12 bits
    if dual:
        return np.array(voltages_1)*(vref/4095), np.array(voltages_2)*(vref/4095), sample_rate
    else:
        return np.array(voltages_1)**(vref/4095), sample_rate 

def FFT(voltage, sample_rate, total_samples, crop=5):

    fft_voltage1 = np.abs(np.fft.fft(voltage))
    fft_freq1 = np.fft.fftfreq(int(total_samples), d=1/sample_rate)
    # stop the horizontal lines
    fft_voltage = np.fft.fftshift(fft_voltage1)
    fft_freq = np.fft.fftshift(fft_freq1)
    # crop first mega spike
    fft_voltage_cropped = fft_voltage1[crop:]
    fundamental = fft_freq1[(crop+fft_voltage_cropped.argmax())]

    return fft_voltage, fft_freq, fundamental

def plotting(voltages1, voltages2, sample_rate, SD_name, dual=True):
    """
    Plotting function to display the voltage vs time and the accompanying FFT. Works for dual and single channel modes.
    Calculates the time series for a set of samples and the fundamental frequency of the signal.

    Inputs:
        voltages1: channel 1 set of voltages
        voltages2: channel 2 set of voltages
        sample_rate: the rate at which the samples were acquired at
        dual: sets between dual channel or single channel mode
    
    Outputs:
        Subplot with two or four figures (dependant on dual mode) for voltage vs. time and FFT of signal.
    """
    
    total_samples = len(voltages1)
    sample_vals = np.arange(0, total_samples, 1)
    times = [i/sample_rate for i in sample_vals]

    crop = 5
    freq_range = int(total_samples)

    if dual:
        fog, axs = plt.subplots(2, 2, figsize = (16,12))
        axs[0][0].plot(times, voltages1, linewidth=0.5, label=f'Data acquired at {sample_rate:.0f} Samples/s')
        axs[0][0].set(xlabel='Time / s', ylabel = 'Voltage / v', title='ADC conversion CH1')
        axs[0][0].legend()

        fft_voltage, fft_freq, fundamental = FFT(voltages1, sample_rate, total_samples)

        # axs[1].plot(fft_freq[5:freq_range], fft_voltage[5:freq_range])
        axs[0][1].loglog(fft_freq[crop:freq_range], fft_voltage[crop:freq_range], linewidth=0.5, 
                    label=f'Loglog FFT with fundamental at {fundamental:0.0f} Hz')
        # axs[1].set_yscale('log')
        # axs[1].set_xscale('log')
        axs[0][1].set(xlabel = 'Frequency / Hz', ylabel = 'FFT absolute magnitude', title = 'FFT of ADC CH1')
        axs[0][1].legend()

        axs[1][0].plot(times, voltages2, linewidth=0.5, label=f'Data acquired at {sample_rate:.0f} Samples/s')
        axs[1][0].set(xlabel='Time / s', ylabel = 'Voltage / v', title='ADC conversion CH2')
        axs[1][0].legend()

        fft_voltage, fft_freq, fundamental = FFT(voltages2, sample_rate, total_samples)

        # axs[1].plot(fft_freq[5:freq_range], fft_voltage[5:freq_range])
        axs[1][1].loglog(fft_freq[crop:freq_range], fft_voltage[crop:freq_range], linewidth=0.5, 
                    label=f'Loglog FFT with fundamental at {fundamental:0.0f} Hz')
        # axs[1].set_yscale('log')
        # axs[1].set_xscale('log')
        axs[1][1].set(xlabel = 'Frequency / Hz', ylabel = 'FFT absolute magnitude', title = 'FFT of ADC CH2')
        axs[1][1].legend()

        plt.tight_layout()
        plt.savefig(f'Figures/{SD_name}.pdf')
        
    else:
        fog, axs = plt.subplots(1, 2, figsize = (16,7))
        axs[0].plot(times, voltages1, linewidth=0.5, label=f'Data acquired at {sample_rate:.0f} Samples/s')
        axs[0].set(xlabel='Time / s', ylabel = 'Voltage / v', title='ADC conversion')
        axs[0].legend()

        fft_voltage, fft_freq, fundamental = FFT(voltages1, sample_rate, total_samples)

        # axs[1].plot(fft_freq[5:freq_range], fft_voltage[5:freq_range])
        axs[1].loglog(fft_freq[crop:freq_range], fft_voltage[crop:freq_range], linewidth=0.5, 
                    label=f'Loglog FFT with fundamental at {fundamental:0.0f} Hz')
        # axs[1].set_yscale('log')
        # axs[1].set_xscale('log')
        axs[1].set(xlabel = 'Frequency / Hz', ylabel = 'FFT absolute magnitude', title = 'FFT of ADC')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(f'Figures/{SD_name}.pdf')


total_samples = 100000
bus_rate = 7000000 # 7MHz seems to be max bus rate, above there are glitches
# voltages, sample_rate = acquire_samples(bus_rate=bus_rate, total_samples=total_samples)
voltages1, voltages2, sample_rate = acquire_samples(bus_rate=bus_rate, total_samples=total_samples, dual=True)

v1 = np.array(voltages1)
v2 = np.array(voltages2)
sr = np.array(sample_rate)
#np.savetxt(f'/home/pi/Desktop/Figs and data/SPI/In Lab measurements/Data/PSD off at {sample_rate}.txt', (v1, v2)) 

print(f'Sample rate is {sample_rate} Hz')
plotting(voltages1, voltages2, sample_rate, SD_name='Slow')
