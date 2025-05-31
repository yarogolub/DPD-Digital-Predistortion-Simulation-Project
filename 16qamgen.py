import numpy as np
import math
import matplotlib.pyplot as plt

#Starting off by generating the symbols for our 16-QAM Signal

#Assumptions:
#Modeling an LTE Signal
#Single Carrier
#BW: 10MHz
#Bit Rate: 50 Mbits/s
#Symbol Rate = 12.5 Msymbols/s = Bit Rate / Bits/Symbol = 50Mbits/s/(4bits/symbol)
#Sampling rate: 15.36 MHz (standard for LTE 10 MHz)



# Parameters
num_symbols = 10000  # number of QAM symbols

# 16-QAM constellation (real and imag levels)
levels = np.array([-3, -1, 1, 3])

# 1-d array of the constellation levels.
constellation = np.array([x + 1j*y for x in levels for y in levels])

# Averaging the constellation magnitude to 1
constellation  = constellation / np.sqrt((np.mean(np.abs(constellation)**2)))

#mapping 4 bit strings to constellation

bit_to_symbol = {
    '0000': (-3, -3), '0001': (-3, -1), '0011': (-3, 1), '0010': (-3, 3),
    '0110': (-1, 3),  '0111': (-1, 1),  '0101': (-1, -1), '0100': (-1, -3),
    '1100': (1, -3),  '1101': (1, -1),  '1111': (1, 1),   '1110': (1, 3),
    '1010': (3, 3),   '1011': (3, 1),   '1001': (3, -1),  '1000': (3, -3),
}

def ascii_to_16qam(text):
    symbols = []
    for char in text:
        #convert char to 8-bit binary string
        bits = format(ord(char), '08b')
        #split into two 4-bit groups
        for i in range(0,8,4):
            four_bits = bits[i:i+4] #slices bits from i to i+3 (not including i+4)
            I, Q = bit_to_symbol[four_bits]
            symbols.append(complex(I,Q))

    

    #s(t) = I(t) + jQ(t)
    symbols = np.array(symbols)
    symbols /= np.sqrt(np.mean(np.abs(symbols)**2)) 

    return symbols
#Write what message you want the wave contain in 'text'
text = "Hello world"

qam_symbols = ascii_to_16qam(text)
#Symbols have now been converted to 16QAM Constellation points

# Now we want to use our symbols to generate a time domain signal. We aren't taking into account jitter noise (from the clock) or phase noise (from the osc), need to double check if these both contribute or just one of them does
#Single carrier

symbol_rate = 12.5e6 
symbol_duration = 1 / symbol_rate  # seconds per symbol
sample_factor = 8 #samples per symbol
sample_rate = symbol_rate*sample_factor  #how many samples we take per each symbol

sec_div = 10**9 #dividing 1 second into 9 billion, 1 billion nanoseconds in 1 second

total_time = 1 #seconds
num_samples = int(sample_rate*total_time)


#1 second split into 1 nanosecond increments
#time = np.linspace(0, total_time, num_samples, endpoint= False)


#Creating our Time domain signal made of points in the IQ constellation
td_signal = np.repeat(qam_symbols, sample_factor) 


def add_gaus_noise(signal):
    n_variance = 0.01
    noise_I = np.random.normal(0, np.sqrt(n_variance), td_signal.shape)
    noise_Q = np.random.normal(0, np.sqrt(n_variance), td_signal.shape)

    noise = noise_I + 1j * noise_Q

    return signal + noise

td_signal_noise = add_gaus_noise(td_signal)

""" Only plotting with small sample sizes, dont want this program to slow down to much.
# Plot real/imag parts
plt.plot(np.imag(td_signal), label='Clean Signal (Imag)')
plt.plot(np.imag(td_signal_noise), label='Noisy Signal (Imag)', alpha=0.7)
#plt.plot(np.real(td_signal), label='Clean Signal (Real)')
#plt.plot(np.real(td_signal_noise), label='Noisy Signal (Real)', alpha=0.7)

plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Real Component of Signal: Clean vs Noisy')
plt.legend()
plt.grid(True)
plt.show()

print(td_signal)
"""


