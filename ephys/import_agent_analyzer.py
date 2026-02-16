import numpy as np
import matplotlib.pyplot as plt

def importagentanlyzer(suffix = "2024-08-22T14_16_06"): #This is just the default the code had
      user_input = input("Enter file suffix: ")
      if user_input:
            suffix = user_input
      print(f"Using suffix: {suffix}")  # based on user's input
      return suffix

suffix = importagentanlyzer()



      #%% Metadata
dt = {'names': ('time', 'acq_clk_hz', 'block_read_sz', 'block_write_sz'),
'formats': ('datetime64[us]', 'u4', 'u4', 'u4')}
meta = np.genfromtxt('start-time_' + suffix + '.csv', delimiter=',', dtype=dt)
print(f"Recording was started at {meta['time']} GMT")

#%% Analog Inputs dictionary and then lists within dictionary made
analog_input = {}
analog_input['time'] = np.fromfile('analog-clock_' + suffix + '.raw', dtype=np.uint64) / meta['acq_clk_hz']
analog_input['O2'] = np.fromfile(f'O2_{suffix}.raw', dtype=np.float32) * 10  #multiply by 10 for O2
analog_input['CO2'] = np.fromfile(f'CO2_{suffix}.raw', dtype=np.float32)
analog_input['SEV'] = np.fromfile(f'SEV_{suffix}.raw', dtype=np.float32)
analog_input['ISO'] = np.fromfile(f'ISO_{suffix}.raw', dtype=np.float32)

plt.close('all')

#plotting O2

def plotO2():
    plt.figure()
    plt.title("%O2 vs Time (s)")
    plt.plot(analog_input['time'], analog_input['O2'])
    plt.xlabel("time (sec)")
    plt.ylabel("% O2") #goes 0-100% 1 volt = 10% O2
    plt.legend(['O2'])
    plt.show()
plotO2()


#plotting CO2
def plotCO2():
    plt.figure()
    plt.title("%CO2 vs Time (s)")
    plt.plot(analog_input['time'], analog_input['CO2'])
    plt.xlabel("time (sec)")
    plt.ylabel("% CO2") #goes 0-10% based on volts 1 volt = 1%
    plt.legend(['CO2'])
plotCO2()

def plotAnesthetic():
    plt.figure()
    anesthetic = "SEV" #Can change this to whatever one
    plt.title(f"{anesthetic}% vs Time (s)")
    plt.plot(analog_input['time'], analog_input[f'{anesthetic}'])
    plt.xlabel("time (sec)")
    plt.ylabel(f"% {anesthetic}" ) # goes 0-10% based on volts 1 volt = 1%
    plt.legend([f"{anesthetic}"])
plotAnesthetic()

#%% Hardware FIFO buffer use
dt = {'names': ('clock', 'bytes', 'percent'),
      'formats': ('u8', 'u4', 'f8')}
memory_use = np.genfromtxt('memory-use_' + suffix + '.csv', delimiter=',', dtype=dt)

plt.figure()
plt.plot(memory_use['clock'] / meta['acq_clk_hz'], memory_use['percent'])
plt.xlabel("time (sec)")
plt.ylabel("FIFO used (%)")

plt.show()