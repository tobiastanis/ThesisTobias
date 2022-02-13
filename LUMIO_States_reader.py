import numpy as np
import pandas as pd
import Input as I
import csv
# Extruding LUMIO data
LUMIO_States = open('LUMIO_states.csv')
csvreader_LUMIO = csv.reader(LUMIO_States)
header_LUMIO = next(csvreader_LUMIO)
rows_LUMIO = []
for row in csvreader_LUMIO:
    rows_LUMIO.append(row)
data_LUMIO = np.array(rows_LUMIO)
LUMIO_dataframe = pd.DataFrame({'MJD':data_LUMIO[:,0], 'ET':data_LUMIO[:,1], 'x':data_LUMIO[:, 2], 'y':data_LUMIO[:, 3],
                                'z':data_LUMIO[:,4], 'vx':data_LUMIO[:,5],'vy':data_LUMIO[:,6],'vz':data_LUMIO[:,7]})
# Timespan data. MJD must be between 59091.50000 and 61325.00000, steps of 0.25 and always five decimals behind .
t_start = '60390.00000'
t_end = float(t_start) + I.simulation_time
t_end = str(t_end)
#t_end = '60405.00000'
# Obtaining data for that timespawn
LUMIOdata_timespan = LUMIO_dataframe.loc[(LUMIO_dataframe['MJD'] >= t_start) & (LUMIO_dataframe['MJD'] <= t_end)]
LUMIOdata_timespan = np.asarray(LUMIOdata_timespan)
state_LUMIO = LUMIOdata_timespan[:, 2:7].astype(float)               # [x, y, z, vx, vy, vz] km and km/s

# Extruding Moon data
Moon_States = open('Moon_states.csv')
csvreader_Moon = csv.reader(Moon_States)
header_Moon = next(csvreader_Moon)
rows_Moon = []
for row in csvreader_Moon:
    rows_Moon.append(row)
data_Moon = np.array(rows_Moon)
Moon_dataframe = pd.DataFrame({'MJD':data_Moon[:,0],'ET':data_Moon[:, 1] ,'x':data_Moon[:, 2], 'y':data_Moon[:, 3],
                               'z':data_Moon[:, 4], 'vx':data_Moon[:, 5], 'vy':data_Moon[:, 6], 'vz':data_Moon[:, 7]})
Moondata_timespan = Moon_dataframe.loc[(Moon_dataframe['MJD'] >= t_start) & (Moon_dataframe['MJD'] <= t_end)]
Moondata_timespan = np.asarray(Moondata_timespan)
state_Moon = Moondata_timespan[:, 2:7].astype(float)


