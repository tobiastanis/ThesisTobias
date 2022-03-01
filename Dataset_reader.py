"""
This file imports the data of LUMIO and the Moon over an epoch of Modified Julian Time 59091.50000 to 61325.00000,
meaning from date 2020-08-30 12:00:00.000 to 2026-10-12 00:00:00.000.

Two important times at which the initial state of LUMIO are considered are at 21-03-2024 and 18-04-2024, since no
stationkeeping is performed from then to a certain amount of days

See below for function description (from line 39)
"""
import numpy as np
import pandas as pd
import csv
print("Running [Dataset_reader.py]")

# Opening csv datasets
LUMIO_datacsv = open('LUMIO_states.csv')
Moon_datacsv = open('Moon_states.csv')
# Reading
csvreader_LUMIO = csv.reader(LUMIO_datacsv)
csvreader_Moon = csv.reader(Moon_datacsv)
# Extruding
header_LUMIO = next(csvreader_LUMIO)
header_Moon = next(csvreader_Moon)
rows_LUMIO = []
rows_Moon = []
for row in csvreader_LUMIO:
    rows_LUMIO.append(row)
for row in csvreader_Moon:
    rows_Moon.append(row)
data_LUMIO = np.array(rows_LUMIO).astype(float)
data_Moon = np.array(rows_Moon).astype(float)

t0_data_mjd = data_Moon[0, 0]
tend_data_mjd = data_Moon[(len(data_Moon) - 1), 0]

LUMIO_dataframe = pd.DataFrame(
    {'MJD': data_LUMIO[:, 0], 'ET': data_LUMIO[:, 1], 'x': data_LUMIO[:, 2], 'y': data_LUMIO[:, 3],
     'z': data_LUMIO[:, 4], 'vx': data_LUMIO[:, 5], 'vy': data_LUMIO[:, 6], 'vz': data_LUMIO[:, 7]})
Moon_dataframe = pd.DataFrame(
    {'MJD': data_Moon[:, 0], 'ET': data_Moon[:, 1], 'x': data_Moon[:, 2], 'y': data_Moon[:, 3],
     'z': data_Moon[:, 4], 'vx': data_Moon[:, 5], 'vy': data_Moon[:, 6], 'vz': data_Moon[:, 7]})

"""
Function file which can be used to obtain data over a time interval defined in Modified Julian Time from 59091.50000 to
61325.00000. The data set has time epochs of 0.25 MJD. However, numbers in between can be used as well, but notice that 
for t0, data will always be rounded up and for tend the data will always be rounded down.

Four functions. Data functions output are in dataframe format and state parameters in [km]. state functions are in 
np.array format and state parameters are in [m]
"""


def data_moon(t0, tend):
    return Moon_dataframe.loc[(Moon_dataframe['MJD'] >= t0) & (Moon_dataframe['MJD'] <= tend)]


def data_lumio(t0, tend):
    return LUMIO_dataframe.loc[(LUMIO_dataframe['MJD'] >= t0) & (LUMIO_dataframe['MJD'] <= tend)]


def state_moon(t0, tend):
    return np.asarray(Moon_dataframe.loc[(Moon_dataframe['MJD'] >= t0) & (Moon_dataframe['MJD'] <= tend)])[:, 2: 8]*10**3


def state_lumio(t0,tend):
    return np.asarray(LUMIO_dataframe.loc[(LUMIO_dataframe['MJD'] >= t0) & (LUMIO_dataframe['MJD'] <= tend)])[:, 2: 8]*10**3


def simulation_start_epoch(t0):
    data = np.asarray(LUMIO_dataframe.loc[(LUMIO_dataframe['MJD'] == t0)])[0]
    return np.asscalar(data[1])

print("[Dataset_reader.py] ran successfully \n")
"""
Bij de MJD die je get, pak ook de ephemeris time, zodat die gelijk als input voor somilation start wordt gedefined.
daarmee kan dat de initial states van alle celestial bodies die meeworden genomen gedifined worden en dan werkt het 
hopleijk
"""