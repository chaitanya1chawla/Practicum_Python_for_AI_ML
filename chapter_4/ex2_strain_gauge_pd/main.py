import numpy as np
import pandas as pd
import math
import os

pd.set_option('display.float_format', lambda x: '%.9f' % x)

gauge = pd.read_csv("__files/strain_gauge_rosette.csv", sep='\t')

gauge["e_1"] = (gauge["R2_1-m/m"] + gauge["R2_3-m/m"] + np.sqrt(2*( (gauge["R2_1-m/m"]-gauge["R2_2-m/m"])**2 + (gauge["R2_2-m/m"]-gauge["R2_3-m/m"])**2) ) )/2 
gauge["e_2"] = (gauge["R2_1-m/m"] + gauge["R2_3-m/m"] - np.sqrt(2*( (gauge["R2_1-m/m"]-gauge["R2_2-m/m"])**2 + (gauge["R2_2-m/m"]-gauge["R2_3-m/m"])**2) ) )/2 

gauge = gauge.round(decimals=9)

gauge.to_csv('strain_gauge_processed.csv',float_format = '%.9f', sep=';',index=False)


