import numpy as np
import pandas as pd
import urllib
import json
from urllib.error import HTTPError
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def  download_bme280(id, dates):

	flag = 0
	dates=sorted(dates)
	result=pd.DataFrame()

	for date in dates:
		try:
			df = pd.read_csv(f"https://archive.sensor.community/{date}/{date}_bme280_sensor_{str(id)}.csv", sep=';', header=0)
			
			if(date == dates[0]):
				result = df
				#print(result)
			else:
				result = pd.concat([result, df], ignore_index=True)
				#print(result)
				


		except urllib.error.HTTPError :
			flag=flag+1
			pass
	
	if(flag==len(dates)):
		raise FileNotFoundError

	return result

def conv_ts(sensordf):
	df=sensordf
	df['timestamp'] = pd.to_datetime( sensordf['timestamp'] )
	return df

def select_time(sensordf, after, before):
	df=conv_ts(sensordf)
	dt_bef = datetime.strptime(before, '%Y-%m-%d %H:%M')
	dt_aft = datetime.strptime(after, '%Y-%m-%d %H:%M')
	return df[np.logical_and(df['timestamp'] > dt_aft, df['timestamp'] < dt_bef)]

def filter_df(sensor_df):
	rd = {"min_T" :0.0, "min_p":0, "min_T_id":0, "min_p_id":0}
	rd["min_T"] = sensor_df['temperature'].min()
	rd["min_T_id"] = sensor_df.iloc[sensor_df['temperature'].idxmin(),0]
	rd["min_p"] = sensor_df['pressure'].min()
	rd["min_p_id"] = sensor_df.iloc[sensor_df['pressure'].idxmin(),0]
	return rd

extrema_dict = {"min_T" :0.0, "min_p":0, "min_T_id":0, "min_p_id":0}
flag=0

#11036 is missing!
for id in [10881,11077,11114]:
	sens_df = download_bme280(id, ["2022-1-30", "2022-01-31", "2022-02-01", "2022-01-28", "2022-01-29", "2022-02-02", "2022-02-03"] )
	sens_df = select_time(sens_df, "2022-01-28 05:13", "2022-02-03 12:31")
	res_dict = filter_df(sens_df)

	if(flag==0):
		extrema_dict["min_T"] = res_dict["min_T"] 
		extrema_dict["min_T_id"] = int(res_dict["min_T_id"])
		extrema_dict["min_p"] = res_dict["min_p"]
		extrema_dict["min_p_id"] = int(res_dict['min_p_id'])
		flag=1
	else:
		if(extrema_dict["min_T"] > res_dict["min_T"] ):
			extrema_dict["min_T"] = res_dict["min_T"]
			extrema_dict["min_T_id"] = int(res_dict["min_T_id"])
		if(extrema_dict["min_p"] > res_dict["min_p"]):
			extrema_dict["min_p"] = res_dict["min_p"]
			extrema_dict["min_p_id"] = int(res_dict['min_p_id'])

with open("extrema.json", "w") as outfile:
    json.dump(extrema_dict, outfile)



pt1 = download_bme280(10881,["2022-01-01"])
pt2 = download_bme280(11036,["2022-01-01"])

plt.title("Temperature-Curve on 2022-01-01 for 2 sensors")
plt.plot(pt1.index,pt1["temperature"],color='g',label="10881")
plt.plot(pt2.index,pt2["temperature"],color='r',label="11036")
plt.xlabel(f"Time from {datetime.strptime(pt1['timestamp'].min(), '%Y-%m-%dT%H:%M:%S').time()} to {datetime.strptime(pt1['timestamp'].max(), '%Y-%m-%dT%H:%M:%S').time()}")
plt.ylabel("Temperature [°C]")

plt.legend(loc="upper right")
plt.savefig("sensors.pdf")



#res = pd.DataFrame()
res = download_bme280(10881, ['2021-11-17', '2021-11-16', '2021-11-15', '2021-11-20', '2021-11-18', '2021-11-19'] )
ress = conv_ts(res)
ret = filter_df(res)

#print(ress)
resss = select_time(ress, "2021-11-15 12:13","2021-11-18 14:15") 
#print(resss["timestamp"])
#print(resss)
