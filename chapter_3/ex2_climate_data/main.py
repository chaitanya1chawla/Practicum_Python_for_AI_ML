import json
import os
from collections import OrderedDict


f=open("raw_data.json", 'w')
data_by_country = {}

for filename in os.listdir("./__files/tas_data"):
    
    with open("./__files/tas_data/" + filename, 'r') as readfile:
        data = []
        for line in readfile:
            data.append(float(line))
        data_by_country[filename[:3]] = data


json.dump(data_by_country, f)
f.close()

g=open("aggregated_data.json", "w")
output_by_country = {}

for item, val in data_by_country.items():
    summ = sum(val)
    length = len(val)
    minn = min(val)
    maxx = max(val)
    avgg= summ/length
    output_by_country[item] = { "t_avg": avgg, "t_max":maxx , "t_min": minn }

output_by_country = OrderedDict(sorted(output_by_country.items()))

json.dump(output_by_country, g)
g.close()

print(type(data_by_country))

print( str ( data_by_country["DNK"])) 
print( str (output_by_country["DNK"]))
print( str (output_by_country["BRA"]))
print( str (output_by_country["CAN"]))
print( str (output_by_country["CIV"]))
print( str (output_by_country["PAK"]))
