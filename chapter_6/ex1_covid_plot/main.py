import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


df_1 = pd.read_csv("https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-BY.tsv", sep='\t', header=0)
df_2 = pd.read_csv("https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-NW.tsv", sep='\t', header=0)
df_3 = pd.read_csv("https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-SN.tsv", sep='\t', header=0)
df_4 = pd.read_csv("https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-TH.tsv", sep='\t', header=0)

df_de = pd.read_csv("https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/main/data/de-states/de-state-DE-total.tsv", sep='\t', header=0)


fig = plt.figure()
axes = fig.add_subplot()

plt.title("7-day incidence/Mio of Covid-cases")
plt.xticks(rotation=70)
axes.plot(df_1['Date'], df_1["Cases_Last_Week_Per_Million"],color='r',label="BY")
axes.plot(df_2['Date'], df_2["Cases_Last_Week_Per_Million"],color='b',label="NW")
axes.plot(df_3['Date'], df_3["Cases_Last_Week_Per_Million"],color='g',label="SN")
axes.plot(df_4['Date'], df_4["Cases_Last_Week_Per_Million"],color='m',label="TH")
axes.annotate(f"\nMaximum n ={df_3['Cases_Last_Week_Per_Million'].max()} in \n SN @ {df_3.iloc[df_3['Cases_Last_Week_Per_Million'].idxmax(),0]}",("2022-03-22",df_3["Cases_Last_Week_Per_Million"].max()),arrowprops=dict(headwidth =6,headlength=6) )
#,cd = {'width' =1, 'headwidth' = 1, 'headlength'=1}
#df_3["Cases_Last_Week_Per_Million"]
plt.xlabel("Date")
plt.ylabel("n/(week - million)")
axes.set_xticks(["2020-02-29", "2020-06-10", "2020-09-20", "2020-12-31", "2021-04-10", "2021-07-20", "2021-11-02", "2022-03-10", "2022-06-28", "2022-09-25", "2022-12-29"])
plt.yscale("log")


inset_ax = axes.inset_axes([0.55,0.1,0.4,0.2] )


inset_ax.set_yscale("log")
inset_ax.set_title("Incidence in Whole Germany")

inset_ax.plot(df_de['Date'], df_de["Cases_Last_Week_Per_Million"],color='r')
inset_ax.set_xticks(["2020-02-29", "2021-08-02", "2022-12-29"])
plt.legend(loc = "upper left")
plt.savefig("plot.pdf", dpi=300, bbox_inches='tight')

# draw maximum, tick x axes, inset graph