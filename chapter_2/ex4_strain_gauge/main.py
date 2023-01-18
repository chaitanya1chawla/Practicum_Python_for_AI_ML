import math


flag=0
f=open("strain_gauge_processed.csv", 'w')

with open("__files/strain_gauge_rosette.csv", 'r') as readfile:

    for line in readfile:

        if(flag==0):
            flag=1
            f.write("R2_1-m/m;R2_2-m/m;R2_3-m/m;e_1;e_2\n")
            continue

        x=line.split("\t")
        a=float(x[0])
        b=float(x[1])
        c=float(x[2])

        eq=(a-b)**2 + (b-c)**2
        e1=(a+c+math.sqrt(2*eq))/2
        e2=(a+c-math.sqrt(2*eq))/2

        f.write("{:.9f};{:.9f};{:.9f};{:.9f};{:0.9f}\n".format(a,b,c,e1,e2))

f.close()



