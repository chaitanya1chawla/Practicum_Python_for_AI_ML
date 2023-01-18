import math as mt



for i in range(316):
    if(i%5!=0):
        continue
    val=i/100
    si=mt.sin(val)
    if(si>0):
        if(si>-0.1 and si<0.1):
            print("sin({:.2f}) =  {:.2f}, near null".format(val, si))
        else:
            print("sin({:.2f}) =  {:.2f}".format(val,si))
    else:
        if(si>-0.1 and si<0.1):
            print("sin({:.2f}) = {:.2f}, near null".format(val, si))
        else:
            print("sin({:.2f}) = {:.2f}".format(val,si))
    
