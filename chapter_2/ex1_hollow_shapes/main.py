import sys 


print("Type \"diamond\" or \"triangle\" - ")
str=input()

if(str=="diamond"):
    print("Enter width of diamond - \n")
    rad = int(input())
    
    for i in range(rad):
        print(" "*(rad-i) + "*" + " "*(2*i) + "*" )
    for i in range(rad+1):
        print(" "*(i) + "*" + " "*(2*(rad-i)) + "*" )
        
if(str=="triangle"):
    print("Enter width of triangle - \n")
    rad = int(input())
    
    for i in range(rad):
        if(i==0):
            print(" "*rad+"*")
            continue
        print(" "*(rad-i) + "*" + " "*(2*i-1) + "*" )
    print("* "*(rad+1))    
