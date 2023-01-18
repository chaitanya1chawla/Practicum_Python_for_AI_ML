import re
import numpy as np

def get_props(text):
    li = re.findall(r"property\s*=\s*[-+]*\d+\.\d+", text)
    arr =[]
    for i in li:
        arr.append(float(re.findall("[-+]*\d+\.\d+",i)[0]))
    print(arr)
    return arr

def check_email(text):
    val = re.fullmatch(r"\S+\@\S+\.\S+",text)
    return bool(val) 

def replace_name(text):
    val = re.findall(r"(Mr.|Mrs.)\s(\"*\w+\-*'*\w+\"*)\s(\"*\w+\-*'*\w+\"*)",text)
    print(val)
    for i in val:
        text=re.sub(f"{i[0]} {i[1]} {i[2]}",f"{i[2]}, {i[1]}",text)
    print(text)
    return text

def get_atoms(path):
    data = []
    #arr = []
    arr = np.array([[0,0,0]])
    #arr = np.delete(arr,[0,1,2])
    with open("./__files/qe_example.log", 'r') as readfile:
        for idx,line in enumerate(readfile):
            if(idx>153):
                li = re.findall(r"(?:\d+\.\d+)", line)  #\s*(?:\d+\.\d+)\s*(?:\d+\.\d+)
                dat = re.findall(r"[A-Z][A-Z]",line)
                #print(dat[0])
                #print(li)
                if(len(li)==3) and (len(dat)!=0):
                    print(li)
                    np.array([[float(li[0]),float(li[1]),float(li[2])]])
                    #arr.append(li)
                    arr = np.append(arr, np.array([[float(li[0]),float(li[1]),float(li[2])]]), axis =0)
                    data.append(dat[0])
        arr = arr[1:,:]
        print("arr = ",arr, data)    
    return(data,arr)

if __name__ == "__main__":
    print("Enter text")

    get_props("blue property=1.453, red property= -4.58, teal property =-1.678")
    #print(check_mail("afa@sda.com"))
    #replace_name("Mr. \"Kisho're\" Kumar fdfd Mrs. Sita Kumar")
    #get_atoms("chapter_10\ex9_regex\__files\qe_example.log")
