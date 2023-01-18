import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import dump

def plotting(income,lim,rating,cards,r2):
    fig = plt.figure()
    axes = fig.add_subplot(211)
    ax = fig.add_subplot(212, sharex=axes)

    plotrange= range(0, 10000,50)
    axes.plot(plotrange, rating, color='b', label="Rating")
    axes.plot(plotrange, lim, color='y',label="Limit")
    axes.plot(plotrange, income, color='g',label="Income")
    axes.plot(plotrange, cards, color='r',label="Cards")
    axes.set_ylabel('Value of the coefficient')
    
    axes.set_xlim(0.1,10000)
    axes.set_ylim(-10,10)
    axes.set_xscale('log')
    ax.set_xlabel("alpha")
    ax.set_ylabel('R^2')
    ax.plot(plotrange, r2)

    axes.legend(loc = "lower left")
    plt.savefig("plot.pdf", dpi=300, bbox_inches='tight')

def alpha_param(x,y): 
    r2 = []
    income = []
    lim = []
    rating = []
    cards = []
    x_train,x_test,y_train,y_test = train_test_split(x[["Income", "Cards", "Rating","Limit"]],y, shuffle=True,test_size=0.2)
    
    for i in range(0, 10000, 50):
        model = Lasso(alpha = i,fit_intercept = True)
        #print(x_train)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(model.coef_)
        income.append( model.coef_[0])
        cards.append(model.coef_[1])
        rating.append(model.coef_[2])
        lim.append(model.coef_[3])

        r2.append(r2_score( y_test, y_pred ))

        if(i==100):
            dump(model, f'model_1.joblib')
        elif(i==2000):
            dump(model, f'model_2.joblib')
        elif(i==5000):
            dump(model, f'model_3.joblib')

    return income,lim,rating,cards,r2

if __name__ == "__main__":
    cred = pd.read_csv("__files/credit.csv", index_col=False, sep=',')
    X = cred.drop(['Unnamed: 0','Age','Education','Gender','Student','Married','Ethnicity','Balance'],axis= 1)
    #print(X)
    inc, lim, rat, car, r2 = alpha_param(X, cred["Balance"])
    plotting(inc,lim,rat,car,r2)