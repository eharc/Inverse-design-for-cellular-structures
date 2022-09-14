import matplotlib.pyplot as plt
import statistics as st

def plot(Xr,Xf):
    ER_1=[]
    
    for i in range(len(Xf)):
        er1=abs((Xf[:,0][i]-Xr[:,0][i])/Xr[:,0][i])
        ER_1.append(er1)
        
    Mean=st.mean(ER_1)


    plt.figure(3)
    plt.plot(Xf[:,0],color='g',alpha=0.5)
    plt.plot(Xr[:,0],color='b',alpha=0.5)
    plt.title('Parameter: P1')
    plt.legend()
    plt.show()
    

        
    plt.figure(5)
    plt.plot(ER_1)
    plt.title("Relative deviation for P1 ")
    plt.show()

    
    print(Mean)
    
    return(Mean)
