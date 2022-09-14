import matplotlib.pyplot as plt
import statistics as st

def plot(Xr,Xf):
    ER_1=[]
    ER_2=[]
    
    for i in range(len(Xf)):
        er1=abs((Xf[:,0][i]-Xr[:,0][i])/Xr[:,0][i])
        er2=abs((Xf[:,1][i]-Xr[:,1][i])/Xr[:,1][i])
        
        ER_1.append(er1)
        ER_2.append(er2)
    
    Mean_ER_1=st.mean(ER_1)
    Mean_ER_2=st.mean(ER_2)
    Mean=(Mean_ER_1+Mean_ER_2)/2

    plt.figure(3)
    plt.plot(Xf[:,0],color='r',label="Synthetic",alpha=0.7)
    plt.plot(Xr[:,0],color='b',label="Original",alpha=0.5)
    plt.title('Parameter: R2')
    plt.legend()
    plt.show()
    
    plt.figure(4)
    plt.plot(Xf[:,1],color='r',label="Synthetic",alpha=0.7)
    plt.plot(Xr[:,1],color='b',label="Original",alpha=0.5)
    plt.title('Parameter: wb')
    plt.legend()
    plt.show()
    

        
    plt.figure(5)
    plt.plot(ER_1)
    plt.title("Relative deviation for P1 ")
    plt.show()
    
    plt.figure(6)
    plt.plot(ER_2)
    plt.title("Relative deviation for P2")
    plt.show()
    
    print(Mean,Mean_ER_1,Mean_ER_2)
    
    return(Mean,Mean_ER_1,Mean_ER_2)
