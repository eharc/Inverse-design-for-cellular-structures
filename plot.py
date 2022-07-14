# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:12:30 2022

@author: eharc
"""

import matplotlib.pyplot as plt
import statistics as st

def plot(Xr,Xf):
    ER_1=[]
    ER_2=[]
    ER_3=[]
    ER_4=[]
    ER_5=[]
    ER_6=[]
    ER_7=[]
    
    for i in range(len(Xf)):
        er1=abs((Xf[:,0][i]-Xr[:,0][i])/Xr[:,0][i])
        er2=abs((Xf[:,1][i]-Xr[:,1][i])/Xr[:,1][i])
        er3=abs((Xf[:,2][i]-Xr[:,2][i])/Xr[:,2][i])
        er4=abs((Xf[:,3][i]-Xr[:,3][i])/Xr[:,3][i])
        er5=abs((Xf[:,4][i]-Xr[:,4][i])/Xr[:,4][i])
        er6=abs((Xf[:,5][i]-Xr[:,5][i])/Xr[:,5][i])
        er7=abs((Xf[:,6][i]-Xr[:,6][i])/Xr[:,6][i])
        
        ER_1.append(er1)
        ER_2.append(er2)
        ER_3.append(er3)
        ER_4.append(er4)
        ER_5.append(er5)
        ER_6.append(er6)
        ER_7.append(er7)

        
    plt.figure(10)
    plt.plot(ER_1)
    plt.title('Relative deviation: a')
    plt.show()
    
    plt.figure(11)
    plt.plot(ER_2)
    plt.title('Relative deviation: R1')
    plt.show()
    
    plt.figure(12)
    plt.plot(ER_3)
    plt.title('Relative deviation: R2')
    plt.show()
    
    plt.figure(13)
    plt.plot(ER_4)
    plt.title('Relative deviation: wb')
    plt.show()
    
    plt.figure(14)
    plt.plot(ER_5)
    plt.title('Relative deviation: lb')
    plt.show()
    
    plt.figure(15)
    plt.plot(ER_6)
    plt.title('Relative deviation: wg')
    plt.show()
    
    plt.figure(16)
    plt.plot(ER_7)
    plt.title('Relative deviation: lg')
    plt.show()
    
    Mean_ER_1=st.mean(ER_1)
    Mean_ER_2=st.mean(ER_2)
    Mean_ER_3=st.mean(ER_3)
    Mean_ER_4=st.mean(ER_4)
    Mean_ER_5=st.mean(ER_5)
    Mean_ER_6=st.mean(ER_6)
    Mean_ER_7=st.mean(ER_7)
    
    plt.figure(3)
    plt.plot(Xf[:,0],color='r',label='Synthetic')
    plt.plot(Xr[:,0],color='b',label='Original')
    plt.title('Parameter: a')
    plt.legend()
    plt.show()
    
    plt.figure(4)
    plt.plot(Xr[:,1],color='b',label='Original')
    plt.plot(Xf[:,1],color='r',label='Synthetic')
    plt.title('Parameter: R1')
    plt.legend()
    plt.show()
    
    plt.figure(5)
    plt.plot(Xf[:,2],color='r',label='Synthetic')
    plt.plot(Xr[:,2],color='b',label='Original')
    plt.title('Parameter: R2')
    plt.legend()
    plt.show()
    
    plt.figure(6)
    plt.plot(Xr[:,3],color='b',label='Original')
    plt.plot(Xf[:,3],color='r',label='Synthetic')
    plt.title('Parameter: wb')
    plt.legend()
    plt.show()
    
    plt.figure(7)
    plt.plot(Xf[:,4],color='r',label='Synthetic')
    plt.plot(Xr[:,4],color='b',label='Original')
    plt.title('Parameter: lb')
    plt.legend()
    plt.show()
    
    plt.figure(8)
    plt.plot(Xr[:,5],color='b',label='Original')
    plt.plot(Xf[:,5],color='r')
    plt.title('Parameter: wg')
    plt.legend()
    plt.show()
    
    plt.figure(9)
    plt.plot(Xf[:,6],color='r',label='Synthetic')
    plt.plot(Xr[:,6],color='b',label='Original')
    plt.title('Parameter: lg')
    plt.legend()
    plt.show()
    
    Mean=(Mean_ER_1+Mean_ER_2+Mean_ER_3+Mean_ER_4+Mean_ER_5+Mean_ER_6+Mean_ER_7)/7
    print(Mean,Mean_ER_1,Mean_ER_2,Mean_ER_3,Mean_ER_4,Mean_ER_5,Mean_ER_6,Mean_ER_7)
    return(Mean,Mean_ER_1,Mean_ER_2,Mean_ER_3,Mean_ER_4,Mean_ER_5,Mean_ER_6,Mean_ER_7)


        
        

    