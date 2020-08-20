import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.special import gamma

def q_tukey(k, v, alpha):

    qarray = []
    for k in range(2,k+1):
        print ("q_tukey: calculating k = ",k)

        dq = 0.004
        qd = np.array(np.arange(0.000,6.000,dq))

        f = []
        fs = []
        found = False

        for i in range(len(qd)):
            q = qd[i]
            prefactor = np.sqrt(2.0*np.pi)*k*(k-1)*v**(v/2.0)/(gamma(v/2.0)*2**(v/2.0-1))
            npts = 30

            xh = 5.0
            xl = 0.0
            dx = (xh-xl)/npts

            # Loop over x to integrate
            xsum = 0.0
            for x in np.arange(xl,xh,dx):
                phi_x = stats.norm.pdf(np.sqrt(v)*x)
    
                ul = -5.0
                uh = 5.0
                du = (uh-ul)/npts
            
                u = np.arange(ul,uh,du)
                phi_u = stats.norm.pdf(u)
                phi_ux = stats.norm.pdf(u-q*x)
                Phi_u = stats.norm.cdf(u)
                Phi_ux = stats.norm.cdf(u-q*x)
                integrand = phi_u*phi_ux*(Phi_u-Phi_ux)**(k-2)*du
                #print(u,phi_u,phi_ux,Phi_u,Phi_ux,integrand)
        
                sumu = integrand.sum()
        
                #print(x,sumu)
        
                integrand2 = x**v*phi_x*sumu*dx
                xsum += integrand2

            f.append(xsum*prefactor)
            if (i>0):
                fs.append(f[i]*dq+fs[i-1])
            else:
                fs.append(f[i]*dq)

            if (fs[i]>(1-alpha) and not(found)):
                q_critical = qd[i-1] + ((1-alpha)-fs[i-1])*(qd[i]-qd[i-1])/(fs[i]-fs[i-1])
                print ("q_critical = ",q_critical)
                found = True
        
            #print(q,f[i],fs[i])
        qarray.append(q_critical)
    
        f = np.array(f)
        fs = np.array(fs)
        #plt.plot(qd,fs)

    qarray=np.array(qarray)
    return qarray
