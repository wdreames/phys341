import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.special import gamma

def fpcalc(MS_treatment,MS_error,dof_treatment,dof_error,alpha):
    
    fSN = float(MS_treatment/MS_error)
    print ("F Signal/Noise = %0.2f " % fSN)

    fdist = stats.f(dof_treatment,dof_error)
    fhigh = fdist.ppf(1-alpha)
    print ("Critical F-value = %0.2f" % (fhigh))

    if (fSN > 1):
        pvalue = (1-fdist.cdf(fSN))
    else:
        pvalue = fdist.cdf(fSN)
        
    print ("Pvalue = %0.3f" % (pvalue))

    return (fSN,pvalue)

def tukey_calc(xbar,MS_error,n,dof_error,alpha):
    
    # Tukey's Procedure
    #
    # Step 1:  Calculate the standard error = sqrt(MS_error/n)

    std_error = np.sqrt(MS_error/n)
    print ("Standard Error = %0.3f" % std_error)

    # Step 2:  Order the means from smallest to largest

    print ("Original Xbar = ",xbar)
    xbar_orig = xbar.copy()
    xbar.sort()
    print ("Sorted Xbar = ",xbar)

    # Step 3:  Get the expected number of error bars (sem) between largest and smallest, for the number
    # of means to be compared (a).  For this, we need the studentized range table values q(alpha,dof_error,k) for k=2..a.
    # 
    # http://www.real-statistics.com/statistics-tables/studentized-range-q-table/
    # For this problem, dof_error = 15, alpha = 0.05, k = 2,3,4,5
    #q = np.array([3.014,3.673,4.076,4.367])

    n_means=len(xbar)
    q = q_tukey(n_means,dof_error,alpha)
    print (q)

    # Step 4:  Calculate W = q(k=5)*std_error

    W = q[n_means-2]*std_error

    print ("W = %0.2f" % W)

    # Step 5:  Make pair-wise comparisons

    print (list(xbar_orig))

    print ("i j xbar1 xbar2 Diff Diff_comp Result")
    for i in range(len(xbar)):
        for j in range(len(xbar)):
            if (xbar[j] > xbar[i]):
                diff = (xbar[j] - xbar[i])/std_error
                diff_comp = q[j-i-1]
                if (diff>diff_comp):
                    Result = "Yes"
                else:
                    Result = "No"
                iorig = list(xbar_orig).index(xbar[i])+1
                jorig = list(xbar_orig).index(xbar[j])+1
                print ("%0.0f %0.0f %0.1f %0.1f %0.3f  %0.3f %s" % (iorig,jorig,xbar[i],xbar[j],diff,diff_comp,Result))
                
def linear_regression(sumx,sumy,sumxy,sumx2,n):
    b0 = (sumx2*sumy-sumx*sumxy)/(n*sumx2-sumx**2)
    b1 = (n*sumxy-sumx*sumy)/(n*sumx2-sumx**2)
    return b0,b1

def sigma_regression_summary_stats(sumx,sumy,sumxy,sumx2,sumy2,n,b0,b1):
    t1 = n*b0*b0
    t2 = 2*b0*b1*sumx
    t3 = b1**2*sumx2
    t4 = -2.0*b0*sumy
    t5 = -2.0*b1*sumxy
    t6 = sumy2
    v_error = n - 2

    sigma = np.sqrt(1.0/v_error*(t1+t2+t3+t4+t5+t6))

    return float(sigma)

def q_tukey(k, v, alpha):

    qarray = []
    for k in range(2,k+1):
        print ("q_tukey: calculating k = ",k)

        dq = 0.003
        qd = np.array(np.arange(0.000,6.000,dq))

        f = []
        fs = []
        found = False

        for i in range(len(qd)):
            q = qd[i]
            prefactor = np.sqrt(2.0*np.pi)*k*(k-1)*v**(v/2.0)/(gamma(v/2.0)*2**(v/2.0-1))
            npts = 100

            xh = 6.0
            xl = 0.0
            dx = (xh-xl)/npts

            x = np.arange(xl,xh,dx)
            x = x.reshape(1,-1)
    
            ul = -6.0
            uh = 6.0
            du = (uh-ul)/npts
            
            u = np.arange(ul,uh,du)
            u = u.reshape(-1,1)
            
            phi_u = stats.norm.pdf(u)
            phi_ux = stats.norm.pdf(u-q*x)
            Phi_u = stats.norm.cdf(u)
            Phi_ux = stats.norm.cdf(u-q*x)
            phi_x = stats.norm.pdf(np.sqrt(v)*x)
            
            integrand = x**v*phi_x*phi_u*phi_ux*(Phi_u-Phi_ux)**(k-2)*du*dx
            
            #print ("U integral matrix")
            #print(u,phi_u,phi_ux,Phi_u,Phi_ux,phi_x,x**v,integrand)
        
            sumux = integrand.sum()
            
            #print ("U sum")
            #print(sumux)
            
            #print ("X sum * prefactor")
            #print (sumux*prefactor)

            f.append(sumux*prefactor)
            
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
        plt.scatter(q_critical,(1-alpha))
    
        f = np.array(f)
        fs = np.array(fs)
        plt.plot(qd,fs)

    qarray=np.array(qarray)
    return qarray
