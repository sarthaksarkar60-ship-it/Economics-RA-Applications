import numpy as np
import scipy.stats as sps
from math import comb
def bern_simul(p):
    b = np.random.rand()
    if b<p:
        return 1
    else:
        return 0
def bayesian_simul(a,n):
    b = 0
    c = np.zeros((a,n))
    while b<a:

        t_m = np.zeros((3,n))
        for i in range(t_m.shape[0]):
            for j in range(t_m.shape[1]):
                t_m[i,j] = bern_simul(0.6)
        for k in range(t_m.shape[1]):
            leader = 0
            swing = 0
            veto = 0
            s = np.random.randn()
            p =np.random.rand()
            beli = np.array(np.random.rand(3))
            for l in range(n):
                for m in range(n-l):
                    if l+m>= n/2:
                        leader += comb(n,l+m)*((beli[1])**l)*(beli[2]**m)*(beli[3]**(n-(l+m)))*(t_m[k,0]+t_m[k,1]-(p*t_m[k,2])-s)
                        swing += (comb(n,l+m)*((beli[1])**l)*(beli[2]**m)*(beli[3]**(n-(l+m)))*(-t_m[k,0]+t_m[k,1]-(p*t_m[k,2])+s))
                        veto+= comb(n,l+m)*((beli[1])**l)*(beli[2]**m)*(beli[3]**(n-(l+m)))*(t_m[k,0]-t_m[k,1]-(p*t_m[k,2]))
                    else:
                        leader += comb(n, l + m) * ((beli[1]) ** l) * (beli[2] ** m) * (beli[3] ** (n - (l + m))) * (
                                    - t_m[k, 0] + t_m[k, 1] - (p * t_m[k, 2]))
                        swing += comb(n, l + m) * ((beli[1]) ** l) * (beli[2] ** m) * (beli[3] ** (n - (l + m))) * (
                                    -t_m[k, 0] + t_m[k, 1] - (p * t_m[k, 2]))
                        veto += comb(n, l + m) * ((beli[1]) ** l) * (beli[2] ** m) * (beli[3] ** (n - (l + m))) * (
                                    t_m[k, 0] - t_m[k, 1] - (p * t_m[k, 2]))
