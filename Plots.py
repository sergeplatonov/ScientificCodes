
# coding: utf-8

# In[61]:


get_ipython().magic(u'matplotlib inline')

from __future__ import division
#plt.axis([xmin, xmax, ymin, ymax])
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
#plt.plot([1,2,3,5],[0,3,4,9],'ro')
#plt.ylabel('some numbers')
b=17.5

def f(x):
     a=np.sinc(b*np.sin(x))
     return a

X=np.arange(-1,1,0.001)

plt.plot(X,f(X)**2)



plt.savefig('C:\Users\freudenfeld\Desktop\test\a', format='pdf')




# In[78]:


get_ipython().magic(u'matplotlib inline')

from __future__ import division
#plt.axis([xmin, xmax, ymin, ymax])
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
#plt.plot([1,2,3,5],[0,3,4,9],'ro')
#plt.ylabel('some numbers')

lambdaF=44.8E-9
d=250.0E-9


b=d*np.pi/lambdaF
print ('b =',b)

def f1(x):
     a=np.sinc(2*b*np.sin(x))
     return a

def f2(x):
    a=np.sinc(b*np.sin(x+0.1))
    return a


X=np.arange(-0.5,0.5,0.001)

print('length X=',len(X))

X1=X*180.0/np.pi


plt.plot(X1,f1(X)**2,'b')
plt.plot(X1,f2(X)**2,'g')
plt.plot(X1,(f1(X)+f2(X))**2,'r')



plt.xlabel("Alpha (°)")
plt.ylabel("I (alpha)")

plt.savefig('C:/Users/freudenfeld/Desktop/test/a', format='pdf')




# In[3]:

get_ipython().magic(u'matplotlib inline')

from __future__ import division
#plt.axis([xmin, xmax, ymin, ymax])
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
#plt.plot([1,2,3,5],[0,3,4,9],'ro')
#plt.ylabel('some numbers')


z=1j
e=1.6E-19
L=4.6E-6
hbar=1.054E-34
lambdaF=44.8E-9
d=250.0E-9
k1=2*np.pi/lambdaF


#b=d*np.pi/lambdaF
#print ('b =',b)

"""

#ohne Magnetfeld, mit Detektor bei Winkel t



def dprime(t):
    
    return k1*np.sin(t)
    
def k(a):
    
    return np.pi*a/d

def Fplus(a,t):
    
    return np.sqrt(d)*np.sinc((d/2.0)*(k(a)+dprime(t)))
                              
def Fminus(a,t):
    
    return np.sqrt(d)*np.sinc((d/2.0)*(k(a)-dprime(t)))
                              
def f(a,t):    
    return (1/2*z)*(np.exp(z*np.pi*a/2)*Fminus(a,t)-np.exp(-z*np.pi*a/2)*Fplus(a,t))

print (z)


range=np.arange(-(3/18)*np.pi,(3/18)*np.pi,0.001)
angle=range*180/np.pi

plt.plot(angle,np.absolute(f(1,range)), 'b')
plt.plot(angle,np.absolute(f(2,range)), 'g')
#plt.plot(angle,np.absolute(f(3,range)), 'y')
plt.plot(angle,np.absolute(f(1,range)+f(2,range)+f(3,range)), 'r')




#mit Magnetfeld und Winkel t:

B=0.00

def dprime(t):
    return k1*np.sin(t)-(e*B*L)/(2*hbar) 
    
def k(a):
    
    return np.pi*a/d

def Fplus(a,t):
    
    return np.sqrt(d)*np.sinc((d/2.0)*(k(a)+dprime(t)))
                              
def Fminus(a,t):
    
    return np.sqrt(d)*np.sinc((d/2.0)*(k(a)-dprime(t)))
                              
def f(a,t):    
    return (1/2*z)*(np.exp(z*np.pi*a/2)*Fminus(a,t)-np.exp(-z*np.pi*a/2)*Fplus(a,t))

print (z)
    
range=np.arange(-(3/18)*np.pi,(3/18)*np.pi,0.001)
angle=range*180/np.pi

#plt.plot(angle,np.absolute(f(1,range)), 'b')
#plt.plot(angle,np.absolute(f(2,range)), 'g')
#plt.plot(angle,np.absolute(f(3,range)), 'y')
plt.plot(angle,np.absolute(f(1,range)+f(2,range)), 'k')
plt.plot(angle,np.absolute(f(1,range)+f(2,range)+f(3,range)), 'r')





#mit Magnetfeld und Detektor bei Winkel t=0:



def dprime(B):
    return (e*B*L)/(2*hbar) 
    
def k(a):
    
    return (np.pi*a/d)+a*1.4E7

def Fplus(a,B):
    
    return np.sqrt(d)*np.sinc((d/2.0)*(k(a)+dprime(B)))
                              
def Fminus(a,B):
    
    return np.sqrt(d)*np.sinc((d/2.0)*(k(a)-dprime(B)))
                              
def f(a,B):    
    return (1/2*z)*(np.exp(z*np.pi*a/2)*Fminus(a,B)-np.exp(-z*np.pi*a/2)*Fplus(a,B))
    

#def i(a,B):
    
    #return sum(f(a,B))

    
print ('pi/d =',np.pi/d)

print (z)
    
range=np.arange(-0.04,0.04,0.0001)
angle=range*180/np.pi


#plt.plot(range,np.absolute(f(1,range)), 'r')
#plt.plot(range,np.absolute(f(2,range)), 'b')
#plt.plot(range,np.absolute(f(3,range)), 'y')
plt.plot(range,np.absolute(f(1,range)+f(2,range)), 'r')
#plt.plot(range,np.absolute(f(1,range)+f(2,range)+f(3,range)), 'k')


#Normiert:
#plt.plot(range,np.absolute(f(1,range))/max(np.absolute(f(1,range))), 'k')
#plt.plot(range,np.absolute(f(1,range)+f(2,range))/max(np.absolute(f(1,range)+f(2,range))), 'r')


"""

#mit Magnetfeld und Detektor bei Winkel t=0, variierendes d:
#fe: für ungerade a=1,3,5...
#fo: für gerade a=2,4,6...

# Berücksichtigung unterschiedlicher Widerstände für verschiedene Plateaus -> I~V/R_0 *a ~ a

def d1(a):
    return 250E-9

def dprime(B):
    return (e*B*L)/(2*hbar)
    
def k(a):    
    return (np.pi*a/d1(a))

def Fplus(a,B):
    
    return np.sqrt(d1(a))*np.sinc((d1(a)/2.0)*(k(a)+dprime(B)))
                              
def Fminus(a,B):
    
    return np.sqrt(d1(a))*np.sinc((d1(a)/2.0)*(k(a)-dprime(B)))

#Mit Strom proportional zum Plateau (fe/o~a)

def fo(a,B):    
    return (1/2*z)*a*(np.exp(z*np.pi*a/2)*Fminus(a,B)-np.exp(-z*np.pi*a/2)*Fplus(a,B))
    
def fe(a,B):    
    return (1/2)*a*(np.exp(z*np.pi*a/2)*Fminus(a,B)+np.exp(-z*np.pi*a/2)*Fplus(a,B))

#def i(a,B):
    
    #return sum(f(a,B))

 
#print ('pi/d =',np.pi/d)

#print ((e*L)/(2*hbar))
    
#print (z)
    
range=np.arange(-0.05,0.05,0.0001)
angle=range*180/np.pi


plt.plot(range,np.absolute(fe(3,range)), 'r')
plt.plot(range,np.absolute(fo(4,range)), 'b')
plt.plot(range,np.absolute(fe(1,range)+fo(2,range)+fe(3,range)+fo(4,range)+fe(5,range)+fo(6,range)+fe(7,range)), 'y')
#plt.plot(range,np.absolute(f(1,range)+f(2,range)), 'r')
#plt.plot(range,np.absolute(fe(4,range)), 'k')
#plt.plot(range,np.absolute(f(2,range)+f(3,range)), 'k')

#Normiert:
#plt.plot(range,np.absolute(f(1,range))/max(np.absolute(f(1,range))), 'k')
#plt.plot(range,np.absolute(f(1,range)+f(2,range))/max(np.absolute(f(1,range)+f(2,range))), 'r')


print (2*np.pi*hbar/(2*e**2))





# In[50]:

#QPC Eigenfunktionen - Annahme: QPC Breite "d" fest und hängt nicht vom Plateau ab

get_ipython().magic(u'matplotlib inline')

from __future__ import division
#plt.axis([xmin, xmax, ymin, ymax])
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
#plt.plot([1,2,3,5],[0,3,4,9],'ro')
#plt.ylabel('some numbers')


z=1j
e=1.6E-19
L=4.6E-6
hbar=1.054E-34
lambdaF=44.8E-9
d=250.0E-9
k1=2*np.pi/lambdaF


def d1(a):
    return a*30E-9

#Quantenzahl a ungerade (a=1,3,5,7...) - Cosinusprofil

def wfo(a,y):
    return np.sqrt(2/d)*np.cos((a*np.pi)*y/d)



#Quantenzahl a gerade (a=2,4,6,8...) - Sinusprofil

def wfe(a,y):
    return np.sqrt(2/d)*np.sin((a*np.pi)*y/d)




range=np.arange(-125E-9,125E-9,1E-11)
angle=range*180/np.pi


plt.plot(range,wfo(1,range),'r')
plt.plot(range,wfe(2,range),'k')
plt.plot(range,wfo(3,range),'g')
plt.plot(range,wfe(4,range),'b')
plt.xlabel('QPC constriction coordinate')


# In[84]:


get_ipython().magic(u'matplotlib inline')

from __future__ import division
#plt.axis([xmin, xmax, ymin, ymax])
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
#plt.plot([1,2,3,5],[0,3,4,9],'ro')
#plt.ylabel('some numbers')
b=17.5

exp1 = np.fromfile("C:\\Users\\freudenfeld\\Desktop\\1.dat",dtype=float,count=-1, sep=" ")
#exp1 = np.reshape(exp1, (10,-1)
#exp2 = exp1.transpose()
#plt.plot(exp1[:,0],exp1[:,1], 'r',mew=0.3, ms=3)
#plt.xlabel("Lense voltage (V)")
#plt.ylabel("conductance (e^2/h)")
#plt.xlim(-2,0.5)
#plt.ylim(0.38,1.1)
#plt.savefig('/home/sergey/Documents/pl1', format='pdf')
                  
                  
                  
                  


# In[81]:

exp1.shape


# In[82]:

exp1


# In[85]:

exp1.shape


# In[95]:

get_ipython().magic(u'matplotlib inline')

from __future__ import division
#plt.axis([xmin, xmax, ymin, ymax])
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import scipy.integrate as integrate
import scipy.special as special
from numpy.polynomial.hermite import hermval


#plt.plot([1,2,3,5],[0,3,4,9],'ro')
#plt.ylabel('some numbers')




#Explanation of the I(B) profile with QPC modes modelled by Harmonic oscillator eigenfunctions 

hbar=1.0545718E-34
omega=(0.001*1.6E-19)/hbar
mstar=0.067*9.1094E-31
L=4.6E-6
z=1j
e=1.6E-19
B=0.01


print ('omega=', omega)


def H0(range):
    return 1

def H1(range):
    return 2*np.sqrt(mstar*omega/hbar)*(range)

def H2(range):
    return 4*(np.sqrt(mstar*omega/hbar)*(range))**2-2

def H3(range):
    return 8*(np.sqrt(mstar*omega/hbar)*(range))**3-12*(np.sqrt(mstar*omega/hbar)*(range))

def H4(range):
    return 16*(np.sqrt(mstar*omega/hbar)*(range))**4-48*(np.sqrt(mstar*omega/hbar)*(range))**2+12

def H5(range):
    return 32*(np.sqrt(mstar*omega/hbar)*(range))**5-160*(np.sqrt(mstar*omega/hbar)*(range))**3+120*(np.sqrt(mstar*omega/hbar)*(range))

def H6(range):
    return 64*(np.sqrt(mstar*omega/hbar)*(range))**6-480*(np.sqrt(mstar*omega/hbar)*(range))**4+720*(np.sqrt(mstar*omega/hbar)*(range))**2-120


"""

coef = [1,0,0]
hermval(np.sqrt(mstar*omega/hbar)*range, coef)



def psi(range,coef):
    return (mstar*omega/(np.pi*hbar))**(1/4)*1/np.sqrt(2**coef*math.factorial(coef))*hermval(range,coef)*np.exp(-(mstar*omega/(2*hbar))*range**2)
    
 """ 


def psi1(range):
    return (mstar*omega/(np.pi*hbar))**(1/4)*(1/np.sqrt(2))*np.exp(-(mstar*omega/(2*hbar))*range**2)

def psi2(range):
    return (mstar*omega/(np.pi*hbar))**(1/4)*(1/np.sqrt(2))*H1(range)*np.exp(-(mstar*omega/(2*hbar))*range**2)

def psi3(range):
    return (mstar*omega/(np.pi*hbar))**(1/4)*(1/np.sqrt(8))*H2(range)*np.exp(-(mstar*omega/(2*hbar))*range**2)

def psi4(range):
    return (mstar*omega/(np.pi*hbar))**(1/4)*(1/np.sqrt(48))*H3(range)*np.exp(-(mstar*omega/(2*hbar))*range**2)

def psi5(range):
    return (mstar*omega/(np.pi*hbar))**(1/4)*(1/np.sqrt(384))*H4(range)*np.exp(-(mstar*omega/(2*hbar))*range**2)

def psi6(range):
    return (mstar*omega/(np.pi*hbar))**(1/4)*(1/np.sqrt(3840))*H5(range)*np.exp(-(mstar*omega/(2*hbar))*range**2)

def psi7(range):
    return (mstar*omega/(np.pi*hbar))**(1/4)*(1/np.sqrt(46080))*H6(range)*np.exp(-(mstar*omega/(2*hbar))*range**2)


#Fourier transforms:


#psitot2 = lambda range: (1/np.sqrt(L))*psi2(range)*np.exp(z*e*B*L*range/(2*hbar))
#integrate.quad(psitot2, -np.inf, np.inf)
#print ('FT=', integrate.quad(psitot2, -np.inf, np.inf))


range=np.arange(-250E-9,250E-9,1E-10)
#angle=range*180/np.pi


#plt.plot(range, psi1(range), 'k')
#plt.plot(range, np.absolute(psi2(range)), 'r')
#plt.plot(range, np.absolute(psi3(range)), 'b')
plt.plot(range, np.absolute(psi7(range)), 'y')


plt.xlabel('QPC constriction coordinate')

np.savetxt('test.out', range, delimiter=';')   # X is an array
np.savetxt('test7.out', np.absolute(psi7(range)), delimiter='\t')   # x,y,z equal sized 1D arrays



# In[ ]:




# In[ ]:



