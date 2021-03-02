# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:50:14 2020

WP1-2

@author: Michael
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize,spatial
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Scalar function only for Newton Rhapsod method
def Ux_sc(x,mu):
    r1=np.sqrt((x+mu)**2)
    r2=np.sqrt((x+mu-1)**2)
    return x-(1-mu)*(x+mu)/r1**3+mu*(1-mu-x)/r2**3;

def dU(pos,mu):
    x,y,z= pos
    r1 = np.sqrt((x+mu)**2+y**2+z**2)
    r2 = np.sqrt((x+mu-1)**2 +y**2+z**2)
    return np.array([ x-(1-mu)*(x+mu)/r1**3+mu*(1-mu-x)/r2**3,
                      y-(1-mu)*y/r1**3-mu*y/r2**3, 
                      -(1-mu)*z/r1**3-mu*z/r2**3 ])   

def dUdxx(pos,mu):
    x,y,z= pos
    r1 = np.sqrt((x+mu)**2+y**2+z**2)
    r2 = np.sqrt((x+mu-1)**2 +y**2+z**2)
    return 1 + 3*mu*(1-mu-x)*(1-mu-x)/r2**5-mu/r2**3-(1-mu)/r1**3 + 3*(1-mu)*(mu+x)*(mu+x)/r1**5

def dUdyy(pos,mu):
    x,y,z= pos
    r1 = np.sqrt((x+mu)**2+y**2+z**2)
    r2 = np.sqrt((x+mu-1)**2 +y**2+z**2)
    return 1 + 3*(mu*y**2)/r2**5 + 3*(1-mu)*(y**2)/r1**5 -mu/r2**3 - (1-mu)/r1**3

def dUdxy(pos,mu):
    x,y,z= pos
    r1 = np.sqrt((x+mu)**2+y**2+z**2)
    r2 = np.sqrt((x+mu-1)**2 +y**2+z**2)
    return -3*(mu-1)*y*(mu+x)/r1**5 -3*mu*y*(-mu-x+1)/r2**5

def nosail(t,X,mu):
    x,y,z,vx,vy,vz=X
    dfdt = np.zeros(6)
    dfdt[0]=vx; dfdt[1]=vy; dfdt[2]=vz
    cr=np.cross([0,0,1],[vx,vy,vz])
    DU = dU([x,y,z],mu)
    dfdt[3]=-2*cr[0]+DU[0]
    dfdt[4]=-2*cr[1]+DU[1]
    dfdt[5]=-2*cr[2]+DU[2]  
    return dfdt

def hat(vec):
    return vec/np.linalg.norm(vec)
    
def n_fun(pos,mu):
    return dU(pos,mu)/np.linalg.norm(dU(pos,mu))

def b_fun(pos,mu):
    return np.linalg.norm(pos)**2/(1-mu)*np.dot(dU(pos,mu),n_fun(pos,mu))/np.dot(hat(pos),n_fun(pos,mu))**2

def K(pos,mu):
    x,y,z = pos
    r1 = np.sqrt((x+mu)**2+y**2+z**2)
    r2 = np.sqrt((x+mu-1)**2 +y**2+z**2)
    K = (1-mu)/r1**3+mu/r2**3
    return K

def sail(t,X,mu,beta,clock,cone): 
    x,y,z,vx,vy,vz=X
    cr=np.cross([0,0,1],[vx,vy,vz])
    DU = dU([x,y,z],mu)  
    a_s = a_sail([x,y,z],mu,beta,clock,cone)
    dfdt = np.zeros(6)
    dfdt[0]=vx; dfdt[1]=vy; dfdt[2]=vz
    dfdt[3]=-2*cr[0]+DU[0]+a_s[0]
    dfdt[4]=-2*cr[1]+DU[1]+a_s[1]
    dfdt[5]=-2*cr[2]+DU[2]+a_s[2]
    return dfdt


def a_sail(pos,mu,beta,clock,cone):
    x,y,z = pos
    r1 = np.array([x+mu,y,z])
    r1_hat = hat(r1)
    th = np.cross([0.,0.,1.],r1_hat)
    th_hat = hat(th)
    eta_hat = np.cross(r1_hat, th_hat)
    T_frame = np.transpose(np.vstack((r1_hat,th_hat,eta_hat)))
    n_angle = np.array([[np.cos(cone)],[np.sin(cone)*np.sin(clock)],[np.sin(cone)*np.cos(clock)]])
    n_hat = T_frame.dot(n_angle)
    a_sail = beta*(1-mu)/(np.linalg.norm(r1)**2)*(np.dot(r1_hat,n_hat)**2)*n_hat
    return a_sail
    



 


   
#%% Define Parameters
ms = 1.9891e+30; mv = 4.8685e+24;
av = 108208179; 
beta= 0.05;
G = 6.67408e-20; epsilon=1.e-5
# Validation Lecture Slides
# mm = 7.34767309e+22; me = 5.9736e+24; am = 384401
#me = 6.0477e+24 #with Moon; 

# Important: change mu, omega for verification
mu = mv/(mv+ms)
omega=np.sqrt(G*(mv+ms)/av**3)

# Find Lagrange points L1, L3 and Eig1, Eig3 (note: Netwon method used first, then hybryd for convenience of vectorial form, same result)
# L1 = optimize.newton(Ux_sc, x0=0.9, fprime=None, args=(mu,), tol=1.48e-08)
# L3 = optimize.newton(Ux_sc, x0=-1.1, fprime=None, args=(mu,), tol=1.48e-08)

L1 = optimize.root(dU, args=(mu), x0 =[0.8,0,0] , method='hybr', jac=None, tol=None, callback=None, options=None)
eig1 = np.linalg.eig(np.array([ [0., 0., 1., 0.],[0., 0., 0., 1.],[dUdxx(L1.x,mu), dUdxy(L1.x,mu), 0., 2.],[dUdxy(L1.x,mu), dUdyy(L1.x,mu), -2., 0.] ]))               
L2 = optimize.root(dU, args=(mu), x0 =[1.3,0,0] , method='hybr', jac=None, tol=None, callback=None, options=None)
eig2 = np.linalg.eig(np.array([ [0., 0., 1., 0.],[0., 0., 0., 1.],[dUdxx(L2.x,mu), dUdxy(L2.x,mu), 0., 2.],[dUdxy(L2.x,mu), dUdyy(L2.x,mu), -2., 0.] ]))
L3 = optimize.root(dU, args=(mu), x0 =[-1.,0,0] , method='hybr', jac=None, tol=None, callback=None, options=None)
eig3 = np.linalg.eig(np.array([ [0., 0., 1., 0.],[0., 0., 0., 1.],[dUdxx(L3.x,mu), dUdxy(L3.x,mu), 0., 2.],[dUdxy(L3.x,mu), dUdyy(L3.x,mu), -2., 0.] ]))

#%% verify eigenvalues wakker method
eigv1 = np.roots((1., 0., (2-K(L1.x,mu)), 0., -(2*K(L1.x,mu)+1)*(K(L1.x,mu)-1)))
eigv2 = np.roots((1., 0., (2-K(L2.x,mu)), 0., -(2*K(L2.x,mu)+1)*(K(L2.x,mu)-1)))
eigv3 = np.roots((1., 0., (2-K(L3.x,mu)), 0., -(2*K(L3.x,mu)+1)*(K(L3.x,mu)-1)))

#%% Compute beta and orientation
shift = np.array([0.011, 0, 0])
L1s = L1.x-shift; n1 = n_fun(L1s,mu); beta1 = b_fun(L1s,mu) 
L3s = L3.x+shift; n3 = n_fun(L3s,mu); beta3 = b_fun(L3s,mu)


#%%# Compute Manifolds, without sail (note: eig1: 0-->unstable, 1-->stable)
tf = 5*365.25*86400*omega
Xs_ep = np.array([L1.x[0], L1.x[1], L1.x[2], 0, 0, 0])+epsilon*np.real(np.insert(eig1[1][:,1], [2,4], [0,0]))
sols_ep = solve_ivp(nosail, (tf, 0), Xs_ep, args=(mu,), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, rtol=1e-12, atol=1e-12)
Xs_en = np.array([L1.x[0], L1.x[1], L1.x[2], 0, 0, 0])-epsilon*np.real(np.insert(eig1[1][:,1], [2,4], [0,0]))
sols_en = solve_ivp(nosail, (tf, 0), Xs_en, args=(mu,), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, rtol=1e-12, atol=1e-12)
Xu_ep = np.array([L1.x[0], L1.x[1], L1.x[2], 0, 0, 0])+epsilon*np.real(np.insert(eig1[1][:,0], [2,4], [0,0]))
solu_ep = solve_ivp(nosail, (0, tf), Xu_ep, args=(mu,), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, rtol=1e-12, atol=1e-12)
Xu_en = np.array([L1.x[0], L1.x[1], L1.x[2], 0, 0, 0])-epsilon*np.real(np.insert(eig1[1][:,0], [2,4], [0,0]))
solu_en = solve_ivp(nosail, (0, tf), Xu_en, args=(mu,), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, rtol=1e-12, atol=1e-12)

fig, ax = plt.subplots()
ax.plot(sols_en.y[0,:], sols_en.y[1,:], 'g')
ax.plot(solu_en.y[0,:], solu_en.y[1,:], 'b')
ax.scatter([L1.x[0],L3.x[0]],[L1.x[1],L3.x[1]], s=50, c='r', marker='x')
ax.legend(['Stable, $\epsilon<0$','Unstable, $\epsilon<0$','L points'],loc='upper right')
plt.title('L1 point manifold')
plt.ylabel('y [-]')
plt.xlabel('x [-]')
plt.grid(True)

axins = zoomed_inset_axes(ax, 30, loc='center')
axins.plot(sols_ep.y[0,:], sols_ep.y[1,:], 'c')
axins.plot(solu_ep.y[0,:], solu_ep.y[1,:], 'm')
plt.yticks(visible=False)
plt.xticks(visible=False)
x1, x2, y1, y2 = 0.99, 1.01, -0.02, 0.02 # specify the limits
axins.set_xlim(x1, x2) 
axins.set_ylim(y1, y2) 
mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")
plt.legend(['Stable, $\epsilon>0$','Unstable, $\epsilon>0$'],loc='best')

plt.show()

#%% Compute Manifolds, with sail (note: eig1: 0-->unstable, 1-->stable), (note: eig1: 3-->unstable, 2-->stable) ONLY e>0!
clock = 0
tf = 5*365.25*86400*omega
Xs = np.array([L3.x[0], L3.x[1], L3.x[2], 0, 0, 0])+epsilon*np.real(np.insert(eig3[1][:,2], [2,4], [0,0]))
Xu = np.array([L1.x[0], L1.x[1], L1.x[2], 0, 0, 0])+epsilon*np.real(np.insert(eig1[1][:,0], [2,4], [0,0]))

#%% Find best angle, generate training data
step=2.5
angles    = np.arange(0, 70+step, step)
err_list  = np.zeros(np.size(angles))
i = 0
for cone in angles:   
    sols = solve_ivp(sail, (tf, 0), Xs, args=(mu,beta,clock,np.radians(cone),), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, rtol=1e-12, atol=1e-12)
    solu = solve_ivp(sail, (0, tf), Xu, args=(mu,beta,clock,np.radians(cone),), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, rtol=1e-12, atol=1e-12)
    EuN = spatial.distance.cdist(np.transpose(sols.y),np.transpose(solu.y),metric='euclidean')
    idx_dist = np.unravel_index(np.argmin(EuN, axis=None), EuN.shape)
    err_list[i] = EuN[idx_dist]
    i = i + 1
    str1_st='C:/Users/Michael/Desktop/Special Topics/output/'+'l1_'+str(cone)+'_state'
    str1_ti='C:/Users/Michael/Desktop/Special Topics/output/'+'l1_'+str(cone)+'_time'
    str3_st='C:/Users/Michael/Desktop/Special Topics/output/'+'l3_'+str(cone)+'_state'
    str3_ti='C:/Users/Michael/Desktop/Special Topics/output/'+'l3_'+str(cone)+'_time'
    np.save(str1_st, np.transpose(solu.y))
    np.save(str1_ti, np.transpose(solu.t))
    np.save(str3_st, np.transpose(sols.y))
    np.save(str3_ti, np.transpose(sols.t))
    
    
#%%
plt.figure(1)
plt.plot(np.degrees(angles),err_list)
plt.ylabel('Euclidean-norm error [-]')
plt.xlabel('Cone Angle [deg]')
plt.show()

#%% Settled on cone angle = 25 -> found tf by inspection
tf_s = 42.28; tf_u = 25.54
cone = np.radians(25);
sols = solve_ivp(sail, (tf_s, 0), Xs, args=(mu,beta,clock,cone,), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, rtol=1e-12, atol=1e-12)
solu = solve_ivp(sail, (0, tf_u), Xu, args=(mu,beta,clock,cone,), method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False, rtol=1e-12, atol=1e-12)
EuN = spatial.distance.cdist(np.transpose(sols.y),np.transpose(solu.y),metric='euclidean')
idx = np.unravel_index(np.argmin(EuN, axis=None), EuN.shape)
err = EuN[idx]
distM = spatial.distance.cdist(np.transpose(sols.y[0:3,:]),np.transpose(solu.y[0:3,:]),metric='euclidean')
idx_dist = np.unravel_index(np.argmin(distM, axis=None), distM.shape)
err_dist = distM[idx_dist]
velM =  spatial.distance.cdist(np.transpose(sols.y[3:6,:]),np.transpose(solu.y[3:6,:]),metric='euclidean')
idx_vel = np.unravel_index(np.argmin(velM, axis=None), velM.shape)
err_vel = velM[idx_vel]

#%% plot xy
plt.subplot(2, 1, 1)
plt.plot(sols.y[0,:],sols.y[1,:], 'g')
plt.plot(solu.y[0,:],solu.y[1,:], 'b')
plt.scatter([L1.x[0],L3.x[0]],[L1.x[1],L3.x[1]], s=50, c='r', marker='x')
plt.legend(['Stable','Unstable','L points'], loc='center')
plt.ylabel('y [-]')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(sols.y[0,:],sols.y[2,:], 'g')
plt.plot(solu.y[0,:],solu.y[2,:], 'b')
plt.scatter([L1.x[0],L3.x[0]],[L1.x[1],L3.x[1]], s=40, c='r', marker='x')
plt.xlabel('x [-]')
plt.ylabel('z [-]')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.grid(True)
plt.show()
















