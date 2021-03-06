import numpy as np
from MPI_functions import globalNorm,globalSum
import sys
#def globalNorm(r,main):
#  ## Create Global residual
#  data = main.comm.gather(np.linalg.norm(r)**2,root = 0)
#  if (main.mpi_rank == 0):
#    rn_glob = 0.
#    for j in range(0,main.num_processes):
#      rn_glob += data[j]
#    rn_glob = np.sqrt(rn_glob)
#    for j in range(1,main.num_processes):
#      main.comm.send(rn_glob, dest=j)
#  else:
#    rn_glob = main.comm.recv(source=0)
#  return rn_glob
#
#def globalSum(r,main):
#  ## Create Global residual
#  data = main.comm.gather(np.sum(r),root = 0)
#  if (main.mpi_rank == 0):
#    rn_glob = 0.
#    for j in range(0,main.num_processes):
#      rn_glob += data[j]
#    for j in range(1,main.num_processes):
#      main.comm.send(rn_glob, dest=j)
#  else:
#    rn_glob = main.comm.recv(source=0)
#  return rn_glob



def rungeKutta(Af, b, x0,main,args,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=0):
     dt = 0.0001
     r = b - Af(x0,args,main)
     rnorm = globalNorm(r,main) #same across procs
     iteration = 0
     rk4const = np.array([0.15,1.0])
     a0 = np.zeros(np.shape(x0))
     a = np.zeros(np.shape(x0))
     a0[:] = x0[:]
     a[:] = x0[:]
     print_freq = 5
     while( rnorm > 1e-9 and iteration <= 50):
       a0[:] = a[:]

       for i in range(0,np.size(rk4const) ):
         r = b - Af(a,args,main)
         rnorm_old = rnorm*1.
         rnorm = globalNorm(r,main) 
         dt = dt*np.fmin(rnorm_old/rnorm,1.001)
         #dt = dt*rnorm_old/rnorm
         a[:] = a0[:] - dt*rk4const[i]*r
       iteration += 1
       if (main.mpi_rank == 0 and iteration%print_freq == 0):# and printnorm == 1):
         sys.stdout.write(' Iteration = ' + str(iteration) + ' Runge Kutta error = ' + str(rnorm) + ' tau = ' + str(dt) +  '\n')
     sys.stdout.write(' ================================== ' +  '\n')
     return a

def GMRes(Af, b, x0,main,args,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=0):
    k_outer = 0
    bnorm = globalNorm(b,main)
    error = 10.
    while (k_outer < maxiter_outer and error >= tol):
      r = b - Af(x0,args,main)
      if (main.mpi_rank == 0 and printnorm==1):
        print('Outer true norm = ' + str(np.linalg.norm(r)))
      cs = np.zeros(maxiter) #should be the same on all procs
      sn = np.zeros(maxiter) #same on all procs
#      e1 = np.zeros(np.size(b)) #should vary across procs
      e1 = np.zeros(maxiter+1)
      e1[0] = 1
  
      rnorm = globalNorm(r,main) #same across procs
      Q = np.zeros((np.size(b),maxiter)) 
      #v = [0] * (nmax_iter)
      Q[:,0] = r / rnorm ## The first index of Q is across all procs
      H = np.zeros((maxiter + 1, maxiter)) ### this should be the same on all procs
      beta = rnorm*e1

      k = 0
      while (k < maxiter - 1  and error >= tol):
  #    for k in range(0,nmax_iter-1):
          Arnoldi(Af,H,Q,k,args,main)
          apply_givens_rotation(H,cs,sn,k)
          #update the residual vector
          beta[k+1] = -sn[k]*beta[k]
          beta[k] = cs[k]*beta[k]
          error = abs(beta[k+1])/bnorm
          ## For testing
          #y = np.linalg.solve(H[0:k,0:k],beta[0:k]) 
          #x = x0 + np.dot(Q[:,0:k],y)
          #rt = b - Af(x)
          #rtnorm = np.linalg.norm(rt)#globalNorm(rt,main)
          if (main.mpi_rank == 0 and printnorm == 1):
            sys.stdout.write('Outer iteration = ' + str(k_outer) + ' Iteration = ' + str(k) + '  GMRES error = ' + str(error) +  '\n')
            #print('Outer iteration = ' + str(k_outer) + ' Iteration = ' + str(k) + '  GMRES error = ' + str(error), ' Real norm = ' + str(rtnorm))

          k += 1
      y = np.linalg.solve(H[0:k,0:k],beta[0:k]) 
      x = x0 + np.dot(Q[:,0:k],y)
      x0[:] = x[:]
      k_outer += 1
    return x[:]



def bicgstab(Af, b, x0,main,args,tol=1e-9,maxiter_outer=1,maxiter=50,printnorm=0):
  r0 = b - Af(x0,args,main)
  rhat0 = np.zeros(np.shape(r0))
  rhat0[:] = r0[:]
  rhat0_norm = globalNorm(rhat0,main)
  r0_norm = rhat0_norm*1.
  rho0,alpha,omega0 = 1.,1.,1.
  v0,p0 = np.zeros(np.shape(r0)),np.zeros(np.shape(r0))
  iterat = 0
  while (r0_norm/rhat0_norm  >= tol and iterat <= maxiter):
    rhoi = globalSum(rhat0*r0,main)
    beta = rhoi/rho0*alpha/omega0
    p0 = r0 + beta*(p0 - omega0*v0)
    v0 = Af(p0,args,main)
    alpha = rhoi/globalSum(rhat0*v0,main)
    h = x0 + alpha*p0
    s = r0 - alpha*v0
    t = Af(s,args,main)
    omega0 = globalSum(t*s,main)/globalSum(t*t,main)
    x0 = h + omega0*s
    r0 = s - omega0*t
    #update old values
    rho0 = rhoi*1.
    r0_norm = globalNorm(r0,main)
    if (main.mpi_rank == 0 and printnorm == 1):
      sys.stdout.write(' Iteration = ' + str(iterat) + '  BICGSTAB residual = ' + str(r0_norm/rhat0_norm) +  '\n')
    iterat += 1
  return x0


def fGMRes(Af, b, x0,main,args,Minv,tol=1e-9,maxiter_outer=1,maxiter=20,printnorm=0):
    k_outer = 0
    bnorm = globalNorm(b,main)
    error = 1.
    coarse_order = np.shape(main.a.a)[1:5]
    while (k_outer < maxiter_outer and error >= tol):
      r = b - Af(x0,args,main)
      if (main.mpi_rank == 0 and printnorm==1):
        print('Outer true norm = ' + str(np.linalg.norm(r)))
      cs = np.zeros(maxiter) #should be the same on all procs
      sn = np.zeros(maxiter) #same on all procs
      e1 = np.zeros(np.size(b)) #should vary across procs
      e1[0] = 1
  
      rnorm = globalNorm(r,main) #same across procs
      Q = np.zeros((np.size(b),maxiter)) 
      Z = np.zeros((np.size(b),maxiter)) 
      #v = [0] * (nmax_iter)
      Q[:,0] = r / rnorm ## The first index of Q is across all procs
      H = np.zeros((maxiter + 1, maxiter)) ### this should be the same on all procs
      beta = rnorm*e1
      k = 0
      while (k < maxiter - 1  and error >= tol):
          Z[:,k] = Minv(Q[:,k],main,Af,args,k)
          Q[:,k+1] = Af(Z[:,k],args,main)
          Arnoldi_fgmres(Af,H,Q,k,args,main)
          apply_givens_rotation(H,cs,sn,k)
          #update the residual vector
          beta[k+1] = -sn[k]*beta[k]
          beta[k] = cs[k]*beta[k]
          error = abs(beta[k+1])/bnorm
          if (main.mpi_rank == 0 and printnorm == 1):
            sys.stdout.write('Outer iteration = ' + str(k_outer) + \
            ' Iteration = ' + str(k) + '  GMRES error = ' + str(error) +  '\n')
          k += 1
      y = np.linalg.solve(H[0:k,0:k],beta[0:k]) 
      Zy = np.dot(Z[:,0:k],y) 
      x = x0 + Zy 
      x0[:] = x[:]
      k_outer += 1
    return x[:]

def Arnoldi_fgmres(Af,H,Q,k,args,main):
    for i in range(0,k+1):
        H[i, k] = globalSum(Q[:,i]*Q[:,k+1],main)
        Q[:,k+1] = Q[:,k+1] - H[i, k] * Q[:,i]
    H[k + 1, k] = globalNorm(Q[:,k+1],main)
#    if (h[k + 1, k] != 0 and k != nmax_iter - 1):
    Q[:,k + 1] = Q[:,k+1] / H[k + 1, k]


def Arnoldi(Af,H,Q,k,args,main):
    Q[:,k+1] = Af(Q[:,k],args,main)
    for i in range(0,k+1):
        H[i, k] = globalSum(Q[:,i]*Q[:,k+1],main)
        Q[:,k+1] = Q[:,k+1] - H[i, k] * Q[:,i]
    H[k + 1, k] = globalNorm(Q[:,k+1],main)
#    if (h[k + 1, k] != 0 and k != nmax_iter - 1):
    Q[:,k + 1] = Q[:,k+1] / H[k + 1, k]
#    return h,v 

def apply_givens_rotation(H, cs, sn, k):
  #apply for ith column
  for i in range(0,k):                              
    temp     =  cs[i]*H[i,k] + sn[i]*H[i+1,k]
    H[i+1,k] = -sn[i]*H[i,k] + cs[i]*H[i+1,k]
    H[i,k]   = temp
  
  #update the next sin cos values for rotation
  cs[k],sn[k] = givens_rotation( H[k,k], H[k+1,k])
  
  #eliminate H(i+1,i)
  H[k,k] = cs[k]*H[k,k] + sn[k]*H[k+1,k]
  H[k+1,k] = 0.0

#----Calculate the Given rotation matrix----%%
def givens_rotation(v1, v2):
  if (v1==0):
    cs = 0
    sn = 1
  else:
    t=np.sqrt(v1**2+v2**2)
    cs = np.abs(v1) / t
    sn = cs * v2 / v1
  return cs,sn

def GMResOrig(Af, b, x0, nmax_iter, restart=None):
    r = b - Af(x0)
    rnorm = np.linalg.norm(r)
    v = [0] * (nmax_iter)
    v[0] = r / rnorm
    h = np.zeros((nmax_iter + 1, nmax_iter))
    for k in range(0,nmax_iter):
        w = Af(v[k])
        for j in range(0,k+1):
            h[j, k] = np.dot(v[j], w)
            w = w - h[j, k] * v[j]
        h[k + 1, k] = np.linalg.norm(w)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            v[k + 1] = w / h[k + 1, k]
        b2 = np.zeros(nmax_iter + 1)
        b2[0] = rnorm
        result = np.linalg.lstsq(h, b2)[0]
        x = np.dot(np.asarray(v).transpose(), result) + x0
        r1 = b - Af(x)
        print(np.linalg.norm(r))
    return x[:]

