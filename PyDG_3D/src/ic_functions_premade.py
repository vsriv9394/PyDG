import numpy as np
def TGVIC(x,y,z,gas):
  rho = 1.
  rho0 = 1.
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  u = np.sin(x)*np.cos(y)*np.cos(z)
  v = -np.cos(x)*np.sin(y)*np.cos(z)
  w = 0
  speed_of_sound = 10.
  gamma = 1.4
  R = 287.
  p0 = speed_of_sound**2*gamma/rho0
  Cv = 5./2.*R
  p = p0 + rho0/16.*(np.cos(2.*x) + np.cos(2.*y) )*(np.cos(2.*z) + 2.)
  T = p/(rho*R)
  E = Cv*T + 0.5*(u**2 + v**2 + w**2)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q


def shocktubeIC(x,y,z,gas):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = gas.gamma 
  Cv = gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  p = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(p))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  p[:] = 1.
  #p[:,:,:,0:Nelx/2,:,:] = 1.
  p[x>0.7] = 0.5
  rho[:] = 1
  rho[x>0.7] = 0.525
#  rho[:,:,:,0:Nelx/2,:,:] = 1.
#  rho[:,:,:,Nelx/2::,:,:] = 0.8
  T = p/(rho*R)
  E[:] = Cv*T + 0.5*(u**2 + v**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q



def zeroFSIC(x,y,z,gas):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = gas.gamma 
  Cv = gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  T = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(T))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  T[:] = 1.
  rho[:] = T**(1./(gamma - 1.))
  E[:] = Cv*T + 0.5*(u**2 + v**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q

def vortexICS(x,y,z,gas):
  nqx,nqy,nqz,Nelx,Nely,Nelz = np.shape(x)
  q = np.zeros((5,nqx,nqy,nqz,Nelx,Nely,Nelz))
  gamma = gas.gamma
  y0 = 5.
  x0 = 5.
  Cv = gas.Cv
  Cp = Cv*gamma
  R = Cp - Cv
  T = np.zeros((nqx,nqy,nqz,Nelx,Nely,Nelz))
  rho = np.zeros(np.shape(T))
  u = np.zeros(np.shape(rho))
  v = np.zeros(np.shape(u))
  w = np.zeros(np.shape(u))
  E = np.zeros(np.shape(u))
  r = ( (x - x0)**2 + (y - y0)**2 )**0.5
  beta = 5.
  pi = np.pi
  T[:] = 1. - (gamma - 1.)*beta**2/(8.*gamma*pi**2)*np.exp(1. - r**2)

  rho[:] = T**(1./(gamma - 1.))
  u[:] = 1. + beta/(2.*pi)*np.exp( (1. - r**2)/2.)*-(y - y0)
  v[:] = 1. +  beta/(2.*pi)*np.exp( (1. - r**2)/2.)*(x - x0)
  E[:] = Cv*T + 0.5*(u**2 + v**2)
  q[0] = rho
  q[1] = rho*u
  q[2] = rho*v
  q[3] = rho*w
  q[4] = rho*E
  return q