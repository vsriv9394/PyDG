import numpy as np

##### =========== Contains all the fluxes and physics neccesary to solve the Navier-Stokes equations within a DG framework #### ============


###### ====== Inviscid Fluxes Fluxes and Eigen Values (Eigenvalues currently not in use) ==== ############
def evalFluxXEuler(main,u,f,args): 
  #f = np.zeros(np.shape(u))
  es = 1.e-30
  gamma = 1.4
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  f[0] = u[1]
  f[1] = u[1]*u[1]/(u[0]) + p
  f[2] = u[1]*u[2]/(u[0])
  f[3] = u[1]*u[3]/(u[0])
  f[4] = (u[4] + p)*u[1]/(u[0])


def evalFluxYEuler(main,u,f,args):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  f[0] = u[2]
  f[1] = u[1]*u[2]/u[0]
  f[2] = u[2]*u[2]/u[0] + p
  f[3] = u[2]*u[3]/u[0] 
  f[4] = (u[4] + p)*u[2]/u[0]



def evalFluxZEuler(main,u,f,args):
  #f = np.zeros(np.shape(u))
  gamma = 1.4
  p = (gamma - 1.)*(u[4] - 0.5*u[1]**2/u[0] - 0.5*u[2]**2/u[0] - 0.5*u[3]**2/u[0])
  f[0] = u[3]
  f[1] = u[1]*u[3]/u[0]
  f[2] = u[2]*u[3]/u[0] 
  f[3] = u[3]*u[3]/u[0] + p 
  f[4] = (u[4] + p)*u[3]/u[0]


def evalFluxXEulerLin(main,U0,f,args): 
  up = args[0]
  #decompose as U = U0 + up, where up is the perturbation
  #f = np.zeros(np.shape(u))
  es = 1.e-30
  gamma = 1.4
  u = U0[1]/U0[0]
  v = U0[2]/U0[0]
  w = U0[3]/U0[0]
  qsqr = u**2 + v**2 + w**2
  # compute H in three steps (H = E + p/rho)
  H = (gamma - 1.)*(U0[4] - 0.5*U0[0]*qsqr) #compute pressure
  H += U0[4]
  H /= U0[0]
  f[0] = up[1]
  f[1] = ( (gamma - 1.)/2.*qsqr - u**2)*up[0] + (3. - gamma)*u*up[1] + (1. - gamma)*v*up[2] + \
         (1. - gamma)*w*up[3] + (gamma - 1.)*up[4]
  f[2] = -u*v*up[0] + v*up[1] + u*up[2]
  f[3] = -u*w*up[0] + w*up[1] + u*up[3]
  f[4] = ((gamma - 1.)/2.*qsqr - H)*u*up[0] + (H + (1. - gamma)*u**2)*up[1] + (1. - gamma)*u*v*up[2] + \
         (1. - gamma)*u*w*up[3] + gamma*u*up[4]


def evalFluxYEulerLin(main,U0,f,args):
  up = args[0]
  gamma = 1.4
  u = U0[1]/U0[0]
  v = U0[2]/U0[0]
  w = U0[3]/U0[0]
  qsqr = u**2 + v**2 + w**2
  # compute H in three steps (H = E + p/rho)
  H = (gamma - 1.)*(U0[4] - 0.5*U0[0]*qsqr) #compute pressure
  H += U0[4]
  H /= U0[0]
  f[0] = up[2]
  f[1] = -v*u*up[0] + v*up[1] + u*up[2]
  f[2] = ( (gamma - 1.)/2.*qsqr - v**2)*up[0] + (1. - gamma)*u*up[1] + (3. - gamma)*v*up[2] + \
         (1. - gamma)*w*up[3] + (gamma - 1.)*up[4]
  f[3] = -v*w*up[0] + w*up[2] + v*up[3]
  f[4] = ((gamma - 1.)/2.*qsqr - H)*v*up[0] + (1. - gamma)*u*v*up[1] + (H + (1. - gamma)*v**2)*up[2] + \
         (1. - gamma)*v*w*up[3] + gamma*v*up[4]



def evalFluxZEulerLin(main,U0,f,args):
  up = args[0]
  gamma = 1.4
  u = U0[1]/U0[0]
  v = U0[2]/U0[0]
  w = U0[3]/U0[0]
  qsqr = u**2 + v**2 + w**2
  # compute H in three steps (H = E + p/rho)
  H = (gamma - 1.)*(U0[4] - 0.5*U0[0]*qsqr) #compute pressure
  H += U0[4]
  H /= U0[0]
  f[0] = up[3]
  f[1] = -u*w*up[0] + w*up[1] + u*up[3]
  f[2] = -v*w*up[0] + w*up[2] + v*up[3]
  f[3] = ( (gamma - 1.)/2.*qsqr - w**2)*up[0] + (1. - gamma)*u*up[1] + (1. - gamma)*v*up[2] + \
         (3. - gamma)*w*up[3] + (gamma - 1.)*up[4]
  f[4] = ((gamma - 1.)/2.*qsqr - H)*w*up[0] + (1. - gamma)*u*w*up[1] + (1. - gamma)*v*w*up[2] + \
          (H + (1. - gamma)*w**2)*up[3] + gamma*w*up[4]



#==================== Numerical Fluxes for the Faces =====================
#== central flux
#== rusanov flux
#== Roe flux

def eulerCentralFlux(main,UL,UR,pL,pR,n,args=None):
# PURPOSE: This function calculates the flux for the Euler equations
# using the Roe flux function
#
# INPUTS:
#    UL: conservative state vector in left cell
#    UR: conservative state vector in right cell
#    n: normal pointing from the left cell to the right cell
#
# OUTPUTS:
#  F   : the flux out of the left cell (into the right cell)
#  smag: the maximum propagation speed of disturbance
#
  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)
  rHL = UL[4] + pL
  HL = rHL/rL
  cL = np.sqrt(gamma*pL/rL)
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL

  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  qR = np.sqrt(UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])/rR
  pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)
  rHR = UR[4] + pR
  HR = rHR/rR
  cR = np.sqrt(gamma*pR/rR)
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])#-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])#-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])#-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.5*(FL[3]+FR[3])#-0.5*smax*(UR[3] - UL[3])
  F[4]    = 0.5*(FL[4]+FR[4])#-0.5*smax*(UR[4] - UL[4])
  return F


def ismailFlux(main,UL,UR,pL,pR,n,args=None):
# PURPOSE: This function calculates the flux for the Euler equations
# using the Roe flux function
#
# INPUTS:
#    UL: conservative state vector in left cell
#    UR: conservative state vector in right cell
#    n: normal pointing from the left cell to the right cell
#
# OUTPUTS:
#  F   : the flux out of the left cell (into the right cell)
#  smag: the maximum propagation speed of disturbance
#
  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)

  z1L = np.sqrt(rL/pL)
  z2L = uL*z1L
  z3L = vL*z1L
  z4L = wL*z1L
  z5L = pL*z1L


  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  qR = np.sqrt(UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])/rR
  pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)

  z1R = np.sqrt(rR/pR)
  z2R = uR*z1R
  z3R = vR*z1R
  z4R = wR*z1R
  z5R = pR*z1R

  z1bar = 0.5*(z1L + z1R)
  z2bar = 0.5*(z2L + z2R)
  z3bar = 0.5*(z3L + z3R)
  z4bar = 0.5*(z4L + z4R)
  z5bar = 0.5*(z5L + z5R)
 
  zeta = z1L/z1R
  f =  (zeta - 1.)/(zeta + 1.)
  uf = f*f
  Fa = 1.0 + 1./3.*uf + 1./5.*uf**2 + 1./7.*uf**3 + 1./9.*uf**4
  eps = 1e-3
  Fa[uf>eps] = (0.5*np.log(zeta[uf>eps])/f[uf>eps])
  z1log = z1bar/Fa

  zeta = z5L/z5R
  f =  (zeta - 1.)/(zeta + 1.)
  uf = f*f
  Fa = 1.0 + 1./3.*uf + 1./5.*uf**2 + 1./7.*uf**3 + 1./9.*uf**4
  eps = 1e-3
  Fa[uf>eps] = (0.5*np.log(zeta[uf>eps])*1./f[uf>eps])
  z5log = z5bar/Fa

  rhohat = z1bar*z5log
  uhat = z2bar/z1bar
  vhat = z3bar/z1bar
  what = z4bar/z1bar
  V_sqr = uhat**2 + vhat**2 + what**2
  V_n = uhat*n[0] + vhat*n[1] + what*n[2]

  p1hat = z5bar/z1bar
  p2hat = 0.5*(gamma+1)/(gamma)*z5log/z1log + 0.5*(gamma - 1.)/gamma*z5bar/z1bar
  ahat_sqr = gamma*p2hat/rhohat
  Hhat = ahat_sqr/(gamma - 1.) + 0.5*V_sqr

  # right flux
  F = np.zeros(np.shape(UL))  # for allocation
  rhoV_n = rhohat*V_n
  F[0]    = rhoV_n 
  F[1]    = rhoV_n*uhat + p1hat*n[0]
  F[2]    = rhoV_n*vhat + p1hat*n[1]
  F[3]    = rhoV_n*what + p1hat*n[2]
  F[4]    = rhoV_n*Hhat 
  return F



def eulerCentralFluxLinearized(main,U0L,U0R,pL,pR,n,args):
  gamma = 1.4
  upL = args[0]
  upR = args[1]
  K = gamma - 1.
  uL = U0L[1]/U0L[0]
  vL = U0L[2]/U0L[0]
  wL = U0L[3]/U0L[0]
  uR = U0R[1]/U0R[0]
  vR = U0R[2]/U0R[0]
  wR = U0R[3]/U0R[0]

  qnL = uL*n[0] + vL*n[1] + wL*n[2]
  qnR = uR*n[0] + vR*n[1] + wR*n[2]
  qsqrL = uL**2 + vL**2 + wL**2
  qsqrR = uR**2 + vR**2 + wR**2

  # compute H in three steps (H = E + p/rho)
  HL = (gamma - 1.)*(U0L[4] - 0.5*U0L[0]*qsqrL) #compute pressure
  HL += U0L[4]
  HL /= U0L[0]
  HR = (gamma - 1.)*(U0R[4] - 0.5*U0R[0]*qsqrR) #compute pressure
  HR += U0R[4]
  HR /= U0R[0]
  
  FL = np.zeros(np.shape(U0L))
  FR = np.zeros(np.shape(U0R))

  #Evaluate linearized normal flux (evaluated as dF/dU V nx + dG/dU v ny + dH/dU v nz. Jacobiam from I do like CFD, vol II) 
  FL[0] = n[0]*upL[1] + n[1]*upL[2] + n[2]*upL[3]
  FL[1] = (K/2.*qsqrL*n[0] - uL*qnL)*upL[0] + (uL*n[0] - K*uL*n[0] + qnL)*upL[1] + (uL*n[1] - K*vL*n[0])*upL[2] + (uL*n[2] - K*wL*n[0])*upL[3] + K*n[0]*upL[4]
  FL[2] = (K/2.*qsqrL*n[1] - vL*qnL)*upL[0] + (vL*n[0] - K*uL*n[1])*upL[1] + (vL*n[1] - K*vL*n[1] + qnL)*upL[2] + (vL*n[2] - K*wL*n[1])*upL[3] + K*n[1]*upL[4]
  FL[3] = (K/2.*qsqrL*n[2] - wL*qnL)*upL[0] + (wL*n[0] - K*uL*n[2])*upL[1] + (wL*n[1] - K*vL*n[2])*upL[2] + (wL*n[2] - K*wL*n[2] + qnL)*upL[3] + K*n[2]*upL[4]
  FL[4] = (K/2.*qsqrL - HL)*qnL*upL[0] + (HL*n[0] - K*uL*qnL)*upL[1] + (HL*n[1] - K*vL*qnL)*upL[2] + (HL*n[2] - K*wL*qnL)*upL[3] + gamma*qnL*upL[4]

  FR[0] = n[0]*upR[1] + n[1]*upR[2] + n[2]*upR[3]
  FR[1] = (K/2.*qsqrR*n[0] - uR*qnR)*upR[0] + (uR*n[0] - K*uR*n[0] + qnR)*upR[1] + (uR*n[1] - K*vR*n[0])*upR[2] + (uR*n[2] - K*wR*n[0])*upR[3] + K*n[0]*upR[4]
  FR[2] = (K/2.*qsqrR*n[1] - vR*qnR)*upR[0] + (vR*n[0] - K*uR*n[1])*upR[1] + (vR*n[1] - K*vR*n[1] + qnR)*upR[2] + (vR*n[2] - K*wR*n[1])*upR[3] + K*n[1]*upR[4]
  FR[3] = (K/2.*qsqrR*n[2] - wR*qnR)*upR[0] + (wR*n[0] - K*uR*n[2])*upR[1] + (wR*n[1] - K*vR*n[2])*upR[2] + (wR*n[2] - K*wR*n[2] + qnR)*upR[3] + K*n[2]*upR[4]
  FR[4] = (K/2.*qsqrR - HR)*qnR*upR[0] + (HR*n[0] - K*uR*qnR)*upR[1] + (HR*n[1] - K*vR*qnR)*upR[2] + (HR*n[2] - K*wR*qnR)*upR[3] + gamma*qnR*upR[4]

  F = np.zeros(np.shape(FL))  
  F[0]    = 0.5*(FL[0]+FR[0])
  F[1]    = 0.5*(FL[1]+FR[1])
  F[2]    = 0.5*(FL[2]+FR[2])
  F[3]    = 0.5*(FL[3]+FR[3])
  F[4]    = 0.5*(FL[4]+FR[4])
  return F


def rusanovFlux(main,UL,UR,pL,pR,n,args=None):
# PURPOSE: This function calculates the flux for the Euler equations
# using the Roe flux function
#
# INPUTS:
#    UL: conservative state vector in left cell
#    UR: conservative state vector in right cell
#    n: normal pointing from the left cell to the right cell
#
# OUTPUTS:
#  F   : the flux out of the left cell (into the right cell)
#  smag: the maximum propagation speed of disturbance
#
  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  pL = (gamma-1)*(UL[4] - 0.5*rL*qL**2.)
  rHL = UL[4] + pL
  HL = rHL/rL
  cL = np.sqrt(gamma*pL/rL)
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL

  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  qR = np.sqrt(UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])/rR
  pR = (gamma-1)*(UR[4] - 0.5*rR*qR**2.)
  rHR = UR[4] + pR
  HR = rHR/rR
  cR = np.sqrt(gamma*pR/rR)
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR

  # difference in states
  du = UR - UL

  # Roe average
  di     = np.sqrt(rR/rL)
  d1     = 1.0/(1.0+di)

  ui     = (di*uR + uL)*d1
  vi     = (di*vR + vL)*d1
  wi     = (di*wR + wL)*d1
  Hi     = (di*HR + HL)*d1

  af     = 0.5*(ui*ui+vi*vi+wi*wi)
  ucp    = ui*n[0] + vi*n[1] + wi*n[2]
  c2     = gmi*(Hi - af)
  ci     = np.sqrt(c2)
  ci1    = 1.0/ci

  #% eigenvalues

  sh = np.shape(ucp)
  lsh = np.append(3,sh)
  l = np.zeros(lsh)
  l[0] = ucp+ci
  l[1] = ucp-ci
  l[2] = ucp
  #print(np.shape(l))
  smax = np.abs(ucp) + np.abs(ci)
  #smax = np.maximum(np.abs(l[0]),np.abs(l[1]))
  # flux assembly
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*smax*(UR[0] - UL[0])
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*smax*(UR[1] - UL[1])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*smax*(UR[2] - UL[2])
  F[3]    = 0.5*(FL[3]+FR[3])-0.5*smax*(UR[3] - UL[3])
  F[4]    = 0.5*(FL[4]+FR[4])-0.5*smax*(UR[4] - UL[4])
  return F
               
  

def kfid_roeflux(main,UL,UR,pL,pR,n,args=None):
# PURPOSE: This function calculates the flux for the Euler equations
# using the Roe flux function
#
# INPUTS:
#    UL: conservative state vector in left cell
#    UR: conservative state vector in right cell
#    n: normal pointing from the left cell to the right cell
#
# OUTPUTS:
#  F   : the flux out of the left cell (into the right cell)
#  smag: the maximum propagation speed of disturbance
#
  gamma = 1.4
  gmi = gamma-1.0
  #process left state
  rL = UL[0]
  uL = UL[1]/rL
  vL = UL[2]/rL
  wL = UL[3]/rL

  unL = uL*n[0] + vL*n[1] + wL*n[2]

  qL = np.sqrt(UL[1]*UL[1] + UL[2]*UL[2] + UL[3]*UL[3])/rL
  pL = (gamma-1)*(UL[4] - 0.5*rL*qL*qL)
  rHL = UL[4] + pL
  HL = rHL/rL
  cL = np.sqrt(gamma*pL/rL)
  # left flux
  FL = np.zeros(np.shape(UL))
  FL[0] = rL*unL
  FL[1] = UL[1]*unL + pL*n[0]
  FL[2] = UL[2]*unL + pL*n[1]
  FL[3] = UL[3]*unL + pL*n[2]
  FL[4] = rHL*unL

  # process right state
  rR = UR[0]
  uR = UR[1]/rR
  vR = UR[2]/rR
  wR = UR[3]/rR
  unR = uR*n[0] + vR*n[1] + wR*n[2]
  qR = np.sqrt(UR[1]*UR[1] + UR[2]*UR[2] + UR[3]*UR[3])/rR
  pR = (gamma-1)*(UR[4] - 0.5*rR*qR*qR)
  rHR = UR[4] + pR
  HR = rHR/rR
  cR = np.sqrt(gamma*pR/rR)
  # right flux
  FR = np.zeros(np.shape(UR))
  FR[0] = rR*unR
  FR[1] = UR[1]*unR + pR*n[0]
  FR[2] = UR[2]*unR + pR*n[1]
  FR[3] = UR[3]*unR + pR*n[2]
  FR[4] = rHR*unR

  # difference in states
  du = UR - UL

  # Roe average
  di     = np.sqrt(rR/rL)
  d1     = 1.0/(1.0+di)

  ui     = (di*uR + uL)*d1
  vi     = (di*vR + vL)*d1
  wi     = (di*wR + wL)*d1
  Hi     = (di*HR + HL)*d1

  af     = 0.5*(ui*ui+vi*vi+wi*wi)
  ucp    = ui*n[0] + vi*n[1] + wi*n[2]
  c2     = gmi*(Hi - af)
  ci     = np.sqrt(c2)
  ci1    = 1.0/ci



  #% eigenvalues

  sh = np.shape(ucp)
  lsh = np.append(3,sh)
  l = np.zeros(lsh)
  l[0] = ucp+ci
  l[1] = ucp-ci
  l[2] = ucp

  #% entropy fix
  epsilon = ci*.1
  #print(l,epsilon)
  labs = np.abs(l)
  for i in range(0,3):
    l[i,labs[i]<epsilon] =  (epsilon[labs[i]<epsilon] + l[i,labs[i]<epsilon]**2)/(2.*epsilon[labs[i]<epsilon])

  l = np.abs(l)
  l3 = l[2]
  # average and half-difference of 1st and 2nd eigs
  s1    = 0.5*(l[0] + l[1])
  s2    = 0.5*(l[0] - l[1])

  # left eigenvector product generators (see Theory guide)
  G1    = gmi*(af*du[0] - ui*du[1] - vi*du[2] -wi*du[3] + du[4])
  G2    = -ucp*du[0]+du[1]*n[0]+du[2]*n[1] + du[3]*n[2]

  # required functions of G1 and G2 (again, see Theory guide)
  C1    = G1*(s1-l3)*ci1*ci1 + G2*s2*ci1
  C2    = G1*s2*ci1          + G2*(s1-l3)

  # flux assembly
  F = np.zeros(np.shape(FL))  # for allocation
  F[0]    = 0.5*(FL[0]+FR[0])-0.5*(l3*du[0] + C1   )
  F[1]    = 0.5*(FL[1]+FR[1])-0.5*(l3*du[1] + C1*ui + C2*n[0])
  F[2]    = 0.5*(FL[2]+FR[2])-0.5*(l3*du[2] + C1*vi + C2*n[1])
  F[3]    = 0.5*(FL[3]+FR[3])-0.5*(l3*du[3] + C1*wi + C2*n[2])
  F[4]    = 0.5*(FL[4]+FR[4])-0.5*(l3*du[4] + C1*Hi + C2*ucp  )
  return F



###============= Diffusion Fluxes =====================
def getGsNS(u,main):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G11 = np.zeros(ashape)
  G21 = np.zeros(ashape)
  G31 = np.zeros(ashape)
  G12 = np.zeros(ashape)
  G22 = np.zeros(ashape)
  G32 = np.zeros(ashape)
  G13 = np.zeros(ashape)
  G23 = np.zeros(ashape)
  G33 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2
  mu_by_rho = main.mu[0]/u[0]
  G11[1,0] = -4./3.*v1*mu_by_rho
  G11[1,1] = 4./3.*mu_by_rho
  G11[2,0] = -v2*mu_by_rho
  G11[2,2] = mu_by_rho
  G11[3,0] = -v3*mu_by_rho
  G11[3,3] = mu_by_rho
  G11[4,0] = -(4./3.*v1**2 + v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G11[4,1] = (4./3. - gamma/Pr)*v1*mu_by_rho
  G11[4,2] = (1. - gamma/Pr)*v2*mu_by_rho
  G11[4,3] = (1. - gamma/Pr)*v3*mu_by_rho
  G11[4,4] = gamma/Pr*mu_by_rho

  G21[1,0] = G11[2,0]#-v2*mu_by_rho
  G21[1,2] = mu_by_rho
  G21[2,0] = 2./3.*v1*mu_by_rho
  G21[2,1] = -2./3.*mu_by_rho
  G21[4,0] = -1./3.*v1*v2*mu_by_rho
  G21[4,1] = -2./3.*v2*mu_by_rho
  G21[4,2] = v1*mu_by_rho

  G31[1,0] = G11[3,0]#-v3*mu_by_rho
  G31[1,3] = mu_by_rho
  G31[3,0] = G21[2,0]#2./3.*v1*mu_by_rho
  G31[3,1] = G21[2,1]#-2./3.*mu_by_rho
  G31[4,0] = -1./3.*v1*v3*mu_by_rho
  G31[4,1] = 2./3.*G11[3,0]#-2./3.*v3*mu_by_rho
  G31[4,3] = G21[4,2]#v1*mu_by_rho

  G12[1,0] = -G21[4,1]#2./3.*v2*mu_by_rho
  G12[1,2] = G21[2,1]#-2./3.*mu_by_rho
  G12[2,0] = -G31[4,3]#-v1*mu_by_rho
  G12[2,1] = mu_by_rho
  G12[4,0] = G21[4,0]#-1./3.*v1*v2*mu_by_rho
  G12[4,1] = -G21[1,0]#v2*mu_by_rho
  G12[4,2] = -G31[3,0]#-2./3.*v1*mu_by_rho

  G22[1,0] = G12[2,0]#-v1*mu_by_rho
  G22[1,1] = mu_by_rho
  G22[2,0] = -4./3.*v2*mu_by_rho
  G22[2,2] = 4./3.*mu_by_rho
  G22[3,0] = G31[1,0]#-v3*mu_by_rho
  G22[3,3] = mu_by_rho
  G22[4,0] = -(v1**2 + 4./3.*v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G22[4,1] = (1. - gamma/Pr)*v1*mu_by_rho
  G22[4,2] = (4./3. - gamma/Pr)*v2*mu_by_rho
  G22[4,3] = G11[4,3]#(1. - gamma/Pr)*v3*mu_by_rho
  G22[4,4] = G11[4,4]#gamma/Pr*mu_by_rho

  G32[2,0] = G22[3,0]#-v3*mu_by_rho
  G32[2,3] = mu_by_rho
  G32[3,0] = G12[1,0]#2./3.*v2*mu_by_rho
  G32[3,2] = G12[1,2]#-2./3.*mu_by_rho
  G32[4,0] = -1./3.*v2*v3*mu_by_rho
  G32[4,2] = G31[4,1]#-2./3.*v3*mu_by_rho
  G32[4,3] = G12[4,1]#v2*mu_by_rho

  G13[1,0] = -G31[4,1]#2./3.*v3*mu_by_rho
  G13[1,3] = -2./3.*mu_by_rho
  G13[3,0] = G12[2,0]#-v1*mu_by_rho
  G13[3,1] = mu_by_rho
  G13[4,0] = G31[4,0]#-1./3.*v1*v3*mu_by_rho
  G13[4,1] = -G31[1,0]#v3*mu_by_rho
  G13[4,3] = G12[4,2]#-2./3.*v1*mu_by_rho

  G23[2,0] = G13[1,0]#2./3.*v3*mu_by_rho
  G23[2,3] = G12[1,2]#-2./3.*mu_by_rho
  G23[3,0] = G21[1,0]#-v2*mu_by_rho
  G23[3,2] = mu_by_rho
  G23[4,0] = -1./3.*v2*v3*mu_by_rho
  G23[4,2] = G13[4,1]#v3*mu_by_rho
  G23[4,3] = G21[4,1]#-2./3.*v2*mu_by_rho

  G33[1,0] = G12[2,0]#-v1*mu_by_rho
  G33[1,1] = mu_by_rho
  G33[2,0] = G21[1,0]#-v2*mu_by_rho
  G33[2,2] = mu_by_rho
  G33[3,0] = -4./3.*v3*mu_by_rho
  G33[3,3] = 4./3.*mu_by_rho
  G33[4,0] = -(v1**2 + v2**2 + 4./3.*v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G33[4,1] = G22[4,1]#(1. - gamma/Pr)*v1*mu_by_rho
  G33[4,2] = G11[4,2]#(1. - gamma/Pr)*v2*mu_by_rho
  G33[4,3] = (4./3. - gamma/Pr)*v3*mu_by_rho
  G33[4,4] = G11[4,4]#gamma/Pr*mu_by_rho

  return G11,G12,G13,G21,G22,G23,G31,G32,G33



def getGsNSX_FAST(u,main,mu,V):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  fvG11 = np.zeros(np.shape(u))
  fvG21 = np.zeros(np.shape(u))
  fvG31 = np.zeros(np.shape(u))

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  fvG11[1] = 4./3.*mu_by_rho*(V[1] - v1*V[0])
  fvG11[2] = mu_by_rho*(V[2] - v2*V[0])
  fvG11[3] = mu_by_rho*(V[3] - v3*V[0])
  fvG11[4] =  -(4./3.*v1**2 + v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*V[0] + \
            (4./3. - gamma/Pr)*v1*V[1] + (1. - gamma/Pr)*v2*V[2] + \
            (1. - gamma/Pr)*v3*V[3] + gamma/Pr*V[4]
  fvG11[4] *= mu_by_rho


  fvG21[1] = mu_by_rho*(V[2] - v2*V[0])
  fvG21[2] = 2./3.*mu_by_rho*(v1*V[0] - V[1])
  fvG21[3] = 0
  fvG21[4] = mu_by_rho*(v1*V[2] - 2./3.*v2*V[1] - 1./3.*v1*v2*V[0] )

  fvG31[1] = mu_by_rho*(V[3] - v3*V[0])
  fvG31[3] = 2./3.*mu_by_rho*(v1*V[0] - V[1])
  fvG31[4] = mu_by_rho*(v1*V[3] - 2./3.*v3*V[1] - 1./3.*v1*v3*V[0])
  return fvG11,fvG21,fvG31


def getGsNSY_FAST(u,main,mu,V):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  fvG12 = np.zeros(np.shape(u))
  fvG22 = np.zeros(np.shape(u))
  fvG32 = np.zeros(np.shape(u))

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  fvG12[1] = 2./3.*mu_by_rho*(v2*V[0] - V[2])
  fvG12[2] = mu_by_rho*(V[1] - v1*V[0])
  fvG12[4] = mu_by_rho*(-2./3.*v1*V[2] + v2*V[1] - 1./3.*v1*v2*V[0])


  fvG22[1] = mu_by_rho*(V[1] - v1*V[0])
  fvG22[2] = 4./3.*mu_by_rho*(V[2] - v2*V[0])
  fvG22[3] = mu_by_rho*(V[3] - v3*V[0])
  fvG22[4] = -(v1**2 + 4./3.*v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*V[0] + \
             (1. - gamma/Pr)*v1*V[1] + (4./3. - gamma/Pr)*v2*V[2] + \
             (1. - gamma/Pr)*v3*V[3] +  gamma/Pr*V[4]
  fvG22[4] *= mu_by_rho

  fvG32[2] = mu_by_rho*(V[3] - v3*V[0])
  fvG32[3] = 2./3.*mu_by_rho*(v2*V[0] - V[2])
  fvG32[4] = mu_by_rho*(v2*V[3] -2./3.*v3*V[2] - 1./3.*v2*v3*V[0])


  return fvG12,fvG22,fvG32

def getGsNSZ_FAST(u,main,mu,V):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  fvG13 = np.zeros(np.shape(u))
  fvG23 = np.zeros(np.shape(u))
  fvG33 = np.zeros(np.shape(u))

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  fvG13[1] = 2./3.*mu_by_rho*(v3*V[0] - V[3])
  fvG13[3] = mu_by_rho*(V[1] - v1*V[0])
  fvG13[4] = mu_by_rho*(-2./3.*v1*V[3] + v3*V[1] - 1./3.*v1*v3*V[0])

  fvG23[2] = 2./3.*mu_by_rho*(v3*V[0] - V[3])
  fvG23[3] = mu_by_rho*(V[2] - v2*V[0])
  fvG23[4] = mu_by_rho*(-2./3.*v2*V[3] + v3*V[2] - 1./3.*v2*v3*V[0])


  fvG33[1] = mu_by_rho*(V[1] - v1*V[0])
  fvG33[2] = mu_by_rho*(V[2] - v2*V[0])
  fvG33[3] = 4./3.*mu_by_rho*(V[3] - v3*V[0])
 
  fvG33[4] = -(v1**2 + v2**2 + 4./3.*v3**2 + gamma/Pr*(E - vsqr) )*V[0] + \
             (1. - gamma/Pr)*v1*V[1] + (1. - gamma/Pr)*v2*V[2] + \
             (4./3. - gamma/Pr)*v3*V[3] + gamma/Pr*V[4]
  fvG33[4] *= mu_by_rho

  return fvG13,fvG23,fvG33



def getGsNSX(u,main,mu):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G11 = np.zeros(ashape)
  G21 = np.zeros(ashape)
  G31 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  G11[1,0] = -4./3.*v1*mu_by_rho
  G11[1,1] = 4./3.*mu_by_rho
  G11[2,0] = -v2*mu_by_rho
  G11[2,2] = mu_by_rho
  G11[3,0] = -v3*mu_by_rho
  G11[3,3] = mu_by_rho
  G11[4,0] = -(4./3.*v1**2 + v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G11[4,1] = (4./3. - gamma/Pr)*v1*mu_by_rho
  G11[4,2] = (1. - gamma/Pr)*v2*mu_by_rho
  G11[4,3] = (1. - gamma/Pr)*v3*mu_by_rho
  G11[4,4] = gamma/Pr*mu_by_rho

  G21[1,0] = G11[2,0]#-v2*mu_by_rho
  G21[1,2] = mu_by_rho
  G21[2,0] = 2./3.*v1*mu_by_rho
  G21[2,1] = -2./3.*mu_by_rho
  G21[4,0] = -1./3.*v1*v2*mu_by_rho
  G21[4,1] = -2./3.*v2*mu_by_rho
  G21[4,2] = v1*mu_by_rho

  G31[1,0] = G11[3,0]#-v3*mu_by_rho
  G31[1,3] = mu_by_rho
  G31[3,0] = G21[2,0]#2./3.*v1*mu_by_rho
  G31[3,1] = G21[2,1]#-2./3.*mu_by_rho
  G31[4,0] = -1./3.*v1*v3*mu_by_rho
  G31[4,1] = 2./3.*G11[3,0]#-2./3.*v3*mu_by_rho
  G31[4,3] = G21[4,2]#v1*mu_by_rho
  return G11,G21,G31



def getGsNSY(u,main,mu):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G12 = np.zeros(ashape)
  G22 = np.zeros(ashape)
  G32 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2

  mu_by_rho = mu[0]/u[0]
  G12[1,0] = 2./3.*v2*mu_by_rho
  G12[1,2] = -2./3.*mu_by_rho
  G12[2,0] = -v1*mu_by_rho
  G12[2,1] = mu_by_rho
  G12[4,0] = -1./3.*v1*v2*mu_by_rho
  G12[4,1] = v2*mu_by_rho
  G12[4,2] = -2./3.*v1*mu_by_rho

  G22[1,0] = G12[2,0]#-v1*mu_by_rho
  G22[1,1] = mu_by_rho
  G22[2,0] = -4./3.*v2*mu_by_rho
  G22[2,2] = 4./3.*mu_by_rho
  G22[3,0] = -v3*mu_by_rho
  G22[3,3] = mu_by_rho
  G22[4,0] = -(v1**2 + 4./3.*v2**2 + v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G22[4,1] = (1. - gamma/Pr)*v1*mu_by_rho
  G22[4,2] = (4./3. - gamma/Pr)*v2*mu_by_rho
  G22[4,3] = (1. - gamma/Pr)*v3*mu_by_rho
  G22[4,4] = gamma/Pr*mu_by_rho

  G32[2,0] = G22[3,0]#-v3*mu_by_rho
  G32[2,3] = mu_by_rho
  G32[3,0] = G12[1,0]#2./3.*v2*mu_by_rho
  G32[3,2] = G12[1,2]#-2./3.*mu_by_rho
  G32[4,0] = -1./3.*v2*v3*mu_by_rho
  G32[4,2] = -2./3.*v3*mu_by_rho
  G32[4,3] = G12[4,1]#v2*mu_by_rho
  return G12,G22,G32

def getGsNSZ(u,main,mu):
  nvars = np.shape(u)[0]
  gamma = 1.4
  Pr = 0.72
  ashape = np.array(np.shape(u[0]))
  ashape = np.insert(ashape,0,nvars)
  ashape = np.insert(ashape,0,nvars)
  G13 = np.zeros(ashape)
  G23 = np.zeros(ashape)
  G33 = np.zeros(ashape)

  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  E = u[4]/u[0]
  vsqr = v1**2 + v2**2 + v3**2
  mu_by_rho = mu[0]/u[0]

  G13[1,0] = 2./3.*v3*mu_by_rho
  G13[1,3] = -2./3.*mu_by_rho
  G13[3,0] = -v1*mu_by_rho
  G13[3,1] = mu_by_rho
  G13[4,0] = -1./3.*v1*v3*mu_by_rho
  G13[4,1] = v3*mu_by_rho
  G13[4,3] = -2./3.*v1*mu_by_rho

  G23[2,0] = G13[1,0]#2./3.*v3*mu_by_rho
  G23[2,3] = -2./3.*mu_by_rho
  G23[3,0] = -v2*mu_by_rho
  G23[3,2] = mu_by_rho
  G23[4,0] = -1./3.*v2*v3*mu_by_rho
  G23[4,2] = G13[4,1]#v3*mu_by_rho
  G23[4,3] = -2./3.*v2*mu_by_rho

  G33[1,0] = -v1*mu_by_rho
  G33[1,1] = mu_by_rho
  G33[2,0] = -v2*mu_by_rho
  G33[2,2] = mu_by_rho
  G33[3,0] = -4./3.*v3*mu_by_rho
  G33[3,3] = 4./3.*mu_by_rho
  G33[4,0] = -(v1**2 + v2**2 + 4./3.*v3**2 + gamma/Pr*(E - vsqr) )*mu_by_rho
  G33[4,1] = (1. - gamma/Pr)*v1*mu_by_rho
  G33[4,2] = (1. - gamma/Pr)*v2*mu_by_rho
  G33[4,3] = (4./3. - gamma/Pr)*v3*mu_by_rho
  G33[4,4] = gamma/Pr*mu_by_rho
  return G13,G23,G33


def evalViscousFluxXNS_IP(main,u,Ux,Uy,Uz,mu):
  gamma = 1.4
  Pr = 0.72
  ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
  ux = 1./u[0]*(Ux[1] - u[1]/u[0]*Ux[0])
  vx = 1./u[0]*(Ux[2] - u[2]/u[0]*Ux[0])
  wx = 1./u[0]*(Ux[3] - u[3]/u[0]*Ux[0])
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uy = 1./u[0]*(Uy[1] - u[1]/u[0]*Uy[0])
  vy = 1./u[0]*(Uy[2] - u[2]/u[0]*Uy[0])
  ## ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
  uz = 1./u[0]*(Uz[1] - u[1]/u[0]*Uz[0])
  wz = 1./u[0]*(Uz[3] - u[3]/u[0]*Uz[0])
  ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
  ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
  kTx =( 1./u[0]*(Ux[4] - u[4]/u[0]*Ux[0] - (u[1]*ux + u[2]*vx + u[3]*wx)  ))*mu[0]*gamma/Pr

  fx = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  fx[1] = 2./3.*mu[0]*(2.*ux - vy - wz) #tau11
  fx[2] = mu[0]*(uy + vx)  #tau11
  fx[3] = mu[0]*(uz + wx) #tau13
  fx[4] = fx[1]*v1 + fx[2]*v2 + fx[3]*v3 + kTx
  return fx


def evalViscousFluxYNS_IP(main,u,Ux,Uy,Uz,mu):
  gamma = 1.4
  Pr = 0.72
  ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
  ux = 1./u[0]*(Ux[1] - u[1]/u[0]*Ux[0])
  vx = 1./u[0]*(Ux[2] - u[2]/u[0]*Ux[0])
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  uy = 1./u[0]*(Uy[1] - u[1]/u[0]*Uy[0])
  vy = 1./u[0]*(Uy[2] - u[2]/u[0]*Uy[0])
  wy = 1./u[0]*(Uy[3] - u[3]/u[0]*Uy[0])
  ## ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
  vz = 1./u[0]*(Uz[2] - u[2]/u[0]*Uz[0])
  wz = 1./u[0]*(Uz[3] - u[3]/u[0]*Uz[0])
  ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
  ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
  kTy =( 1./u[0]*(Uy[4] - u[4]/u[0]*Uy[0] - (u[1]*uy + u[2]*vy + u[3]*wy)  ))*mu[0]*gamma/Pr

  fy = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  fy[1] = mu[0]*(vx + uy)  #tau12
  fy[2] = 2./3.*mu[0]*(2.*vy - ux - wz) #tau22
  fy[3] = mu[0]*(vz + wy) #tau23
  fy[4] = fy[1]*v1 + fy[2]*v2 + fy[3]*v3 + kTy
  return fy

def evalViscousFluxZNS_IP(main,u,Ux,Uy,Uz,mu):
  gamma = 1.4
  Pr = 0.72
  ## ->  v_x = 1/rho d/dx(rho v) - rho v /rho^2 rho_x
  ux = 1./u[0]*(Ux[1] - u[1]/u[0]*Ux[0])
  wx = 1./u[0]*(Ux[3] - u[3]/u[0]*Ux[0])
  ## ->  u_y = 1/rho d/dy(rho u) - rho u /rho^2 rho_y
  vy = 1./u[0]*(Uy[2] - u[2]/u[0]*Uy[0])
  wy = 1./u[0]*(Uy[3] - u[3]/u[0]*Uy[0])
  ## ->  u_z = 1/rho d/dz(rho u) - rho u /rho^2 rho_z
  uz = 1./u[0]*(Uz[1] - u[1]/u[0]*Uz[0])
  vz = 1./u[0]*(Uz[2] - u[2]/u[0]*Uz[0])
  wz = 1./u[0]*(Uz[3] - u[3]/u[0]*Uz[0])
  ## -> (kT)_x =d/dx[ (mu gamma)/Pr*(E - 1/2 v^2 ) ]
  ## ->        =mu gamma/Pr *[ dE/dx - 0.5 d/dx(v1^2 + v2^2) ]
  ## ->        =mu gamma/Pr *[ dE/dx - (v1 v1_x + v2 v2_x) ]
  ## ->  E_x = 1/rho d/x(rho E) - rho E /rho^2 rho_x
  kTz =( 1./u[0]*(Uz[4] - u[4]/u[0]*Uz[0] - (u[1]*uz + u[2]*vz + u[3]*wz) ) )*mu[0]*gamma/Pr

  fz = np.zeros(np.shape(u))
  v1 = u[1]/u[0]
  v2 = u[2]/u[0]
  v3 = u[3]/u[0]
  fz[1] = mu[0]*(uz + wx)  #tau13
  fz[2] = mu[0]*(vz + wy)  #tau23
  fz[3] = 2./3.*mu[0]*(2.*wz - ux - vy)
  fz[4] = fz[1]*v1 + fz[2]*v2 + fz[3]*v3 + kTz
  return fz


### Diffusion fluxes for BR1

### viscous fluxes
def evalViscousFluxXNS_BR1(main,U,fv,cgas):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  fv[0] =  4./3.*u  #tau11 = (du/dx + du/dx - 2/3 (du/dx + dv/dy + dw/dz) ) 
  fv[1] = -2./3.*u  #tau22 = (dv/dy + dv/dy - 2/3 (du/dx + dv/dy + dw/dZ) )
  fv[2] = -2./3.*u  #tau33 = (dw/dz + dw/dz - 2/3 (du/dx + dv/dy + dw/dz) )
  fv[3] = v         #tau12 = (du/dy + dv/dx)
  fv[4] = w         #tau13 = (du/dz + dw/dx)
  fv[5] = 0.           #tau23 = (dv/dz + dw/dy)
  T = (U[4]/U[0] - 0.5*( u**2 + v**2 + w**2 ) ) #kinda a psuedo tmp, should divide by Cv but it's constant so this is taken care of in the tauFlux with gamma
  fv[6] = T
  fv[7] = 0.
  fv[8] = 0.
#
def evalViscousFluxYNS_BR1(main,U,fv,cgas):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  fv[0] = -2./3.*v  #tau11 = (du/dx + du/dx - 2/3 (du/dx + dv/dy + dw/dz) ) 
  fv[1] =  4./3.*v  #tau22 = (dv/dy + dv/dy - 2/3 (du/dx + dv/dy + dw/dZ) )
  fv[2] = -2./3.*v  #tau33 = (dw/dz + dw/dz - 2/3 (du/dx + dv/dy + dw/dz) )
  fv[3] = u        #tau12 = (du/dy + dv/dx)
  fv[4] = 0            #tau13 = (du/dz + dw/dx)
  fv[5] = w         #tau23 = (dv/dz + dw/dy)
  fv[6] = 0.
  T = (U[4]/U[0] - 0.5*( u**2 + v**2 + w**2 ) )
  fv[7] = T
  fv[8] = 0.

def evalViscousFluxZNS_BR1(main,U,fv,cgas):
  u = U[1]/U[0]
  v = U[2]/U[0]
  w = U[3]/U[0]
  fv[0] = -2./3.*w  #tau11 = (du/dx + du/dx - 2/3 (du/dx + dv/dy + dw/dz) ) 
  fv[1] = -2./3.*w  #tau22 = (dv/dy + dv/dy - 2/3 (du/dx + dv/dy + dw/dZ) )
  fv[2] =  4./3.*w  #tau33 = (dw/dz + dw/dz - 2/3 (du/dx + dv/dy + dw/dz) )
  fv[3] = 0.           #tau12 = (du/dy + dv/dx)
  fv[4] = u         #tau13 = (du/dz + dw/dx)
  fv[5] = v        #tau23 = (dv/dz + dw/dy)
  T = (U[4]/U[0] - 0.5*( u**2 + v**2 + w**2 ) )
  fv[6] = 0.
  fv[7] = 0.
  fv[8] = T

def evalTauFluxXNS_BR1(main,tau,u,fvX,mu,cgas):
  Pr = 0.72
  gamma = 1.4
  fvX[0] = 0.
  fvX[1] = mu*tau[0] #tau11
  fvX[2] = mu*tau[3] #tau21
  fvX[3] = mu*tau[4] #tau31
  fvX[4] = mu*(tau[0]*u[1]/u[0] + tau[3]*u[2]/u[0] + tau[4]*u[3]/u[0] + gamma/Pr*tau[6] )

def evalTauFluxYNS_BR1(main,tau,u,fvY,mu,cgas):
  Pr = 0.72
  gamma = 1.4
  fvY[0] = 0.
  fvY[1] = mu*tau[3] #tau21
  fvY[2] = mu*tau[1] #tau22
  fvY[3] = mu*tau[5] #tau23
  fvY[4] = mu*(tau[3]*u[1]/u[0] + tau[1]*u[2]/u[0] + tau[5]*u[3]/u[0] + gamma/Pr*tau[7])

def evalTauFluxZNS_BR1(main,tau,u,fvZ,mu,cgas):
  Pr = 0.72
  gamma = 1.4
  fvZ[0] = 0.
  fvZ[1] = mu*tau[4] #tau31
  fvZ[2] = mu*tau[5] #tau32
  fvZ[3] = mu*tau[2] #tau33
  fvZ[4] = mu*(tau[4]*u[1]/u[0] + tau[5]*u[2]/u[0] + tau[2]*u[3]/u[0] + gamma/Pr*tau[8])

