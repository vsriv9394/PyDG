import numpy as np
def computeEnergyCantera(main,u,v,w):
  # get internal energy
  e = main.cgas_field.UV[0]
  # now add kinetic energy
  e += 0.5*(u**2 + v**2 + w**2)
  return e

def update_state_cantera(main):
    rhoi = 1./main.a.u[0]
#    rhoiL = 1./main.a.uL[0]
#    rhoiR = 1./main.a.uR[0]
#    rhoiU = 1./main.a.uU[0]
#    rhoiD = 1./main.a.uD[0]
#    rhoiF = 1./main.a.uF[0]
#    rhoiB = 1./main.a.uB[0]
#    rhoiL_edge = 1./main.a.uL_edge[0]
#    rhoiR_edge = 1./main.a.uR_edge[0]
#    rhoiU_edge = 1./main.a.uU_edge[0]
#    rhoiD_edge = 1./main.a.uD_edge[0]
#    rhoiF_edge = 1./main.a.uF_edge[0]
#    rhoiB_edge = 1./main.a.uB_edge[0]
#
    fa = np.zeros((np.size(main.a.u[0]),np.shape(main.a.u)[0]-5))
#    faR = np.zeros((np.size(main.a.uL[0]),np.shape(main.a.uR)[0]-5))
#    faL = np.zeros((np.size(main.a.uR[0]),np.shape(main.a.uL)[0]-5))
#    faU = np.zeros((np.size(main.a.uU[0]),np.shape(main.a.uU)[0]-5))
#    faD = np.zeros((np.size(main.a.uD[0]),np.shape(main.a.uD)[0]-5))
#    faF = np.zeros((np.size(main.a.uF[0]),np.shape(main.a.uF)[0]-5))
#    faB = np.zeros((np.size(main.a.uB[0]),np.shape(main.a.uB)[0]-5))
#    faR_edge = np.zeros((np.size(main.a.uL_edge[0]),np.shape(main.a.uR_edge)[0]-5))
#    faL_edge = np.zeros((np.size(main.a.uR_edge[0]),np.shape(main.a.uL_edge)[0]-5))
#    faU_edge = np.zeros((np.size(main.a.uU_edge[0]),np.shape(main.a.uU_edge)[0]-5))
#    faD_edge = np.zeros((np.size(main.a.uD_edge[0]),np.shape(main.a.uD_edge)[0]-5))
#    faF_edge = np.zeros((np.size(main.a.uF_edge[0]),np.shape(main.a.uF_edge)[0]-5))
#    faB_edge = np.zeros((np.size(main.a.uB_edge[0]),np.shape(main.a.uB_edge)[0]-5))

    for i in range(0,np.shape(main.a.u)[0]-5):
      fa[:,i] = ( main.a.u[5+i]*rhoi ).flatten()
#      faR[:,i] = ( main.a.uR[5+i]*rhoiR ).flatten()
#      faL[:,i] = ( main.a.uL[5+i]*rhoiL ).flatten()
#      faU[:,i] = ( main.a.uU[5+i]*rhoiU ).flatten()
#      faD[:,i] = ( main.a.uD[5+i]*rhoiD ).flatten()
#      faF[:,i] = ( main.a.uF[5+i]*rhoiF ).flatten()
#      faB[:,i] = ( main.a.uB[5+i]*rhoiB ).flatten()
#      faR_edge[:,i] = ( main.a.uR_edge[5+i]*rhoiR_edge ).flatten()
#      faL_edge[:,i] = ( main.a.uL_edge[5+i]*rhoiL_edge ).flatten()
#      faU_edge[:,i] = ( main.a.uU_edge[5+i]*rhoiU_edge ).flatten()
#      faD_edge[:,i] = ( main.a.uD_edge[5+i]*rhoiD_edge ).flatten()
#      faF_edge[:,i] = ( main.a.uF_edge[5+i]*rhoiF_edge ).flatten()
#      faB_edge[:,i] = ( main.a.uB_edge[5+i]*rhoiB_edge ).flatten()


    e = main.a.u[4]*rhoi - 0.5*rhoi**2*(main.a.u[1]**2 + main.a.u[2]**2 + main.a.u[3]**2)
#    eR = main.a.uR[4]*rhoiR - 0.5*rhoiR**2*(main.a.uR[1]**2 + main.a.uR[2]**2 + main.a.uR[3]**2)
#    eL = main.a.uL[4]*rhoiL - 0.5*rhoiL**2*(main.a.uL[1]**2 + main.a.uL[2]**2 + main.a.uL[3]**2)
#    eU = main.a.uU[4]*rhoiU - 0.5*rhoiU**2*(main.a.uU[1]**2 + main.a.uU[2]**2 + main.a.uU[3]**2)
#    eD = main.a.uD[4]*rhoiD - 0.5*rhoiD**2*(main.a.uD[1]**2 + main.a.uD[2]**2 + main.a.uD[3]**2)
#    eF = main.a.uF[4]*rhoiF - 0.5*rhoiF**2*(main.a.uF[1]**2 + main.a.uF[2]**2 + main.a.uF[3]**2)
#    eB = main.a.uB[4]*rhoiB - 0.5*rhoiB**2*(main.a.uB[1]**2 + main.a.uB[2]**2 + main.a.uB[3]**2)
#    eR_edge = main.a.uR_edge[4]*rhoiR_edge - 0.5*rhoiR_edge**2*(main.a.uR_edge[1]**2 + main.a.uR_edge[2]**2 + main.a.uR_edge[3]**2)
#    eL_edge = main.a.uL_edge[4]*rhoiL_edge - 0.5*rhoiL_edge**2*(main.a.uL_edge[1]**2 + main.a.uL_edge[2]**2 + main.a.uL_edge[3]**2)
#    eU_edge = main.a.uU_edge[4]*rhoiU_edge - 0.5*rhoiU_edge**2*(main.a.uU_edge[1]**2 + main.a.uU_edge[2]**2 + main.a.uU_edge[3]**2)
#    eD_edge = main.a.uD_edge[4]*rhoiD_edge - 0.5*rhoiD_edge**2*(main.a.uD_edge[1]**2 + main.a.uD_edge[2]**2 + main.a.uD_edge[3]**2)
#    eF_edge = main.a.uF_edge[4]*rhoiF_edge - 0.5*rhoiF_edge**2*(main.a.uF_edge[1]**2 + main.a.uF_edge[2]**2 + main.a.uF_edge[3]**2)
#    eB_edge = main.a.uB_edge[4]*rhoiB_edge - 0.5*rhoiB_edge**2*(main.a.uB_edge[1]**2 + main.a.uB_edge[2]**2 + main.a.uB_edge[3]**2)
    main.cgas_field.UVY = e.flatten(),rhoi.flatten(),fa
 #   main.cgas_field_R.UVY = eR.flatten(),rhoiR.flatten(),faR
 #   main.cgas_field_L.UVY = eL.flatten(),rhoiL.flatten(),faL
 #   main.cgas_field_U.UVY = eU.flatten(),rhoiU.flatten(),faU
 #   main.cgas_field_D.UVY = eD.flatten(),rhoiD.flatten(),faD
 #   main.cgas_field_F.UVY = eF.flatten(),rhoiF.flatten(),faF
 #   main.cgas_field_B.UVY = eB.flatten(),rhoiB.flatten(),faB
 #   main.cgas_field_R_edge.UVY = eR_edge.flatten(),rhoiR_edge.flatten(),faR_edge
 #   main.cgas_field_L_edge.UVY = eL_edge.flatten(),rhoiL_edge.flatten(),faL_edge
 #   main.cgas_field_U_edge.UVY = eU_edge.flatten(),rhoiU_edge.flatten(),faU_edge
 #   main.cgas_field_D_edge.UVY = eD_edge.flatten(),rhoiD_edge.flatten(),faD_edge
 #   main.cgas_field_F_edge.UVY = eF_edge.flatten(),rhoiF_edge.flatten(),faF_edge
 #   main.cgas_field_B_edge.UVY = eB_edge.flatten(),rhoiB_edge.flatten(),faB_edge

 
def computePressure_and_Temperature_Cantera(main,U,cgas_field):
  ## need to go from total energy to internal energy
  e = np.zeros(np.shape(U[4]))
  e[:] = U[4]
  rhoi = 1./U[0]
  e*= rhoi
  u,v,w = U[1]*rhoi, U[2]*rhoi, U[3]*rhoi 

  fa = np.zeros((np.size(U[0]),np.shape(U)[0]-5))
  for i in range(0,np.shape(U)[0]-5):
    fa[:,i] = ( U[5+i]*rhoi ).flatten()

  e -= 0.5*(u**2 + v**2 + w**2)
  cgas_field.UVY = e.flatten(),rhoi.flatten(),fa

  return np.reshape(cgas_field.P,np.shape(U[0])),np.reshape(cgas_field.T,np.shape(U[0]))

def computeEnergy(main,T,Y,u,v,w):
  R = 8.314
  T0 = 298.15
  n_reacting = np.size(main.delta_h0)
  Cv = np.einsum('i...,ijk...->jk...',main.Cv,Y)
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W,Y)
  e = Cv*(T - T0) - R*T0*Winv
  for i in range(0,np.size(main.delta_h0)):
    e += main.delta_h0[i]*Y[i]
  e += 0.5*(u**2 + v**2 + w**2)
  return e 

def computeTemperature(main,u):
  R = 8.314
  T0 = 298.15
  n_reacting = np.size(main.delta_h0)
  #Cv = 0
  #Winv = 0
  #for i in range(0,n_reacting):
  #  Cv += main.Cv[i]*u[5+i] #Cv of the mixture
  #  Winv += u[5+i]/main.W[i] #mean molecular weight
  Cv = np.einsum('i...,ijk...->jk...',main.Cv,u[5::]/u[0])
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W,u[5::]/u[0])
  # sensible + chemical
  T = u[4]/u[0] - 0.5/u[0]**2*( u[1]**2 + u[2]**2 + u[3]**2 )
  # subtract formation of enthalpy
  for i in range(0,np.size(main.delta_h0)):
    T -= main.delta_h0[i]*u[5+i]/u[0]
  T += R * T0 * Winv 
  T /= Cv
  T += T0

  return T


def computePressure(main,u,T):
  R = 8.314 
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W,u[5::])
  p = u[0]*R*Winv*T
  return p


def computePressure_and_Temperature(main,u):
  R = 8.314
  T0 = 298.15
  n_reacting = np.size(main.delta_h0)
  #Cv = 0
  #Winv = 0
  #for i in range(0,n_reacting):
  #  Cv += main.Cv[i]*u[5+i] #Cv of the mixture
  #  Winv += u[5+i]/main.W[i] #mean molecular weight
  Cv = np.einsum('i...,ijk...->jk...',main.Cv,u[5::]/u[0])
  Winv =  np.einsum('i...,ijk...->jk...',1./main.W,u[5::]/u[0])
  # sensible + chemical
  T = u[4]/u[0] - 0.5/u[0]**2*( u[1]**2 + u[2]**2 + u[3]**2 )
  # subtract formation of enthalpy
  for i in range(0,np.size(main.delta_h0)):
    T -= main.delta_h0[i]*u[5+i]
  T += R * T0 * Winv 
  T /= Cv
  T += T0
  p = u[0]*R*Winv*T
  return p,T

