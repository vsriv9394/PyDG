import numpy as np
import numexpr as ne
def applyMassMatrix(main,RHS):
  RHS[:] = np.einsum('ijklpqrs...,zpqrs...->zijkl...',main.Minv,RHS)

def applyMassMatrix_orthogonal(main,RHS):
  ord_arr0= np.linspace(0,main.order[0]-1,main.order[0])
  ord_arr1= np.linspace(0,main.order[1]-1,main.order[1])
  ord_arr2= np.linspace(0,main.order[2]-1,main.order[2])
  ord_arr3= np.linspace(0,main.order[3]-1,main.order[3])
  scale =  (2.*ord_arr0[:,None,None,None] + 1.)*(2.*ord_arr1[None,:,None,None] + 1.)*(2.*ord_arr2[None,None,:,None]+1.)*(2.*ord_arr3[None,None,None,:] + 1. )/16.
  RHS[:] = RHS*scale[None,:,:,:,:,None,None,None,None]/main.Jdet[None,0,0,0,None,None,None,None,:,:,:,None]

def applyVolIntegral(main,f1,f2,f3,RHS):
  f = f1*main.Jinv[0,0][None,:,:,:,None,:,:,:,None]
  f += f2*main.Jinv[0,1][None,:,:,:,None,:,:,:,None]
  f += f3*main.Jinv[0,2][None,:,:,:,None,:,:,:,None]
  f *= main.Jdet[None,:,:,:,None,:,:,:,None]
  RHS[:] += main.basis.volIntegrateGlob(main,f,main.wp0,main.w1,main.w2,main.w3)


  f = f1*main.Jinv[1,0][None,:,:,:,None,:,:,:,None]
  f += f2*main.Jinv[1,1][None,:,:,:,None,:,:,:,None]
  f += f3*main.Jinv[1,2][None,:,:,:,None,:,:,:,None]
  f *= main.Jdet[None,:,:,:,None,:,:,:,None]
  RHS[:] += main.basis.volIntegrateGlob(main,f,main.w0,main.wp1,main.w2,main.w3)

  f = f1*main.Jinv[2,0][None,:,:,:,None,:,:,:,None]
  f += f2*main.Jinv[2,1][None,:,:,:,None,:,:,:,None]
  f += f3*main.Jinv[2,2][None,:,:,:,None,:,:,:,None]
  f *= main.Jdet[None,:,:,:,None,:,:,:,None]
  RHS[:] += main.basis.volIntegrateGlob(main,f,main.w0,main.w1,main.wp2,main.w3)

def applyVolIntegral_numexpr(main,f1,f2,f3,RHS):
  J1 = main.Jinv[0,0][None,:,:,:,None,:,:,:,None]
  J2 = main.Jinv[0,1][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[0,2][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[0,2][None,:,:,:,None,:,:,:,None]
  JD = main.Jdet[None,:,:,:,None,:,:,:,None]
  f = ne.evaluate("JD*(f1*J1 + f2*J2 + f3*J3)")
  f = main.basis.volIntegrateGlob(main,f,main.wp0,main.w1,main.w2,main.w3)
  ne.evaluate("RHS+f",out=RHS)

  J1 = main.Jinv[1,0][None,:,:,:,None,:,:,:,None]
  J2 = main.Jinv[1,1][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[1,2][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[1,2][None,:,:,:,None,:,:,:,None]
  #f = ne.re_evaluate()#"JD*(a*J1 + b*J2 + c*J3)")
  f = ne.evaluate("JD*(f1*J1 + f2*J2 + f3*J3)")
  f = main.basis.volIntegrateGlob(main,f,main.w0,main.wp1,main.w2,main.w3)
  ne.evaluate("RHS+f",out=RHS)

  J1 = main.Jinv[2,0][None,:,:,:,None,:,:,:,None]
  J2 = main.Jinv[2,1][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[2,2][None,:,:,:,None,:,:,:,None]
  J3 = main.Jinv[2,2][None,:,:,:,None,:,:,:,None]
  #f = ne.re_evaluate()#"JD*(a*J1 + b*J2 + c*J3)")
  f = ne.evaluate("JD*(f1*J1 + f2*J2 + f3*J3)")
  f = main.basis.volIntegrateGlob(main,f,main.w0,main.w1,main.wp2,main.w3)
  ne.evaluate("RHS+f",out=RHS)

def applyVolIntegral_numexpr_orthogonal(main,f1,f2,f3,RHS):
  J1 = main.Jinv[0,0][None,:,:,:,None,:,:,:,None]
  JD = main.Jdet[None,:,:,:,None,:,:,:,None]
  f = ne.evaluate("JD*(f1*J1)")
  RHS[:] += main.basis.volIntegrateGlob(main,f,main.wp0,main.w1,main.w2,main.w3)

  J2 = main.Jinv[1,1][None,:,:,:,None,:,:,:,None]
  f = ne.evaluate("JD*(f2*J2)")
  RHS[:] += main.basis.volIntegrateGlob(main,f,main.w0,main.wp1,main.w2,main.w3)

  J3 = main.Jinv[2,2][None,:,:,:,None,:,:,:,None]
  f = ne.evaluate("JD*(f3*J3)")
  RHS[:] += main.basis.volIntegrateGlob(main,f,main.w0,main.w1,main.wp2,main.w3)


def diffU_tensordot(a,main):
  tmp = np.tensordot(a,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzeta = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)
  tmpu = np.tensordot(a,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  ueta = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

  #tmp = np.tensordot(a,main.w,axes=([1],[0]))
  tmp = np.tensordot(tmpu,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  umu = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

#  uzeta *= 2./main.dx2[None,None,None,None,None,:,None,None,None]
#  ueta *= 2./main.dy2[None,None,None,None,None,None,:,None,None]
#  umu *= 2./main.dz2[None,None,None,None,None,None,None,:,None]
  
  ux = uzeta*main.Jinv[0,0][None,:,:,:,None,:,:,:,None] + ueta*main.Jinv[1,0][None,:,:,:,None,:,:,:,None] + umu*main.Jinv[2,0][None,:,:,:,None,:,:,:,None]
  uy = uzeta*main.Jinv[0,1][None,:,:,:,None,:,:,:,None] + ueta*main.Jinv[1,1][None,:,:,:,None,:,:,:,None] + umu*main.Jinv[2,1][None,:,:,:,None,:,:,:,None]
  uz = uzeta*main.Jinv[0,2][None,:,:,:,None,:,:,:,None] + ueta*main.Jinv[1,2][None,:,:,:,None,:,:,:,None] + umu*main.Jinv[2,2][None,:,:,:,None,:,:,:,None]

  return ux,uy,uz



def diffUXEdge_edge_tensordot(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge*main.wpedge0[:,1][None,:,None,None,None,None,None,None])
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.wpedge0[:,0][None,:,None,None,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray1[None,None,:,None,None,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray2[None,None,None,:,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxR = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxL = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aU,main.wp0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxU = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxD = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aF,main.wp0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  return uxR,uxL,uxU,uxD,uxF,uxB

def diffUYEdge_edge_tensordot(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray0[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge*main.wpedge1[:,1][None,None,:,None,None,None])
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.wpedge1[:,0][None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray2[None,None,None,:,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.wp1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))*2./main.dx
  uyD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))*2./main.dx
  uyF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))*2./main.dx
  uyB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uyR,uyL,uyU,uyD,uyF,uyB


def diffUZEdge_edge_tensordot(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray0[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray1[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge*main.wpedge2[:,1][None,None,None,:,None,None])
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.wpedge2[:,0][None,None,None,:,None,None])


#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))*2./main.dx
  uzD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uzF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))*2./main.dx
  uzB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uzR,uzL,uzU,uzD,uzF,uzB



def diffUX_edge_tensordot(a,main):
  aR = np.einsum('zpqr...->zqr...',a*main.wpedge0[:,1][None,:,None,None,None,None,None,None,None])
  aL = np.einsum('zpqr...->zqr...',a*main.wpedge0[:,0] [None,:,None,None,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]

  tmp = np.tensordot(aU,main.wp0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]
  tmp = np.tensordot(aD,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]

  tmp = np.tensordot(aF,main.wp0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]
  tmp = np.tensordot(aB,main.wp0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uxB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dx2[None,None,None,None,:,None,None,None]
  return uxR,uxL,uxU,uxD,uxF,uxB

def diffUY_edge_tensordot(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a*main.wpedge1[:,1][None,None,:,None,None,None,None,None,None])
  aD = np.einsum('zpqr...->zpr...',a*main.wpedge1[:,0][None,None,:,None,None,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.wp1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]
  tmp = np.tensordot(aL,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uyB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dy2[None,None,None,None,None,:,None,None]
  return uyR,uyL,uyU,uyD,uyF,uyB



def diffUZ_edge_tensordot(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a*main.wpedge2[:,1][None,None,None,:,None,None,None,None,None])
  aB = np.einsum('zpqr...->zpq...',a*main.wpedge2[:,0][None,None,None,:,None,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzR = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzL = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzU = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.wp2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzD = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzF = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uzB = np.rollaxis(np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)*2./main.dz2[None,None,None,None,None,None,:,None]
  return uzR,uzL,uzU,uzD,uzF,uzB




def diffU_einsum(a,main):
  tmp =  np.einsum('rn,zpqr...->zpqn...',main.w,a) #reconstruct along third axis 
  tmp2 = np.einsum('qm,zpqn...->zpmn...',main.w,tmp) #reconstruct along second axis
  ux = np.einsum('pl,zpmn...->zlmn...',main.wp,tmp2) # get ux by differentiating along the first axis

  tmp2 = np.einsum('qm,zpqn...->zpmn...',main.wp,tmp) #diff tmp along second axis
  uy = np.einsum('pl,zpmn...->zlmn...',main.w,tmp2) # get uy by reconstructing along the first axis

  tmp =  np.einsum('rn,zpqr...->zpqn...',main.wp,a) #diff along third axis 
  tmp = np.einsum('qm,zpqn...->zpmn...',main.w,tmp) #reconstruct along second axis
  uz = np.einsum('pl,zpmn...->zlmn...',main.w,tmp) # reconstruct along the first axis

  ux *= 2./main.dx2[None,None,None,None,:,None,None]
  uy *= 2./main.dy2[None,None,None,None,None,:,None]
  uz *= 2./main.dz2[None,None,None,None,None,None,:]

  return ux,uy,uz

def diffUXEdge_edge_einsum(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge*main.wpedge[:,1][None,:,None,None,None,None])
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.wpedge[:,0][None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray[None,None,None,:,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face

  tmp = np.einsum('rn,zqr...->zqn...',main.w,aR)  #reconstruct in y and z
  uxR  = np.einsum('qm,zqn...->zmn...',main.w,tmp)*2./main.dx 
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aL)
  uxL  = np.einsum('qm,zqn...->zmn...',main.w,tmp)*2./main.dx

  tmp = np.einsum('rn,zpr...->zpn...',main.w,aU) #reconstruct in x and z 
  uxU  = np.einsum('pl,zpn...->zln...',main.wp,tmp)*2./main.dx
  tmp = np.einsum('rn,zpr...->zpn...',main.w,aD)
  uxD  = np.einsum('pl,zpn...->zln...',main.wp,tmp)*2./main.dx

  tmp = np.einsum('qm,zpq...->zpm...',main.w,aF) #reconstruct in x and y
  uxF  = np.einsum('pl,zpm...->zlm...',main.wp,tmp)*2./main.dx
  tmp = np.einsum('qm,zpq...->zpm...',main.w,aB)
  uxB  = np.einsum('pl,zpm...->zlm...',main.wp,tmp)*2./main.dx
  return uxR,uxL,uxU,uxD,uxF,uxB


def diffUX_edge_einsum(a,main):
  aR = np.einsum('zpqr...->zqr...',a*main.wpedge[:,1][None,:,None,None,None,None,None])
  aL = np.einsum('zpqr...->zqr...',a*main.wpedge[:,0] [None,:,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray[None,None,:,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.w2,aR)  #reconstruct in y and z
  uxR  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dx 
  tmp = np.einsum('rn,zqr...->zqn...',main.w2,aL)
  uxL  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dx

  tmp = np.einsum('rn,zpr...->zpn...',main.w2,aU) #reconstruct in x and z 
  uxU  = np.einsum('pl,zpn...->zln...',main.wp0,tmp)*2./main.dx
  tmp = np.einsum('rn,zpr...->zpn...',main.w2,aD)
  uxD  = np.einsum('pl,zpn...->zln...',main.wp0,tmp)*2./main.dx

  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aF) #reconstruct in x and y
  uxF  = np.einsum('pl,zpm...->zlm...',main.wp0,tmp)*2./main.dx
  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aB)
  uxB  = np.einsum('pl,zpm...->zlm...',main.wp0,tmp)*2./main.dx
  return uxR,uxL,uxU,uxD,uxF,uxB

def diffUYEdge_edge_einsum(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge*main.wpedge[:,1][None,None,:,None,None,None])
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.wpedge[:,0][None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray[None,None,None,:,None,None])
#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aR)  #reconstruct in y and z
  uyR  = np.einsum('qm,zqn...->zmn...',main.wp,tmp)*2./main.dy
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aL)
  uyL  = np.einsum('qm,zqn...->zmn...',main.wp,tmp)*2./main.dy

  tmp = np.einsum('rn,zpr...->zpn...',main.w,aU) #recounstruct in x and z
  uyU  = np.einsum('pl,zpn...->zln...',main.w,tmp)*2./main.dy
  tmp = np.einsum('rn,zpr...->zpn...',main.w,aD)
  uyD  = np.einsum('pl,zpn...->zln...',main.w,tmp)*2./main.dy

  tmp = np.einsum('qm,zpq...->zpm...',main.wp,aF) #reconstruct in x and y
  uyF  = np.einsum('pl,zpm...->zlm...',main.w,tmp)*2./main.dy
  tmp = np.einsum('qm,zpq...->zpm...',main.wp,aB)
  uyB  = np.einsum('pl,zpm...->zlm...',main.w,tmp)*2./main.dy
  return uyR,uyL,uyU,uyD,uyF,uyB




def diffUY_edge_einsum(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray[None,:,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a*main.wpedge[:,1][None,None,:,None,None,None,None])
  aD = np.einsum('zpqr...->zpr...',a*main.wpedge[:,0][None,None,:,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.w2,aR)  #reconstruct in y and z
  uyR  = np.einsum('qm,zqn...->zmn...',main.wp1,tmp)*2./main.dy
  tmp = np.einsum('rn,zqr...->zqn...',main.w2,aL)
  uyL  = np.einsum('qm,zqn...->zmn...',main.wp1,tmp)*2./main.dy

  tmp = np.einsum('rn,zpr...->zpn...',main.w2,aU) #recounstruct in x and z
  uyU  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dy
  tmp = np.einsum('rn,zpr...->zpn...',main.w2,aD)
  uyD  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dy

  tmp = np.einsum('qm,zpq...->zpm...',main.wp1,aF) #reconstruct in x and y
  uyF  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dy
  tmp = np.einsum('qm,zpq...->zpm...',main.wp1,aB)
  uyB  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dy
  return uyR,uyL,uyU,uyD,uyF,uyB



def diffUZEdge_edge_einsum(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge*main.wpedge[:,1][None,None,None,:,None,None])
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.wpedge[:,0][None,None,None,:,None,None])

  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.wp2,aR)  #reconstruct in y and z
  uzR  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dz 
  tmp = np.einsum('rn,zqr...->zqn...',main.wp2,aL)
  uzL  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dz

  tmp = np.einsum('rn,zpr...->zpn...',main.wp2,aU) #recounstruct in x and z
  uzU  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dz
  tmp = np.einsum('rn,zpr...->zpn...',main.wp2,aD)
  uzD  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dz

  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aF) #reconstruct in x and y
  uzF  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dz
  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aB)
  uzB  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dz
  return uzR,uzL,uzU,uzD,uzF,uzB




def diffUZ_edge_einsum(a,main):
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray[None,:,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray[None,None,:,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a*main.wpedge[:,1][None,None,None,:,None,None,None])
  aB = np.einsum('zpqr...->zpq...',a*main.wpedge[:,0][None,None,None,:,None,None,None])

  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.wp2,aR)  #reconstruct in y and z
  uzR  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dz 
  tmp = np.einsum('rn,zqr...->zqn...',main.wp2,aL)
  uzL  = np.einsum('qm,zqn...->zmn...',main.w1,tmp)*2./main.dz

  tmp = np.einsum('rn,zpr...->zpn...',main.wp2,aU) #recounstruct in x and z
  uzU  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dz
  tmp = np.einsum('rn,zpr...->zpn...',main.wp2,aD)
  uzD  = np.einsum('pl,zpn...->zln...',main.w0,tmp)*2./main.dz

  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aF) #reconstruct in x and y
  uzF  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dz
  tmp = np.einsum('qm,zpq...->zpm...',main.w1,aB)
  uzB  = np.einsum('pl,zpm...->zlm...',main.w0,tmp)*2./main.dz
  return uzR,uzL,uzU,uzD,uzF,uzB




def diffCoeffs(a):
  atmp = np.zeros(np.shape(a))
  atmp[:] = a[:]
  nvars,orderx,ordery,orderz,ordert,Nelx,Nely,Nelz,Nelt = np.shape(a)
  order = np.array([orderx,ordery,orderz,ordert])
  ax = np.zeros((nvars,order[0],order[1],order[2],Nelx,Nely,Nelz,Nelt))
  ay = np.zeros((nvars,order[0],order[1],order[2],Nelx,Nely,Nelz,Nelt))
  az = np.zeros((nvars,order[0],order[1],order[2],Nelx,Nely,Nelz,Nelt))

  for j in range(order-1,2,-1):
    ax[:,j-1,:,:] = (2.*j-1)*atmp[:,j,:,:]
    atmp[:,j-2,:,:] = atmp[:,j-2,:,:] + atmp[:,j,:,:]

  if (order >= 3):
    ax[:,1,:,:] = 3.*atmp[:,2,:,:]
    ax[:,0,:,:] = atmp[:,1,:,:]
  if (order == 2):
    ax[:,1,:,:] = 0.
    ax[:,0,:,:] = atmp[:,1,:,:]
  if (order == 1):
    ax[:,0,:,:] = 0.

  atmp[:] = a[:]
  for j in range(order-1,2,-1):
    ay[:,:,j-1,:] = (2.*j-1)*atmp[:,:,j,:]
    atmp[:,:,j-2,:] = atmp[:,:,j-2,:] + atmp[:,:,j,:]

  if (order >= 3):
    ay[:,:,1,:] = 3.*atmp[:,:,2,:]
    ay[:,:,0,:] = atmp[:,:,1,:]
  if (order == 2):
    ay[:,:,1,:] = 0.
    ay[:,:,0,:] = atmp[:,:,1,:]
  if (order == 1):
    ay[:,:,0,:] = 0.


  atmp[:] = a[:]
  for j in range(order-1,2,-1):
    az[:,:,:,j-1] = (2.*j-1)*atmp[:,:,:,j]
    atmp[:,:,:,j-2] = atmp[:,:,:,j-2] + atmp[:,:,:,j]
  if (order >= 3):
    az[:,:,:,1] = 3.*atmp[:,:,:,2]
    az[:,:,:,0] = atmp[:,:,:,1]
  if (order == 2):
    az[:,:,:,1] = 0.
    az[:,:,:,0] = atmp[:,:,:,1]
  if (order == 1):
    az[:,:,:,0] = 0.


  return ax,ay,az


def volIntegrate(weights0,weights1,weights2,weights3,f):
  return  np.einsum('zpqrl...->z...',weights0[None,:,None,None,None,None,None,None,None]*weights1[None,None,:,None,None,None,None,None,None]*\
          weights2[None,None,None,:,None,None,None,None,None]*weights3[None,None,None,None,:,None,None,None,None]*f)


def volIntegrateGlob_tensordot3(main,f,w0,w1,w2,w3):
  tmp = np.tensordot(main.weights0[None,:,None,None,None,None,None,None,None]*f,w0,axes=([1],[1]))
  tmp = np.tensordot(main.weights1[None,:,None,None,None,None,None,None,None]*tmp,w1,axes=([1],[1]))
  tmp = np.tensordot(main.weights2[None,:,None,None,None,None,None,None,None]*tmp,w2,axes=([1],[1]))
  tmp = np.tensordot(main.weights3[None,:,None,None,None,None,None,None,None]*tmp,w3,axes=([1],[1]))
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

def volIntegrateGlob_tensordot2(main,f,w0,w1,w2,w3):
  tmp = np.tensordot(f,w0*main.weights0[None,:],axes=([1],[1]))
  tmp = np.tensordot(tmp,w1*main.weights1[None,:],axes=([1],[1]))
  tmp = np.tensordot(tmp,w2*main.weights2[None,:],axes=([1],[1]))
  tmp = np.tensordot(tmp,w3*main.weights3[None,:],axes=([1],[1]))
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

def volIntegrateGlob_tensordot(main,f,w0,w1,w2,w3):
  tmp = np.rollaxis(np.tensordot(w0*main.weights0[None,:],f,axes=([1],[1])),0,9)
  tmp = np.rollaxis(np.tensordot(w1*main.weights1[None,:],tmp,axes=([1],[1])) , 0 , 9)
  tmp = np.rollaxis(np.tensordot(w2*main.weights2[None,:],tmp,axes=([1],[1])) , 0 , 9)
  tmp = np.rollaxis(np.tensordot(w3*main.weights3[None,:],tmp,axes=([1],[1])) , 0 , 9)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)


def volIntegrateGlob_tensordot_collocate(main,f,w0,w1,w2,w3):
  tmp = np.tensordot(main.weights0_c[None,:,None,None,None,None,None,None,None]*f  ,w0,axes=([1],[1]))
  tmp = np.tensordot(main.weights1_c[None,:,None,None,None,None,None,None,None]*tmp,w1,axes=([1],[1]))
  tmp = np.tensordot(main.weights1_c[None,:,None,None,None,None,None,None,None]*tmp,w2,axes=([1],[1]))
  tmp = np.tensordot(main.weights3_c[None,:,None,None,None,None,None,None,None]*tmp,w3,axes=([1],[1]))
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

def volIntegrateGlob_einsumMM2(main,f,w0,w1,w2,w3):
  tmp = np.einsum('dos...,zpqrs...->zpqrdo...',w3[:,None]*w3[None,:]*main.weights3[None,None,:],f)
  tmp = np.einsum('cnr...,zpqrdo...->zpqcndo...',w2[:,None]*w2[None,:]*main.weights2[None,None,:],tmp)
  tmp = np.einsum('bmq...,zpqcndo...->zpbmcndo...',w1[:,None]*w1[None,:]*main.weights1[None,None,:],tmp)
  tmp = np.einsum('alp...,zpbmcndo...->zalbmcndo...',w0[:,None]*w0[None,:]*main.weights0[None,None,:],tmp)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp, 7,9) , 5,8) ,3 , 7) , 1 , 6)

def volIntegrateGlob_einsumMM3(main,f,w0,w1,w2,w3):
  tmp = np.einsum('os...,zdpqras...->zdpqrao...',w3[None,:]*main.weights3[None,None,:],f[:,:,:,:,None]*main.w3[None,None,None,None,:,:,None,None,None,None])
  tmp = np.einsum('nr...,zdpqro...->zcdpqno...',w2[:,None]*w2[None,:]*main.weights2[None,None,:],tmp)
  tmp = np.einsum('mq...,zcdpqno...->zbcdpmno...',w1[:,None]*w1[None,:]*main.weights1[None,None,:],tmp)
  return np.einsum('lp...,zbcdpmno...->zabcdlmno...',w0[:,None]*w0[None,:]*main.weights0[None,None,:],tmp)


def volIntegrateGlob_einsumMM(main,f,w0,w1,w2,w3):
  tmp = np.einsum('os...,zabcdpqrs...->zabcdpqro...',w3*main.weights3[None,:],f)
  tmp = np.einsum('nr...,zabcdpqro...->zabcdpqno...',w2*main.weights2[None,:],tmp)
  tmp = np.einsum('mq...,zabcdpqno...->zabcdpmno...',w1*main.weights1[None,:],tmp)
  return np.einsum('lp...,zabcdpmno...->zabcdlmno...',w0*main.weights0[None,:],tmp)

def volIntegrateGlob_tensordotMM(main,f,w0,w1,w2,w3):
  tmp = np.rollaxis(np.tensordot(w0*main.weights0[None,:],f,axes=([1],[5])),0,13)
  tmp = np.rollaxis(np.tensordot(w1*main.weights1[None,:],tmp,axes=([1],[5])) , 0 , 13)
  tmp = np.rollaxis(np.tensordot(w2*main.weights2[None,:],tmp,axes=([1],[5])) , 0 , 13)
  tmp = np.rollaxis(np.tensordot(w3*main.weights3[None,:],tmp,axes=([1],[5])) , 0 , 13)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 5) , -3 , 6), -2, 7), -1, 8)


def volIntegrateGlob_einsum(main,f,w0,w1,w2,w3):
  tmp = np.einsum('os...,zpqrs...->zpqro...',w3,main.weights3[None,None,None,None,:,None,None,None,None]*f)
  tmp = np.einsum('nr...,zpqro...->zpqno...',w2,main.weights2[None,None,None,:,None,None,None,None,None]*tmp)
  tmp = np.einsum('mq...,zpqno...->zpmno...',w1,main.weights1[None,None,:,None,None,None,None,None,None]*tmp)
  return np.einsum('lp...,zpmno...->zlmno...',w0,main.weights0[None,:,None,None,None,None,None,None,None]*tmp)

def volIntegrateGlob_einsum_2(main,f):
#  weights = main.weights0[None,None,None,None,None,:,None,None,None,None,None,None,None]*main.weights1[None,None,None,None,None,None,:,None,None,None,None,None,None]*\
#            main.weights1[None,None,None,None,None,None,None,:,None,None,None,None,None]*main.weights3[None,None,None,None,None,None,None,None,:,None,None,None,None]
  weights = main.weights0[:,None,None,None]*main.weights1[None,:,None,None]*main.weights2[None,None,:,None]*main.weights3[None,None,None,:]
  tmp = np.einsum('pqrs,zlmnopqrs...->zlmno...',weights,f)
#  tmp = np.einsum('s,zlmnopqrs...->zlmnopqr...',main.weights3,f)
#  tmp = np.einsum('r,zlmnopqr...->zlmnopq...',main.weights2,tmp)
#  tmp = np.einsum('q,zlmnopq...->zlmnop...',main.weights1,tmp)
#  tmp = np.einsum('p,zlmnop...->zlmno...',main.weights0,tmp)

  return tmp


def faceIntegrate(weights,f):
  return np.einsum('zpqijk->zijk',weights[None,:,None,None,None,None]*weights[None,None,:,None,None,None]*f)

def faceIntegrateGlob_tensordot(main,f,w1,w2,w3,weights1,weights2,weights3):
  tmp = np.tensordot(f,w1*weights1[None,:],axes=([1],[1]))
  tmp = np.tensordot(tmp,w2*weights2[None,:],axes=([1],[1]))
  tmp = np.tensordot(tmp,w3*weights3[None,:],axes=([1],[1]))
  #return np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5)
  return np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2),-1,3)


def faceIntegrateGlob_einsum(main,f,w1,w2,weights0,weights1):
  tmp = np.einsum('nr,zqrijk->zqnijk',w2,weights1[None,None,:,None,None,None]*f)
  return np.einsum('mq,zqnijk->zmnijk',w1,weights0[None,:,None,None,None,None]*tmp)



def reconstructU_einsum(main,var):
  var.u[:] = 0.
  #var.u = np.einsum('lmpq,klmij->kpqij',main.w[:,None,:,None]*main.w[None,:,None,:],var.a) ## this is actually much slower than the two line code
  tmp =  np.einsum('so,zpqrs...->zpqro...',main.w3,var.a)
  tmp =  np.einsum('rn,zpqro...->zpqno...',main.w2,var.a)
  tmp = np.einsum('qm,zpqno...->zpmno...',main.w1,tmp)
  var.u = np.einsum('pl,zpmno...->zlmno...',main.w0,tmp)

def reconstructUGeneral_einsum(main,a):
  tmp =  np.einsum('so,zpqrs...->zpqro...',main.w3,a)
  tmp =  np.einsum('rn,zpqro...->zpqno...',main.w2,a)
  tmp = np.einsum('qm,zpqno...->zpmno...',main.w1,tmp)
  return np.einsum('pl,zpmno...->zlmno...',main.w0,tmp)


def reconstructU_tensordot2(main,var):
  var.u[:] = 0.
  tmp = np.tensordot(var.a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0])) 
#  var.u = np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  var.u = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)

def reconstructU_tensordot(main,var):
  var.u[:] = 0.
  tmp = np.rollaxis(np.tensordot(main.w0,var.a,axes=([0],[1])) ,0,9)
  tmp = np.rollaxis(np.tensordot(main.w1,tmp,axes=([0],[1])) ,0,9)
  tmp = np.rollaxis(np.tensordot(main.w2,tmp,axes=([0],[1])) ,0,9)
  tmp = np.rollaxis(np.tensordot(main.w3,tmp,axes=([0],[1])) ,0,9)
#  var.u = np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  var.u = np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4)


def reconstructUGeneral_tensordot(main,a):
  tmp = np.tensordot(a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0])) 
  #return np.swapaxes( np.swapaxes( np.swapaxes( tmp , 1 , 4) , 2 , 5), 3, 6)
  return np.rollaxis( np.rollaxis( np.rollaxis( np.rollaxis( tmp , -4 , 1) , -3 , 2), -2, 3), -1, 4 )




def reconstructEdgesGeneral_tensordot(a,main):
  nvars = np.shape(a)[0]
  #aR = np.einsum('zpqr...->zqr...',a)
  aR = np.sum(a,axis=1)
  #aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None])
  aL = np.tensordot(main.altarray0,a,axes=([0],[1]) )

  #aU = np.einsum('zpqr...->zpr...',a)
  aU = np.sum(a,axis=2)
  #aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None])
  aD = np.tensordot(main.altarray1,a,axes=([0],[2]) )

#  aF = np.einsum('zpqr...->zpq...',a)
  aF = np.sum(a,axis=3)
#  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None])
  aB = np.tensordot(main.altarray2,a,axes=([0],[3]) )

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uR = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uL = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uU = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uD = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uF = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w3,axes=([1],[0]))
  uB = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  return uR,uL,uU,uD,uF,uB


def reconstructEdgesGeneralTime_tensordot(a,main):
  nvars = np.shape(a)[0]
  aFuture = np.einsum('zpqrl...->zpqr...',a)
  aPast = np.tensordot(main.altarray3,a,axes=([0],[4]) )

  tmp = np.tensordot(aFuture,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uFuture = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  tmp = np.tensordot(aPast,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uPast = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  return uFuture,uPast


def reconstructEdgesGeneral_einsum(a,main):
  nvars = np.shape(a)[0]
  aR = np.einsum('zpqr...->zqr...',a)
  aL = np.einsum('zpqr...->zqr...',a*main.altarray0[None,:,None,None,None,None,None,None,None])

  aU = np.einsum('zpqr...->zpr...',a)
  aD = np.einsum('zpqr...->zpr...',a*main.altarray1[None,None,:,None,None,None,None,None,None])

  aF = np.einsum('zpqr...->zpq...',a)
  aB = np.einsum('zpqr...->zpq...',a*main.altarray2[None,None,None,:,None,None,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('so,zqrs...->zqro...',main.w3,aR)
  tmp = np.einsum('rn,zqro...->zqno...',main.w2,tmp)
  uR  = np.einsum('qm,zqno...->zmno...',main.w1,tmp)
  tmp = np.einsum('so,zqrs...->zqro...',main.w3,aL)
  tmp = np.einsum('rn,zqro...->zqno...',main.w2,tmp)
  uL  = np.einsum('qm,zqno...->zmno...',main.w1,tmp)

  tmp = np.einsum('so,zprs...->zpro...',main.w3,aU)
  tmp = np.einsum('rn,zpro...->zpno...',main.w2,tmp)
  uU  = np.einsum('pl,zpno...->zlno...',main.w0,tmp)
  tmp = np.einsum('so,zprs...->zpro...',main.w3,aD)
  tmp = np.einsum('rn,zpro...->zpno...',main.w2,tmp)
  uD  = np.einsum('pl,zpno...->zlno...',main.w0,tmp)

  tmp = np.einsum('so,zpqs...->zpqo...',main.w3,aF)
  tmp = np.einsum('qm,zpqo...->zpmo...',main.w1,tmp)
  uF  = np.einsum('pl,zpmo...->zlmo...',main.w0,tmp)
  tmp = np.einsum('so,zpqs...->zpqo...',main.w3,aB)
  tmp = np.einsum('qm,zpqo...->zpmo...',main.w1,tmp)
  uB  = np.einsum('pl,zpmo...->zlmo...',main.w0,tmp)
  return uR,uL,uU,uD,uF,uB


def reconstructEdgeEdgesGeneral_tensordot(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray0[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray1[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray2[None,None,None,:,None,None])
#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uR = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uL = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uU = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uD = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uF = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uB = np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2)
  return uR,uL,uU,uD,uF,uB

def reconstructEdgeEdgesGeneral_einsum(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray[None,None,None,:,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aR)
  uR  = np.einsum('qm,zqn...->zmn...',main.w,tmp)
  tmp = np.einsum('rn,zqr...->zqn...',main.w,aL)
  uL  = np.einsum('qm,zqn...->zmn...',main.w,tmp)

  tmp = np.einsum('rn,zpr...->zpn...',main.w,aU)
  uU  = np.einsum('pl,zpn...->zln...',main.w,tmp)
  tmp = np.einsum('rn,zpr...->zpn...',main.w,aD)
  uD  = np.einsum('pl,zpn...->zln...',main.w,tmp)

  tmp = np.einsum('qm,zpq...->zpm...',main.w,aF)
  uF  = np.einsum('pl,zpm...->zlm...',main.w,tmp)
  tmp = np.einsum('qm,zpq...->zpm...',main.w,aB)
  uB  = np.einsum('pl,zpm...->zlm...',main.w,tmp)
  return uR,uL,uU,uD,uF,uB





































def reconstructU_entropy(main,var):
  var.u[:] = 0.
  tmp = np.tensordot(var.a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
  v = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  var.u[:] = entropy_to_conservative(v)


def reconstructUGeneral_entropy(main,a):
  tmp = np.tensordot(a,main.w0,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0])) 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0])) 
  v = np.rollaxis( np.rollaxis( np.rollaxis( tmp , -3 , 1) , -2 , 2), -1, 3)
  ## This is the same, but first we need to make the transformation from entropy variables 
  ## to conservative variables (adopted from murman)
  return entropy_to_conservative(v)






def reconstructEdgesGeneral_entropy(a,main):
  nvars = np.shape(a)[0]
  aR = np.einsum('zpqrijk->zqrijk',a)
  aL = np.einsum('zpqrijk->zqrijk',a*main.altarray0[None,:,None,None,None,None,None])

  aU = np.einsum('zpqrijk->zprijk',a)
  aD = np.einsum('zpqrijk->zprijk',a*main.altarray1[None,None,:,None,None,None,None])

  aF = np.einsum('zpqrijk->zpqijk',a)
  aB = np.einsum('zpqrijk->zpqijk',a*main.altarray2[None,None,None,:,None,None,None])

#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uR = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uL = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uU = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uD = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uF = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uB = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  return uR,uL,uU,uD,uF,uB


def reconstructEdgeEdgesGeneral_entropy(main):
  aL = np.einsum('zpqr...->zqr...',main.a.aL_edge)
  aR = np.einsum('zpqr...->zqr...',main.a.aR_edge*main.altarray0[None,:,None,None,None,None])

  aD = np.einsum('zpqr...->zpr...',main.a.aD_edge)
  aU = np.einsum('zpqr...->zpr...',main.a.aU_edge*main.altarray1[None,None,:,None,None,None])

  aB = np.einsum('zpqr...->zpq...',main.a.aB_edge)
  aF = np.einsum('zpqr...->zpq...',main.a.aF_edge*main.altarray2[None,None,None,:,None,None])
#  uU = np.zeros((main.nvars,main.quadpoints,main.quadpoints,main.Npx,main.Npy,main.Npz))
  # need to do 2D reconstruction to the gauss points on each face
  tmp = np.tensordot(aR,main.w1,axes=([1],[0])) #reconstruct in y and z
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uR = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aL,main.w1,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uL = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))

  tmp = np.tensordot(aU,main.w0,axes=([1],[0])) #reconstruct in x and z 
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uU = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aD,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w2,axes=([1],[0]))
  uD = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))

  tmp = np.tensordot(aF,main.w0,axes=([1],[0])) #reconstruct in x and y 
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uF = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  tmp = np.tensordot(aB,main.w0,axes=([1],[0]))
  tmp = np.tensordot(tmp,main.w1,axes=([1],[0]))
  uB = entropy_to_conservative(np.rollaxis( np.rollaxis( tmp , -2 , 1) , -1 , 2))
  return uR,uL,uU,uD,uF,uB
























def reconstructEdgesGeneral_tensordot_linRecon(a,main):

  du_r = np.zeros(main.a.u.shape,dtype='float64')  
  du_l = np.zeros(main.a.u.shape,dtype='float64')  
  du_u = np.zeros(main.a.u.shape,dtype='float64')  
  du_d = np.zeros(main.a.u.shape,dtype='float64')  
  du_f = np.zeros(main.a.u.shape,dtype='float64')  
  du_b = np.zeros(main.a.u.shape,dtype='float64')

  dx_r = main.LSQfactors[0,0,0,:,:,:]
  dx_l = main.LSQfactors[1,0,0,:,:,:]
  dx_u = main.LSQfactors[2,0,0,:,:,:]
  dx_d = main.LSQfactors[3,0,0,:,:,:]
  dx_f = main.LSQfactors[4,0,0,:,:,:]
  dx_b = main.LSQfactors[5,0,0,:,:,:]

  dy_r = main.LSQfactors[0,1,0,:,:,:]
  dy_l = main.LSQfactors[1,1,0,:,:,:]
  dy_u = main.LSQfactors[2,1,0,:,:,:]
  dy_d = main.LSQfactors[3,1,0,:,:,:]
  dy_f = main.LSQfactors[4,1,0,:,:,:]
  dy_b = main.LSQfactors[5,1,0,:,:,:]

  dz_r = main.LSQfactors[0,2,0,:,:,:]
  dz_l = main.LSQfactors[1,2,0,:,:,:]
  dz_u = main.LSQfactors[2,2,0,:,:,:]
  dz_d = main.LSQfactors[3,2,0,:,:,:]
  dz_f = main.LSQfactors[4,2,0,:,:,:]
  dz_b = main.LSQfactors[5,2,0,:,:,:]

  du_r[:,0,0,0,0,0:-1,:,:,:] = main.a.u[:,0,0,0,0,1:: ,:,:,:]-main.a.u[:,0,0,0,0,0:-1,:,:,:]
  du_l[:,0,0,0,0,1:: ,:,:,:] = main.a.u[:,0,0,0,0,0:-1,:,:,:]-main.a.u[:,0,0,0,0,1:: ,:,:,:]
  du_u[:,0,0,0,0,:,0:-1,:,:] = main.a.u[:,0,0,0,0,:,1:: ,:,:]-main.a.u[:,0,0,0,0,:,0:-1,:,:]
  du_d[:,0,0,0,0,:,1:: ,:,:] = main.a.u[:,0,0,0,0,:,0:-1,:,:]-main.a.u[:,0,0,0,0,:,1:: ,:,:]
  du_f[:,0,0,0,0,:,:,0:-1,:] = main.a.u[:,0,0,0,0,:,:,1:: ,:]-main.a.u[:,0,0,0,0,:,:,0:-1,:]
  du_b[:,0,0,0,0,:,:,1:: ,:] = main.a.u[:,0,0,0,0,:,:,0:-1,:]-main.a.u[:,0,0,0,0,:,:,1:: ,:]
  
  if (main.BC_rank[1]==True):
    du_r[:,0,0,0,0,-1,:,:,:] = main.a.uR_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,-1,:,:,:]
  else:
    du_r[:,0,0,0,0,-1,:,:,:] = main.a.uR_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,-1,:,:,:]
    dx_r[-1,:,:]   = main.LSQfactors[0,0,1,-1,:,:]
    dy_r[-1,:,:]   = main.LSQfactors[0,1,1,-1,:,:]
    dz_r[-1,:,:]   = main.LSQfactors[0,2,1,-1,:,:]
  
  if (main.BC_rank[0]==True):
    du_l[:,0,0,0,0,0 ,:,:,:] = main.a.uL_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,0,:,:,:]
  else:
    du_l[:,0,0,0,0,0 ,:,:,:] = main.a.uL_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,0,:,:,:]
    dx_l[0,:,:]   = main.LSQfactors[1,0,1,0,:,:]
    dy_l[0,:,:]   = main.LSQfactors[1,1,1,0,:,:]
    dz_l[0,:,:]   = main.LSQfactors[1,2,1,0,:,:]
  
  if (main.BC_rank[3]==True):
    du_u[:,0,0,0,0,:,-1,:,:] = main.a.uU_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,:,-1,:,:]
  else:
    du_u[:,0,0,0,0,:,-1,:,:] = main.a.uU_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,:,-1,:,:]
    dx_u[:,-1,:]   = main.LSQfactors[2,0,1,:,-1,:]
    dy_u[:,-1,:]   = main.LSQfactors[2,1,1,:,-1,:]
    dz_u[:,-1,:]   = main.LSQfactors[2,2,1,:,-1,:]
  
  if (main.BC_rank[2]==True):
    du_d[:,0,0,0,0,:,0,:,:] = main.a.uD_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,:,0,:,:]
  else:
    du_d[:,0,0,0,0,:,0,:,:] = main.a.uD_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,:,0,:,:]
    dx_d[:,0,:]   = main.LSQfactors[3,0,1,:,0,:]
    dy_d[:,0,:]   = main.LSQfactors[3,1,1,:,0,:]
    dz_d[:,0,:]   = main.LSQfactors[3,2,1,:,0,:]
  
  if (main.BC_rank[5]==True):
    du_f[:,0,0,0,0,:,:,-1,:] = main.a.uF_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,:,:,-1,:]
  else:
    du_f[:,0,0,0,0,:,:,-1,:] = main.a.uF_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,:,:,-1,:]
    dx_f[:,:,-1]   = main.LSQfactors[4,0,1,:,:,-1]
    dy_f[:,:,-1]   = main.LSQfactors[4,1,1,:,:,-1]
    dz_f[:,:,-1]   = main.LSQfactors[4,2,1,:,:,-1]
  
  if (main.BC_rank[4]==True):
    du_b[:,0,0,0,0,:,:,0,:]  = main.a.uB_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,:,:,0,:]
  else:
    du_b[:,0,0,0,0,:,:,0,:]  = main.a.uB_edge[:,0,0,0,:,:,:]-main.a.u[:,0,0,0,0,:,:,0,:]
    dx_b[:,:,0]   = main.LSQfactors[5,0,1,:,:,0]
    dy_b[:,:,0]   = main.LSQfactors[5,1,1,:,:,0]
    dz_b[:,:,0]   = main.LSQfactors[5,2,1,:,:,0]
  
  dx_r = dx_r[None,None,None,None,None,:,:,:,None]
  dx_l = dx_l[None,None,None,None,None,:,:,:,None]
  dx_u = dx_u[None,None,None,None,None,:,:,:,None]
  dx_d = dx_d[None,None,None,None,None,:,:,:,None]
  dx_f = dx_f[None,None,None,None,None,:,:,:,None]
  dx_b = dx_b[None,None,None,None,None,:,:,:,None]
  dy_r = dy_r[None,None,None,None,None,:,:,:,None]
  dy_l = dy_l[None,None,None,None,None,:,:,:,None]
  dy_u = dy_u[None,None,None,None,None,:,:,:,None]
  dy_d = dy_d[None,None,None,None,None,:,:,:,None]
  dy_f = dy_f[None,None,None,None,None,:,:,:,None]
  dy_b = dy_b[None,None,None,None,None,:,:,:,None]
  dz_r = dz_r[None,None,None,None,None,:,:,:,None]
  dz_l = dz_l[None,None,None,None,None,:,:,:,None]
  dz_u = dz_u[None,None,None,None,None,:,:,:,None]
  dz_d = dz_d[None,None,None,None,None,:,:,:,None]
  dz_f = dz_f[None,None,None,None,None,:,:,:,None]
  dz_b = dz_b[None,None,None,None,None,:,:,:,None]

  dx_m = (dx_r**2 + dx_l**2 + dx_u**2 + dx_d**2 + dx_f**2 + dx_b**2)
  dy_m = (dy_r**2 + dy_l**2 + dy_u**2 + dy_d**2 + dy_f**2 + dy_b**2)
  dz_m = (dz_r**2 + dz_l**2 + dz_u**2 + dz_d**2 + dz_f**2 + dz_b**2)

  dudx = (du_r*dx_r + du_l*dx_l + du_u*dx_u + du_d*dx_d + du_f*dx_f + du_b*dx_b)/dx_m
  dudy = (du_r*dy_r + du_l*dy_l + du_u*dy_u + du_d*dy_d + du_f*dy_f + du_b*dy_b)/dy_m
  dudz = (du_r*dz_r + du_l*dz_l + du_u*dz_u + du_d*dz_d + du_f*dz_f + du_b*dz_b)/dz_m

  # Reconstruct
  
  xR = main.LSQfactors[0,0,1,:,:,:]
  yR = main.LSQfactors[0,1,1,:,:,:]
  zR = main.LSQfactors[0,2,1,:,:,:]
  
  main.a.uR[:,0,0,0,:,:,:,0] = main.a.u[:,0,0,0,0,:,:,:,0]
  main.a.uR[:,0,0,0,:,:,:,0] += dudx[:,0,0,0,0,:,:,:,0]*xR[None,:,:,:]
  main.a.uR[:,0,0,0,:,:,:,0] += dudy[:,0,0,0,0,:,:,:,0]*yR[None,:,:,:]
  main.a.uR[:,0,0,0,:,:,:,0] += dudz[:,0,0,0,0,:,:,:,0]*zR[None,:,:,:]

  xL = main.LSQfactors[1,0,1,:,:,:]
  yL = main.LSQfactors[1,1,1,:,:,:]
  zL = main.LSQfactors[1,2,1,:,:,:]
  
  main.a.uL[:,0,0,0,:,:,:,0] = main.a.u[:,0,0,0,0,:,:,:,0]
  main.a.uL[:,0,0,0,:,:,:,0] += dudx[:,0,0,0,0,:,:,:,0]*xL[None,:,:,:]
  main.a.uL[:,0,0,0,:,:,:,0] += dudy[:,0,0,0,0,:,:,:,0]*yL[None,:,:,:]
  main.a.uL[:,0,0,0,:,:,:,0] += dudz[:,0,0,0,0,:,:,:,0]*zL[None,:,:,:]
  
  xU = main.LSQfactors[2,0,1,:,:,:]
  yU = main.LSQfactors[2,1,1,:,:,:]
  zU = main.LSQfactors[2,2,1,:,:,:]
  
  main.a.uU[:,0,0,0,:,:,:,0] = main.a.u[:,0,0,0,0,:,:,:,0]
  main.a.uU[:,0,0,0,:,:,:,0] += dudx[:,0,0,0,0,:,:,:,0]*xU[None,:,:,:]
  main.a.uU[:,0,0,0,:,:,:,0] += dudy[:,0,0,0,0,:,:,:,0]*yU[None,:,:,:]
  main.a.uU[:,0,0,0,:,:,:,0] += dudz[:,0,0,0,0,:,:,:,0]*zU[None,:,:,:]
  
  xD = main.LSQfactors[3,0,1,:,:,:]
  yD = main.LSQfactors[3,1,1,:,:,:]
  zD = main.LSQfactors[3,2,1,:,:,:]
  
  main.a.uD[:,0,0,0,:,:,:,0] = main.a.u[:,0,0,0,0,:,:,:,0]
  main.a.uD[:,0,0,0,:,:,:,0] += dudx[:,0,0,0,0,:,:,:,0]*xD[None,:,:,:]
  main.a.uD[:,0,0,0,:,:,:,0] += dudy[:,0,0,0,0,:,:,:,0]*yD[None,:,:,:]
  main.a.uD[:,0,0,0,:,:,:,0] += dudz[:,0,0,0,0,:,:,:,0]*zD[None,:,:,:]
  
  xF = main.LSQfactors[4,0,1,:,:,:]
  yF = main.LSQfactors[4,1,1,:,:,:]
  zF = main.LSQfactors[4,2,1,:,:,:]
  
  main.a.uF[:,0,0,0,:,:,:,0] = main.a.u[:,0,0,0,0,:,:,:,0]
  main.a.uF[:,0,0,0,:,:,:,0] += dudx[:,0,0,0,0,:,:,:,0]*xF[None,:,:,:]
  main.a.uF[:,0,0,0,:,:,:,0] += dudy[:,0,0,0,0,:,:,:,0]*yF[None,:,:,:]
  main.a.uF[:,0,0,0,:,:,:,0] += dudz[:,0,0,0,0,:,:,:,0]*zF[None,:,:,:]
  
  xB = main.LSQfactors[5,0,1,:,:,:]
  yB = main.LSQfactors[5,1,1,:,:,:]
  zB = main.LSQfactors[5,2,1,:,:,:]
  
  main.a.uB[:,0,0,0,:,:,:,0] = main.a.u[:,0,0,0,0,:,:,:,0]
  main.a.uB[:,0,0,0,:,:,:,0] += dudx[:,0,0,0,0,:,:,:,0]*xB[None,:,:,:]
  main.a.uB[:,0,0,0,:,:,:,0] += dudy[:,0,0,0,0,:,:,:,0]*yB[None,:,:,:]
  main.a.uB[:,0,0,0,:,:,:,0] += dudz[:,0,0,0,0,:,:,:,0]*zB[None,:,:,:]
  
  #return uR,uL,uU,uD,uF,uB