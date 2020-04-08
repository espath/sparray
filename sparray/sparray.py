from __future__ import print_function, division
import sys, os, time
import numpy as np
import sympy as sp
import scipy as sc
import sparray as sr
# import numba
# from numba import jit
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches
from matplotlib.path import Path

#------------------------------------------------------------------------------#

def fourier_basis(m,x,T):
    return np.exp(2.0j*np.pi*np.sum(m*x*T))

#------------------------------------------------------------------------------#

def fourier_tensor_basis(m,x,T,multi_index):
    A = sp.sparray(tuple(m),dtype=complex)
    for index in multi_index:
        A[index] = fourier_basis(np.asarray(index),x,T)
    return A

#------------------------------------------------------------------------------#

def fourier_residual(d,A,u,U):
    residual = np.zeros(d,dtype=complex)
    for i in range(d):
        residual[i] = np.sum(A*U[i]) - u[i]
    return np.real(residual)

#------------------------------------------------------------------------------#

def fourier_eval(d,m,x,T,U):
    result = np.zeros(d,dtype=complex)
    A = fourier_tensor_basis(m,x,T)
    for i in range(d):
        result[i] = np.sum(A*U[i])
    return np.real(result)

#------------------------------------------------------------------------------#

def fourier_stabilization(d,m,eps,p,T,U,multi_index):
    U2 = np.power(np.abs(U),2.0)
    stabilization = 0.0
    for index in multi_index:
        factor = np.asarray(index)
        factor_sq = np.sum(np.power(2.0*np.pi*np.abs(factor)*T,2.0*p))
        for i in range(d):
            stabilization += eps*U2[(i,)+index]*factor_sq
    return np.real(stabilization)

#------------------------------------------------------------------------------#

def immersed_boundaries(d,m,B,xB,nB,T,U):
    residual = 0.0
    for iB in range(B):
        v = fourier_eval(d,m,xB[iB],T,U)
        residual += np.dot(np.real(v),-nB[iB])**2.0
    return np.real(residual)

#------------------------------------------------------------------------------#

def wall_boundaries(d,m,W,xW,nW,T,U):
    residual = 0.0
    for iW in range(W):
        v = fourier_eval(d,m,xW[iW],T,U)
        residual += np.dot(np.real(v),-nW[iW])**2.0
    return np.real(residual)

#------------------------------------------------------------------------------#

def fourier_all_residuals(d,m,P,xP,uP,T,U):
    residual = np.zeros((P,d),dtype=complex)
    for iP in range(P):
        A = fourier_tensor_basis(m,xP[iP],T)
        residual[iP,...] = fourier_residual(d,A,uP[iP],U)
    return np.real(residual)

#------------------------------------------------------------------------------#

def grad_fourier_all_residuals(d,m,P,xP,uP,T,U):
    GradR = np.zeros((d,)+tuple(m),dtype=complex)
    for iP in range(P):
        A = fourier_tensor_basis(m,xP[iP],T)
        residual = fourier_residual(d,A,uP[iP],U)
        for i in range(d): GradR[i] += 2.0*A.conj()*residual[i]
    return GradR/P

#------------------------------------------------------------------------------#

def grad_stabilization(d,m,eps,p,T,U):
    GradS = np.zeros((d,)+tuple(m),dtype=complex)
    it = np.nditer(np.zeros(tuple(m)),flags=['multi_index'])
    while not it.finished:
        factor = np.asarray(it.multi_index)-(np.asarray(m)-np.ones(d))/2.0
        factor_sq = np.sum(np.power(2.0*np.pi*np.abs(factor)*T,2.0*p))
        for i in range(d):
            GradS[(i,)+it.multi_index] = eps*U[(i,)+it.multi_index]*factor_sq
        it.iternext()
    return GradS

#------------------------------------------------------------------------------#

def grad_immersed_boundaries(d,m,B,xB,nB,T,U):
    GradB = np.zeros((d,)+tuple(m),dtype=complex)
    for iB in range(B):
        v = fourier_eval(d,m,xB[iB],T,U)
        factor = np.dot(np.real(v),-nB[iB])
        A = fourier_tensor_basis(m,xB[iB],T)
        for i in range(d): GradB[i] += factor*(-nB[iB,i])*A
    return GradB/B

#------------------------------------------------------------------------------#

def grad_wall_boundaries(d,m,W,xW,nW,T,U):
    GradW = np.zeros((d,)+tuple(m),dtype=complex)
    for iW in range(W):
        v = fourier_eval(d,m,xW[iW],T,U)
        factor = np.dot(np.real(v),-nW[iW])
        A = fourier_tensor_basis(m,xW[iW],T)
        for i in range(d): GradW[i] += factor*(-nW[iW,i])*A
    return GradW/W

#------------------------------------------------------------------------------#

def comulative_stabilized_residual(d,m,eps,p,P,xP,uP,B,xB,nB,W,xW,nW,T,U):
    residualP = 0.0; residualB = 0.0; residualW = 0.0
    if (P):
        residualsP = fourier_all_residuals(d,m,P,xP,uP,T,U)
        for iP in range(P):
            residualP += np.dot(residualsP[iP],residualsP[iP])
        stabilization = fourier_stabilization(d,m,eps,p,T,U)
        residualP = residualP/P + stabilization
    if (B):
        residualB = immersed_boundaries(d,m,B,xB,nB,T,U)
        residualB = residualB/B
    if (W):
        residualW = wall_boundaries(d,m,W,xW,nW,T,U)
        residualW = residualW/W
    return residualP + residualB + residualW

#------------------------------------------------------------------------------#

def constraints(d,m,T):
    # building indices to name the variables
    indices = np.zeros((2,)+(d,)+tuple(m),dtype=int)
    indices_inv = [] # np.zeros(2*d*np.prod(m),dtype=int)
    it = np.nditer(indices,flags=['multi_index'])
    count = 0
    while not it.finished:
        indices[it.multi_index] = count
        indices_inv += [it.multi_index]
        count += 1
        it.iternext()
    # divergence constraint
    const = -1
    # counting the number of constraints, divergence free constraint
    c_d = 2*(np.prod(m) - 1)
    # counting the number of constraints, real function constraint
    c_r = 2*d*(np.prod(m) - 1)
    # constraint matrix
    C = np.zeros((c_d+c_r,2*d*np.prod(m)),dtype=float)
    for ri in range(2):
        it = np.nditer(np.zeros(tuple(m)),flags=['multi_index'])
        while not it.finished:
            const += 1
            m_shift = np.asarray(it.multi_index)-(np.asarray(m)-1)/2
            for i in range(d):
                C[const,indices[(ri,)+(i,)+it.multi_index]] = 2.0*np.pi*m_shift[i]*T[i]
            it.iternext()
    # real function constraint
    it = np.nditer(np.zeros(tuple(m)),flags=['multi_index'])
    while not it.finished:
        m_shift = np.asarray(it.multi_index)-(np.asarray(m)-1)/2
        m_conj = -m_shift+(np.asarray(m)-1)/2
        for i in range(d):
            for ri in range(2):
                const += 1
                if (ri==0): aux = -1.0
                if (ri>0): aux = 1.0
                C[const,indices[(ri,)+(i,)+it.multi_index]] = 1.0
                C[const,indices[(ri,)+(i,)+tuple(m_conj.astype(int))]] = aux
        if ((m_shift==0).all()): break
        it.iternext()
    Proj = sc.linalg.null_space(C)
    return Proj

#------------------------------------------------------------------------------#

def grad_inline(d,m,eps,p,P,xP,uP,B,xB,nB,W,xW,nW,T,U):
    gradR = grad_fourier_all_residuals(d,m,P,xP,uP,T,U)
    gradS = grad_stabilization(d,m,eps,p,T,U)
    if (B): gradB = grad_immersed_boundaries(d,m,B,xB,nB,T,U)
    if (W): gradW = grad_wall_boundaries(d,m,W,xW,nW,T,U)
    if (B and W): grad = gradR + gradS + gradB + gradW
    if (B==None and W==None): grad = gradR + gradS
    if (B and W==None): grad = gradR + gradS + gradB
    if (B==None and W): grad = gradR + gradS + GradW
    grad_in = coefficients_to_array(d,m,grad)
    return grad_in

#------------------------------------------------------------------------------#

def coefficients_to_array(d,m,X_in):
    X_out = np.zeros(2*d*np.prod(m),dtype=float)
    toit = np.zeros((d,)+tuple(m),dtype=int)
    it = np.nditer(toit,flags=['multi_index'])
    i = 0
    dist = d*np.prod(m)
    while not it.finished:
        X_out[i] = np.real(X_in[it.multi_index])
        X_out[i+dist] = np.imag(X_in[it.multi_index])
        i += 1
        it.iternext()
    return X_out

#------------------------------------------------------------------------------#

def array_to_coefficients(d,m,X_in):
    X_out = np.zeros((d,)+tuple(m),dtype=complex)
    toit = np.zeros((d,)+tuple(m),dtype=int)
    it = np.nditer(toit,flags=['multi_index'])
    i = 0
    dist = d*np.prod(m)
    while not it.finished:
        X_out[it.multi_index] = X_in[i] + X_in[i+dist]*1.0j
        i += 1
        it.iternext()
    return X_out

#------------------------------------------------------------------------------#

def foo(d,m,eps,p,P,xP,uP,B,xB,nB,W,xW,nW,T,Proj,U):
    U = np.dot(Proj,U)
    U = array_to_coefficients(d,m,U)
    residual = comulative_stabilized_residual(d,m,eps,p,P,xP,uP,B,xB,nB,W,\
        xW,nW,T,U)
    return np.real(residual)

#------------------------------------------------------------------------------#

def grad_foo(d,m,eps,p,P,xP,uP,B,xB,nB,W,xW,nW,T,Proj,U):
    U = np.dot(Proj,U)
    U = array_to_coefficients(d,m,U)
    grad_in = grad_inline(d,m,eps,p,P,xP,uP,B,xB,nB,W,xW,nW,T,U)
    grad_in_proj = np.dot(Proj.T,grad_in)
    return grad_in_proj

#------------------------------------------------------------------------------#

def reconstruct_2D(d,m,T,X):
    U = np.zeros((d,)+tuple(m),dtype=complex)
    i = 0
    it = np.nditer(np.zeros(tuple(m)),flags=['multi_index'])
    # Real part divergence conforming
    while not it.finished:
        m_shift = np.asarray(it.multi_index)-(np.asarray(m)-1)/2
        if ((m_shift==0).all()):
            U[(0,)+it.multi_index] = X[i]
            break
        if (np.int(m_shift[1])!=0):
            U[(0,)+it.multi_index] = X[i]
            U[(1,)+it.multi_index] = - m_shift[0]*T[0]*X[i]/(T[1]*m_shift[1])
        if (np.int(m_shift[1])==0):
            U[(0,)+it.multi_index] = 0.0 + 0.0j
            U[(1,)+it.multi_index] = X[i]
        i += 1
        it.iternext()
    i += 1
    it = np.nditer(np.zeros(tuple(m)),flags=['multi_index'])
    # Imaginary part divergence conforming
    while not it.finished:
        m_shift = np.asarray(it.multi_index)-(np.asarray(m)-1)/2
        if ((m_shift==0).all()):
            break
        if (np.int(m_shift[1])!=0):
            U[(0,)+it.multi_index] = np.real(U[(0,)+it.multi_index]) + X[i]*1j
            U[(1,)+it.multi_index] = np.real(U[(1,)+it.multi_index]) \
            - (m_shift[0]*T[0]*X[i]/(T[1]*m_shift[1]))*1j
        if (np.int(m_shift[1])==0):
            U[(1,)+it.multi_index] = np.real(U[(1,)+it.multi_index]) + X[i]*1j
        i += 1
        it.iternext()
    # Conjugate
    it = np.nditer(np.zeros(tuple(m)),flags=['multi_index'])
    while not it.finished:
        m_shift = np.asarray(it.multi_index)-(np.asarray(m)-1)/2
        m_conj = m - np.asarray(it.multi_index) - 1
        if ((m_shift==0).all()):
            U[(1,)+it.multi_index] = U[(0,)+it.multi_index]
            break
        for i in range(d):
            U[(i,)+tuple(m_conj)] = np.conjugate(U[(i,)+it.multi_index])
        it.iternext()
    return U

#------------------------------------------------------------------------------#

def reconstruct_grad_2D(d,m,T,U):
    X = np.zeros(np.prod(m))
    i = 0
    it = np.nditer(np.zeros(tuple(m)),flags=['multi_index'])
    # Real part divergence conforming
    while not it.finished:
        m_shift = np.asarray(it.multi_index)-(np.asarray(m)-1)/2
        if ((m_shift==0).all()):
            X[i] = np.real(U[(0,)+it.multi_index])
            break
        if (np.int(m_shift[1])!=0):
            X[i] = (1.0-m_shift[0]/m_shift[1])*np.real(U[(0,)+it.multi_index])
        if (np.int(m_shift[1])==0):
            X[i] = np.real(U[(1,)+it.multi_index])
        i += 1
        it.iternext()
    i += 1
    it = np.nditer(np.zeros(tuple(m)),flags=['multi_index'])
    # Imaginary part divergence conforming
    while not it.finished:
        m_shift = np.asarray(it.multi_index)-(np.asarray(m)-1)/2
        if ((m_shift==0).all()):
            break
        if (np.int(m_shift[1])!=0):
            X[i] = (1.0-m_shift[0]/m_shift[1])*np.imag(U[(0,)+it.multi_index])
        if (np.int(m_shift[1])==0):
            X[i] = np.imag(U[(1,)+it.multi_index])
        i += 1
        it.iternext()
    return X

#------------------------------------------------------------------------------#

def Adam(k,lrnRt,m_k,v_k,Grad):
    Adam_eps = 1e-15; beta1 = 0.9; beta2 = 0.999
    g_k = np.mean(Grad,axis=0)
    g_k2 = np.mean(np.square(Grad),axis=0)
    if k==0:
        m_k = np.zeros(g_k.size)
        v_k = np.zeros(g_k.size)
    m_k = beta1*m_k + (1.0-beta1)*g_k
    v_k = beta2*v_k + (1.0-beta2)*g_k2
    mHat = m_k/(1.0-beta1**(k+1))
    vHat = v_k/(1.0-beta2**(k+1))
    # dx = - lrnRt/np.sqrt(k+1)*mHat/(np.sqrt(vHat) + Adam_eps)
    dx = - lrnRt*mHat/(np.sqrt(vHat) + Adam_eps)
    return dx, m_k, v_k

#------------------------------------------------------------------------------#

# sanity check
def sanity_divFree(d,m,U):
    # U = U[:np.int(d*np.prod(m))] + U[np.int(d*np.prod(m)):]*1j
    # U = np.reshape(U,(d,)+tuple(m))
    constraint = np.zeros(tuple(m),dtype=complex)
    it = np.nditer(np.zeros(tuple(m)),flags=['multi_index'])
    while not it.finished:
        m_shift = np.asarray(it.multi_index)-(np.asarray(m)-1)/2
        for i in range(d):
            constraint[it.multi_index] += m_shift[i]*U[(i,)+it.multi_index]
        it.iternext()
    return np.sum(constraint)

#------------------------------------------------------------------------------#

# sanity check
def sanity_realValue(d,m,U):
    # U = U[:np.int(d*np.prod(m))] + U[np.int(d*np.prod(m)):]*1j
    # U = np.reshape(U,(d,)+tuple(m))
    constraint = np.zeros((d,)+tuple(m),dtype=complex)
    i_cons = 0
    it = np.nditer(np.zeros((d,)+tuple(m)),flags=['multi_index'])
    flip_conj = np.zeros((d,)+tuple(m),dtype=complex)
    flip_conj[0] = np.flip(np.conjugate(U[0]))
    flip_conj[1] = np.flip(np.conjugate(U[1]))
    while not it.finished:
        constraint[it.multi_index] = U[it.multi_index] - flip_conj[it.multi_index]
        it.iternext()
    return np.sum(constraint)

#------------------------------------------------------------------------------#

def reshaped_CSR(d,m,P,xP,uP,B,xB,nB,W,xW,nW,eps,p,T,U):
    U = reconstruct_2D(d,m,T,U)
    residual = comulative_stabilized_residual(d,m,eps,p,P,xP,uP,B,xB,nB,W,\
        xW,nW,T,U)
    return np.real(residual)

#------------------------------------------------------------------------------#

def plotDivField_2D(fig,ax,nx,d,m,T,U):
    D = 1.0/T
    Y, X = np.mgrid[0.0:D[1]:nx[1]*1j, 0.0:D[0]:nx[0]*1j]
    divField = np.zeros((d,)+X.shape)
    for i in range(nx[0]):
        for j in range(nx[1]):
            pos = np.array([X[j,i],Y[j,i]])
            divField[:,j,i] = np.real(fourier_eval(d,m,pos,T,U))
    speed = np.sqrt(divField[0]*divField[0] + divField[1]*divField[1])
    # fig = plt.figure()
    ax = fig.add_subplot(111)
    color = 2.0*np.log(np.hypot(divField[0],divField[1]))
    lw = 5.0*speed/speed.max()
    ax.streamplot(X, Y, divField[0], divField[1], \
        linewidth=lw, color='k', density=1.0, arrowstyle='->', \
        arrowsize=1.0)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(0.0,D[0])
    ax.set_ylim(0.0,D[1])
    ax.set_aspect('equal')
    plt.show()

#------------------------------------------------------------------------------#

def plotDivField_P_2D(nx,d,m,P,xP,uP,T,U):
    D = 1.0/T
    Y, X = np.mgrid[0.0:D[1]:nx[1]*1j, 0.0:D[0]:nx[0]*1j]
    divField = np.zeros((d,)+X.shape)
    for i in range(nx[0]):
        for j in range(nx[1]):
            pos = np.array([X[j,i],Y[j,i]])
            divField[:,j,i] = np.real(fourier_eval(d,m,pos,T,U))
    speed = np.sqrt(divField[0]*divField[0] + divField[1]*divField[1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = 2.0*np.log(np.hypot(divField[0],divField[1]))
    lw = 5.0*speed/speed.max()
    for i in range(P):
        ax.add_artist(Circle(np.real(xP[i]), 0.1, color='red'))
        ax.arrow(np.real(xP[i,0]),np.real(xP[i,1]), \
            np.real(uP[i,0]),np.real(uP[i,1]), \
            head_width=0.15,head_length=0.15,width=0.05, \
            fc='b',ec='b',clip_on=False)
    ax.streamplot(X, Y, divField[0], divField[1], \
        linewidth=lw, color='k', density=1.0, arrowstyle='->', \
        arrowsize=1.0)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(0.0,D[0])
    ax.set_ylim(0.0,D[1])
    ax.set_aspect('equal')
    plt.show()

#------------------------------------------------------------------------------#

def plotDivField_P_IB_2D(nx,d,m,P,xP,uP,B,xB,nB,T,U):
    D = 1.0/T
    Y, X = np.mgrid[0.0:D[1]:nx[1]*1j, 0.0:D[0]:nx[0]*1j]
    divField = np.zeros((d,)+X.shape)
    for i in range(nx[0]):
        for j in range(nx[1]):
            pos = np.array([X[j,i],Y[j,i]])
            divField[:,j,i] = np.real(fourier_eval(d,m,pos,T,U))
    speed = np.sqrt(divField[0]*divField[0] + divField[1]*divField[1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color = 2.0*np.log(np.hypot(divField[0],divField[1]))
    lw = 5.0*speed/speed.max()
    for i in range(P):
        ax.add_artist(Circle(np.real(xP[i]), 0.1, color='red',clip_on=False))
        ax.arrow(np.real(xP[i,0]),np.real(xP[i,1]), \
            np.real(uP[i,0]),np.real(uP[i,1]), \
            head_width=0.15,head_length=0.15,width=0.05, \
            fc='b',ec='b',clip_on=False)
    ax.streamplot(X, Y, divField[0], divField[1], \
        linewidth=lw, color='k', density=1.0, arrowstyle='->', \
        arrowsize=1.0)

    verts = []
    codes = [Path.MOVETO]
    for iB in range(B):
        verts += [tuple(xB[iB])]
        codes += [Path.LINETO]
    verts += [tuple(xB[0])]
    codes[-1] = Path.CLOSEPOLY
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='b', lw=2)
    ax.add_patch(patch)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(0.0,D[0])
    ax.set_ylim(0.0,D[1])
    ax.set_aspect('equal')
    plt.show()
