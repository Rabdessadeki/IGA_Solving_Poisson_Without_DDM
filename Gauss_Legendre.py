
from numpy import zeros
from math import cos, pi

__all__ = ['Gauss_Legendre',
           'quadrature_grid'
           ]

def legendre_pol(x,m):
    p0=1
    p1 = x
    for i in range(1,m):
        p = ((2*i+1)*x*p1-i*p0)/(i+1)
        p0 = p1
        p1 = p
    dp = m*(x*p1-p0)/(x**2-1)
    return p1,dp
def Gauss_Legendre(orderlp,tol=1e-14):
    m=orderlp+1
    A = zeros(m)
    xx = zeros(m)
    mroots = (m+1)//2 # Number of roots non negative
    for i in range(mroots):
        x  = cos(pi*(i+0.75)/(m+0.5))
        for j in range(50):
            
            p,dp = legendre_pol(x,m)
            dx = -p/dp
            x = x+dx
            if(abs(dx)<tol):
                xx[i] = x
                xx[m-i-1]=-x
                A[i]= 2.0/((1.0-x**2)*dp**2)
                A[m-i-1]=A[i]
                break
    return xx,A
def quadrature_grid(breakp,qx,qw):
    ne   = len(breakp)-1
    nq   = len(qx)
    ru_x = zeros((ne,nq))
    ru_w = zeros((ne,nq))
    for ir , (ai,bi) in enumerate(zip(breakp[:-1],breakp[1:])):
        c = (bi-ai)/2
        b = (bi+ai)/2 
        ru_x[ir,]=c*qx+b
        ru_w[ir,] = c*qw
    return ru_x,ru_w
