from bspline   import elements_spans  # computes the span for each element
from bspline   import make_knots,basis_funs,find_span     # create a knot sequence from a grid
from bspline   import quadrature_grid ,breakpoints# create a quadrature rule over the whole 1d grid
from bspline   import basis_ders_on_quad_grid,basis_funs_all_ders # evaluates all bsplines and their derivatives on the quad grid
from Gauss_Legendre import Gauss_Legendre, quadrature_grid
from stdio     import Mass_Matrix, Stiffness_Matrix, assemble_rhs_with_Non_homogenuous_DBC
from stdio     import B_Spline_Least_Square, assemble_stiffness_2D, L2_projection
from equipment import L2_norm_2D, H1_norm_2D, plot_field_2D
from scipy.sparse.linalg import cg
from scipy.linalg import norm, inv, solve, det,inv
from numpy import zeros, ones, linspace,double,float64, cos,array, dot, zeros_like, asarray,floor,arange,append,random,sqrt, int32, meshgrid,sin
import  matplotlib.pyplot as plt
from scipy.sparse        import csr_matrix
from scipy.sparse        import csc_matrix, linalg as sla
from scipy.sparse.linalg import gmres
from numpy               import zeros, linalg, asarray
from numpy               import cos, pi

#from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pyccel.decorators import types
                   
from scipy.sparse import kron as spkron





@types('double[:,:,:,:]', 'int', 'int')
def tensor_to_matrix(Mat, nh, nm) :
    sti = zeros((nh * nm, nh * nm), dtype=double)
    for i in range(nh):
        for j in range(nm):
            for k in range(nh):
                for l in range(nm):
                    i_1 = k + i * nm
                    i_2 = l + j * nh
                    sti[i_1, i_2] = Mat[i, j, k, l]                    
    return sti
def full_table_2d(a, b, c, d, my_elements, my_degree):

    '''test1
    gx0 = lambda x    :  0.
    gx1 = lambda x    :  2.* cos(pi*x)
    gy0 = lambda y    :  2.* y 
    gy1 = lambda y    : -2.* y
    Uex = lambda x, y : 2.*x*cos(pi*y) 
    dUx = lambda x, y : 2.*cos(pi*y)  
    dUy = lambda x, y : -2*pi*x*sin(pi*y)
    f   = lambda x, y : 2.*x*pi**2*cos(pi*y)
    
    '''
    kappa = 2
    gx0 = lambda x    :  0
    gx1 = lambda x    :  0
    gy0 = lambda y    : 0 
    gy1 = lambda y    : 0
   
    dUx = lambda x, y : kappa*pi*cos(kappa*pi*x)*sin(kappa*pi*y)  
    dUy = lambda  x, y : kappa*pi*sin(kappa*pi*x)*cos(kappa*pi*y)  
    Uex = lambda x,y: sin(kappa*pi*x)*sin(kappa*pi*y)  
    f   = lambda x,y: 2.*(pi*kappa)**2*sin(kappa*pi*x)*sin(kappa*pi*y)
    L2  = []
    H1  = []
    print('###########################################################################################')
    print('Please wait until the program finished it take a few minutes' )
    for ne1, ne2 in zip(my_elements,my_elements): 
        for p1, p2 in zip(my_degree, my_degree):
            grid1, grid2            = linspace(a,b,ne1+1), linspace(c,d, ne2+1)            

            knots1, knots2          = make_knots(grid1, p1, False), make_knots(grid2, p2, False)
            spans1, spans2          = elements_spans(knots1, p1), elements_spans(knots2, p2)
            nelements1, nelements2  = len(grid1)-1,len(grid2)-1
            nbasis1, nbasis2        = len(knots1)-p1-1, len(knots2)-p2-1
            nders                   = 1
            U1 , W1                 = Gauss_Legendre(p1)
            U2 , W2                 = Gauss_Legendre(p2)
            points1, weights1       = quadrature_grid(grid1,U1,W1)
            points2, weights2       = quadrature_grid(grid2,U2,W2)
            
            basis1, basis2          = basis_ders_on_quad_grid(knots1, p1, points1, nders, normalize=False),basis_ders_on_quad_grid(knots2, p2, points2, nders, normalize=False)
            #stiffness     = zeros((nbasis1, nbasis2, nbasis1, nbasis2))
            #stiffness     = assemble_stiffness_2D(nelements1, nelements2, p1, p2, spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, stiffness)
            #stiffness     = tensor_to_matrix(stiffness[1:-1,1:-1,1:-1,1:-1], nbasis1-2, nbasis2-2)
            stiffness1      = zeros((nbasis1,nbasis2))
            stiffness2      = zeros((nbasis1,nbasis2))
            mass1           = zeros((nbasis1,nbasis2))
            mass2           = zeros((nbasis1,nbasis2))
            stiffness1      = Stiffness_Matrix(nelements1, p1, spans1, basis1, weights1, points1, stiffness1)
            stiffness2      = Stiffness_Matrix(nelements2, p2, spans2, basis2, weights2, points2, stiffness2)
            mass1           = Mass_Matrix(nelements1, p1, spans1, basis1, weights1, points1, mass1)
            mass2           = Mass_Matrix(nelements2, p2, spans2, basis2, weights2, points2, mass2)
            C1              = spkron(csr_matrix(stiffness1[1:-1,1:-1]),csr_matrix(mass2[1:-1,1:-1]))

            C2              = spkron(csr_matrix(mass1[1:-1,1:-1]), csr_matrix(stiffness2[1:-1,1:-1]))
# C1              = kron(stiffness1,mass2)
# C2              = kron(mass1,stiffness2)

            
            stiffness = C1+C2
            
            rhs1          = zeros((nbasis1,nbasis2))
            gx0_h         = L2_projection(knots1, p1, gx0)
            gx1_h         = L2_projection(knots1, p1, gx1)
            gy0_h         = L2_projection(knots2, p2, gy0)
            gy1_h         = L2_projection(knots2, p2, gy1)

            g_bou         = zeros((nbasis1, nbasis2), dtype = double)
            g_bou[0,:]    = gx0_h
            g_bou[-1,:]   = gx1_h
            g_bou[:,0]    = gy0_h
            g_bou[:,-1]   = gy1_h
            rhs1          = assemble_rhs_with_Non_homogenuous_DBC(f, g_bou, nelements1, nelements2, p1, p2, spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, rhs1)
            rhs1          = rhs1[1:-1, 1:-1]
            rhs1          = rhs1.reshape((nbasis1-2)*(nbasis2-2))
            lu            = sla.splu(csc_matrix(stiffness))
            Uapp          = lu.solve(rhs1) 
            Uh            = zeros((nbasis1,nbasis2))
            Uh[1:-1,1:-1] = Uapp.reshape((nbasis1-2),(nbasis2-2))
            Uh            = Uh + g_bou
            l2 = L2_norm_2D( nelements1, nelements2, p1, p2, spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, Uh, Uex)
            L2.append(l2)
            print('#############################################',' p1=p1 =',p1,'and  ne1 = ne2 = ', ne1,'##############################################')
            print('\n')
            print('The L2 norm for p1=p1 =',p1,'and  ne1 = ne2 = ', ne1, ' is equal to ', l2)
            h1 = H1_norm_2D( nelements1, nelements2, p1, p2, spans1, spans2, basis1, basis2, weights1, weights2, points1, points2, Uh, dUx, dUy, Uex)
            H1.append(h1)
            print('\n')
          
            print('The H1 norm for p1=p1 =',p1,'and  ne1 = ne2 = ', ne1, ' is equal to ', h1)
            
    return L2, H1


def number_pow_mod_2(nmax):
    #here we start 16 as last first emelent so the the variable g start with the intiale value is 4
    #becuase in 16 =2^4
    g = 4
    g = 4
    q = nmax//2
    k=16
    while q>=16:
        g += 1
        q  = q//2
    return g
#  Please wait until the program finished it take a few minutes 
Ncells_max        = 256
N_2_in_Ncells_max = number_pow_mod_2(Ncells_max)
max_deg           = 5                 
my_elements       = [2**i for i in range(4, N_2_in_Ncells_max+1)]
my_degrees        = [i for i in range(2, max_deg+1)]
L2, H1            = full_table_2d(0., 1., 0., 1., my_elements, my_degrees)
ML2norm = array(L2)
ML2norm = ML2norm.reshape(len(my_elements), len(my_degrees))
plt.figure(figsize=(7,7))
for j in range(len(my_degrees)):
    # if j ==0:
    #     plt.plot([12, 32, 64, 128, 256] ,ML2norm[:,j] ,'o-')
    # elif j==1:
    #     plt.plot([12, 32, 64, 128, 256] ,ML2norm[:,j] ,'v-')
    # elif j==2:
    #     plt.plot([12, 32, 64, 128, 256] ,ML2norm[:,j] ,'s-')
    # elif j==3:
    #     plt.plot([12, 32, 64, 128, 256] ,ML2norm[:,j] ,'P-')
    # elif j==4:
    #     plt.plot([12, 32, 64, 128, 256] ,ML2norm[:,j] ,'*-')
    # else:
    #     plt.plot([12, 32, 64, 128, 256], ML2norm[:,j],'d-')
    plt.plot([2**(2*i) for i in range(4, N_2_in_Ncells_max+1)], ML2norm[:,j],'*-')        
plt.yscale('log')
plt.xscale('log')
plt.legend(['$p=2$', '$p=3$', '$p=4$','$p=5$'])
plt.xlabel('Ncells ')
plt.ylabel('L2-norm')
plt.grid()
plt.show()
plt.savefig('L2_Norm_dependsVsNcells.png')
