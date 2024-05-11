from numpy import linspace, meshgrid, zeros, sqrt, empty



__all__=['L2_norm_2D',
         'plot_field_2D'
]
################################### L2_norm ##############################

def L2_norm_2D(nelements, degree, spans, basis, weights, points, Uh, Uex):

    # ... sizes
    ne1,ne2              = nelements
    p1,p2                = degree
    spans1, spans2       = spans
    basis1, basis2       = basis
    weights1, weights2   = weights
    points1, points2     = points
    
    k1    = weights1.shape[1]
    k2    = weights2.shape[1]
    v     = 0.0
    for i_e1 in range(ne1):
        i_span_1 = spans1[i_e1] 
        for i_e2 in range(ne2):
            i_span_2 = spans2[i_e2]
   
            for i_k1 in range(k1):
                x1 = points1[i_e1, i_k1]
                for i_k2 in range(k2):
                    x2 =  points2[i_e2, i_k2]
                    w = 0.0
                    for i_1 in range(p1+1):
                        i = i_span_1 - p1 + i_1
                        for j_1 in range(p2+1):
                            j = i_span_2 - p2 + j_1
                            bxi = basis1[i_e1, i_1, 0, i_k1]* basis2[i_e2, j_1, 0, i_k2]
                            w += bxi*Uh[i,j]
                    wvol = weights1[i_e1, i_k1]*weights2[i_e2, i_k2]
                    v += wvol*(Uex(x1, x2)-w)**2
    return sqrt(v)
######################################### H1_Norm#############################################################
def H1_norm_2D(nelements, degree, spans, basis, weights, points, Uh, dUx, dUy, Uex):

    # ... sizes
    ne1,ne2              = nelements
    p1,p2                = degree
    spans1, spans2       = spans
    basis1, basis2       = basis
    weights1, weights2   = weights
    points1, points2     = points
    
    k1    = weights1.shape[1]
    k2    = weights2.shape[1]
    v     = 0.0
    for i_e1 in range(ne1):
        i_span_1 = spans1[i_e1] 
        for i_e2 in range(ne2):
            i_span_2 = spans2[i_e2]
   
            for i_k1 in range(k1):
                x1 = points1[i_e1, i_k1]
                for i_k2 in range(k2):
                    x2 =  points2[i_e2, i_k2]
                    w1 = 0. ; w2 = 0.; w3 = 0.
                    for i_1 in range(p1+1):
                        i = i_span_1 - p1 + i_1
                        for j_1 in range(p2+1):
                            j = i_span_2 - p2 + j_1
                            bxi_i  = basis1[i_e1, i_1, 0, i_k1]* basis2[i_e2, j_1, 0, i_k2]
                            bdx_i  = basis1[i_e1, i_1, 1, i_k1]* basis2[i_e2, j_1, 0, i_k2]
                            bdy_i  = basis1[i_e1, i_1, 0, i_k1]* basis2[i_e2, j_1, 1, i_k2]
                            w1    += bxi_i*Uh[i,j]
                            w2    += bdx_i*Uh[i,j]       
                            w3    += bdy_i*Uh[i,j]                            
                    wvol = weights1[i_e1, i_k1] * weights2[i_e2, i_k2]
                    v += wvol * (Uex(x1, x2)-w1)**2 +  wvol * (dUx(x1, x2)-w2)**2 + wvol * (dUy(x1, x2)-w3)**2
    return sqrt(v)
########################################## Plotting in 2D ###################################################
def find_span( knots, degree, x ):
    knots = knots
    p     = degree
    
    low   = p
    high = len(knots)-1-p
    if   x <= knots[low ]: mid = low
    elif x >= knots[high]: mid =  high-1
    else:
        mid = (low+high)//2
        while x < knots[mid] or x >= knots[mid+1]:
            if x < knots[mid]:
               high = mid
            else:
               low  = mid
            mid = (low+high)//2

    return mid
    
def basis_funs( knots, degree, x, span ):

    left   = empty( degree  , dtype=float )
    right  = empty( degree  , dtype=float )
    values = empty( degree+1, dtype=float )

    values[0] = 1.0
    for j in range(0,degree):
        left [j] = x - knots[span-j]
        right[j] = knots[span+1+j] - x
        saved    = 0.0
        for r in range(0,j+1):
            temp      = values[r] / (right[r] + left[j-r])
            values[r] = saved + right[r] * temp
            saved     = left[j-r] * temp
        values[j+1] = saved

    return values


def plot_field_2D(knots, degree, u, nx = 101, ny=101):

    knots1, knots2 = knots
    degree1, degree2 = degree
    xmin, xmax, ymin ,ymax = knots1[degree1], knots1[-degree1], knots2[degree2], knots2[-degree2]
    xs = linspace(xmin, xmax, nx)
    ys = linspace(ymin, ymax, ny)
    P = zeros((nx,ny,1))
    nu1,nu2 = u.shape
    Q = zeros((nu1,nu2))
    
    Q[:,:]= u[:,:]
    for i ,xi in enumerate(xs):
        for j,yj in enumerate(xs):
        
            i_span = find_span(knots1, degree1, xi)
            j_span = find_span(knots2, degree2, yj)
            valuesx = basis_funs(knots1, degree1, xi, i_span)
            valuesy = basis_funs(knots2, degree2, yj, j_span)
            
            C = zeros(P.shape[-1])
            for i_1 in range(degree1+1):
                for j_1 in range(degree2+1):
                
                    C[:]+=valuesx[i_1]*valuesy[j_1]*Q[i_span-degree1+i_1, j_span-degree2+j_1]
            P[i,j,:] = C[:]
            
    X,Y = meshgrid(xs,ys)
    ax = plt.axes(projection = '3d')
    
    ax.plot_surface(X,Y, P[:,:,0].T, cmap ='viridis', edgecolor ='orange')
    # plt.contourf(X,Y, P[:,:,0].T, cmap ='viridis', edgecolor ='orange')
    plt.show()
     
