"""
Rosemary Teague
00828351

Assumes cost.f90 has been compiled with f2py to generate the module,
hw2mod.so (filename may also be of form hw2mod.xxx.so where xxx is system-dependent text) using
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
import scipy.optimize
from hw2mod import cost
from hw2mod import hw2
from time import time
from time import clock
import timeit

def visualize(Nx,Ny,xrange=[-10,10],yrange=[-10,10], Noise=1.0):
    """
    ===============================================================
    Visualization of a 2D cost function, j, of the form:

        j = j + (1 - x)^2 + 20(y - x^2)^2
    ===============================================================

    Parameters
    ------------
    Nx : Integer
        Number of points along the x-direction to be plotted
    Ny : Integer
        Number of points along the y-direction to be plotted
    xrange : list, optional
        Range of x-points to be considered. Default from -10<x<10
    yrange : list, optional
        Range of y-points to be considered. Default from -10<y<10
    Noise : float
        Amplitude of Noise to be considered in second plot.

    Returns
    ----------
    N/A
    Calling this function will save two figures to the users directory. A plot
    titled hw211.png will display a contour plot of the cost function, on a
    logarithmic scale in j, between the values specified in xrange and yrange.
    A second plot titled hw212.png will display the same function over the same
    range but with a random noise added, the amplitude of which can be set as
    a parameter.
    """
    #Create 2D array of points
    [X,Y]=np.linspace(xrange[0],xrange[1],Nx),np.linspace(yrange[0],yrange[1],Ny)

    #calculate noiseless cost function at each point on 2D grid
    cost.c_noise=False
    j=[[cost.costj([xi,yi]) for xi in X] for yi in Y]

    #calculate noisey cost function at each point in 2D grid.
    cost.c_noise = True
    cost.c_noise_amp = Noise
    jn=[[cost.costj([xi,yi]) for xi in X] for yi in Y]

    #create contour plots of cost functions with and without noise
    plt.figure()
    fig, ax = plt.subplots()
    cp = ax.contourf(X, Y, j, locator=ticker.LogLocator(), cmap=cm.GnBu)
    cbar = fig.colorbar(cp)
    plt.title('Rosemary Teague, Visualize \n 2D cost function, no noise.')
    plt.savefig('hw211', dpi = 700)
    plt.figure()
    fig, ax = plt.subplots()
    cpn = ax.contourf(X, Y, jn, locator=ticker.LogLocator(), cmap=cm.GnBu)
    cbar = fig.colorbar(cpn)
    plt.title('Rosemary Teague, Visualize \n 2D cost function, Noise amplitude='+str(cost.c_noise_amp))
    plt.savefig('hw212', dpi = 700)



def newton_test(xg,display=False,i=1,timing=False):
    """
    ============================================================================
    Use Newton's method to minimize a cost function, j, defined in cost module.
    ============================================================================

    Parameters
    ----------
    xg : list
        Initial guess
    display : Boolean, Optional
        If set to True, figures will be created to illustrate the optimization
        path taken and the distance from convergence at each step.
    i=1 : Integer, Optional
        Sets the name of the figures as hw22i.png
    timing : Boolean, Optional
        If set to true, an average time will be calculated for the completion
        of finding a minimum and will be appended to the tuple output.

    Returns
    ---------
    xf : ndarray
        Computed location of minimum
    jf : float
        Computed minimum
    output : Tuple
        containing the time taken for the minimia to be found. An
        average over 10 tests, only set if timining parameter set to True, otherwise
        empty.

    Calling this function will produce a figure containing two subplots. The first
    will illustrate the location of each step in the minimization path, overlayed
    over the initial cost function. The second will illustrate the distance from
    the final, computed minimum at each iteration.

    """
    cost.c_noise=False
    hw2.tol=10**(-6)
    hw2.itermax=1000
    t21=0
    output=()

    if timing:
        N=10
    else:
        N=1

    for j in range(1,N):
        t1=time()
        hw2.newton(xg)
        t2=time()
        t21=t21+(t2-t1)

    X,Y=hw2.xpath
    xf=[X[-1],Y[-1]]
    jpathn=[j for j in hw2.jpath]
    jf=hw2.jpath[-1]

    output=(t21/N, X, Y, jpathn)

    if display:
        Minx=min(X)-1
        Maxx=max(X)+1
        Miny=min(Y)-1
        Maxy=max(Y)+1
        [Xj,Yj]=np.linspace(Minx,Maxx,200),np.linspace(Miny,Maxy,200)
        #calculate noiseless cost function at each point on 2D grid
        j=[[cost.costj([xi,yi]) for xi in Xj] for yi in Yj]

        f, (p1,p2) = plt.subplots(1,2)
        p1.contourf(Xj, Yj, j, locator=ticker.LogLocator(), cmap=cm.GnBu)
        p1.plot(X,Y,'g',marker='d')
        p1.set_xlim(min(X)-1,max(X)+1)
        p1.set_xlabel('X1-location')
        p1.set_ylabel('X2-location')
        p1.set_title('Convergence Path')
        p2.plot(np.linspace(0,len(X)-1,len(X)),hw2.jpath-jf)
        p2.set_xlabel('Iteration number')
        p2.set_ylabel('distance from converged minimum')
        p2.set_title('Rate')
        plt.suptitle('Rosemary Teague, Newton_test, initial guess ='+str(xg)+' \n Convergence of a cost function')
        plt.tight_layout(pad=4)
        plt.savefig('hw22'+str(i), dpi=700)

    return xf,jf,output


def bracket_descent_test(xg,display=False,compare=False,i=1):
    """
    ======================================================================================
    Use the Bracket Descent method to minimize a cost function, j, defined in cost module.
    ======================================================================================

    Parameters
    ----------
    xg : list
        Initial guess
    display : Boolean, Optional
        If set to True, figures will be created to illustrate the optimization
        path taken and the distance from convergence at each step.
    compare : Boolean, optional
        If set to True, a figure will be created to directly compare Newton and
        Bracket Descent methods.
    i=1 : Integer, Optional
        Sets the name of the figures as hw231(/2)_i.png.

    Returns
    ---------
    xf : ndarray
        Computed location of minimum
    jf : float
        Computed minimum
    output : Tuple
        containing the time taken for the minimia to be found for each of newton and
        bracket descent methods. An average over 10 tests is taken, only set if
        compare parameter set to True, otherwise empty.

    Calling this function will produce two figures. The first will containing two
    subplots illustrating the location of each step in the minimization path, overlayed
    over the initial cost function, and the distance of j from the final, computed
    minimum at each iteration.
    The second plot (which is only produced when 'compare' is set to True) demonstrates
    the distance of each step from the final, converged minimum at each iteration.
    This shows that the newton method requires significantly fewer steps and is hence
    faster.

    Trends Observed
    ----------------
    Figures hw321_i show the path taken during a bracket descent conversion is much
    longer than in a newton conversion (shown in figures hw22i). This is because
    the B-D method limits the size of a step to 2*L where L is definied by the size
    of an equilateral triangle whose centroid moved with each step. The method is
    furthermore designed such that this triangle will only decrease in size per
    iteration, and hence the maximum length a step can take can only be
    decreased (not increased) throughout the convergence. The figures further show
    that steps appear to be taken initially perpendicular to the curvature, finding
    the minimum along that strip, and then converging in down the parallel Path
    until they reach a level of tolerance.

    In contrast, the Newton approach is not limited in the size of the steps it is
    able to take and can hence converge in a much smaller number of iterations.
    This is a result of the use of gradients in this method. Figures hw22i illustrate
    how each step travels through many bands on the contour plot (representing
    differences of 1 order of magnitude each) as the method searches for the
    direction of minimisation.

    """
    cost.c_noise=False
    hw2.tol=10**(-6)
    hw2.itermax=1000
    t34=0
    output = ()

    if compare:
        N=10
    else:
        N=1

    for j in range(1,N):
        t3=time()
        hw2.bracket_descent(xg)
        t4=time()
        t34=t34+(t4-t3)

    X,Y=hw2.xpath
    xf=[X[-1],Y[-1]]
    jf=hw2.jpath[-1]
    d1=np.sqrt((X-xf[0])**2+(Y-xf[1])**2)


    if display:
        Minx=min(X)-1
        Maxx=max(X)+1
        Miny=min(Y)-1
        Maxy=max(Y)+1
        [Xj,Yj]=np.linspace(Minx,Maxx,200),np.linspace(Miny,Maxy,200)
        #calculate noiseless cost function at each point on 2D grid
        j=[[cost.costj([xi,yi]) for xi in Xj] for yi in Yj]
        f, (p1,p2) = plt.subplots(1,2)
        p1.contourf(Xj, Yj, j, locator=ticker.LogLocator(), cmap=cm.GnBu)
        p1.plot(X,Y,'g',marker='d')
        p1.set_xlabel('X1-location')
        p1.set_ylabel('X2-location')
        p1.set_title('Convergence Path')
        p2.semilogy(np.linspace(1,len(X),len(X)),hw2.jpath)
        p2.set_xlabel('Iteration number')
        p2.set_ylabel('distance from converged minimum')
        p2.set_title('Rate')
        plt.suptitle('Rosemary Teague, bracket_descent_test, initial guess ='+str(xg)+' \n Rate of convergence of a cost function')
        plt.tight_layout(pad=4)
        plt.savefig('hw231_'+str(i), dpi=700)

    if compare:
        plt.close('all')
        One,=plt.loglog(np.linspace(1,len(X),len(X)),hw2.jpath)
        xf2,jf2,outputn=newton_test(xg,timing=True)
        X2,Y2=outputn[1],outputn[2]
        d2=np.sqrt((X2-xf2[0])**2+(Y2-xf2[1])**2)
        print(np.linspace(1,len(X2),len(X2)),outputn[3])
        Two,=plt.loglog(np.linspace(1,len(X2),len(X2)),outputn[3])
        One.set_label('Bracket Descent')
        Two.set_label('Newton')
        plt.xlabel('Iteration number')
        plt.ylabel('Distance from converged minimum')
        plt.legend()
        plt.title('Rosemary Teague, bracket_descent_test, initial guess ='+str(xg)+' \n Comparison of Newton and Bracket Descent Methods')
        plt.savefig('hw232_'+str(i), dpi=700)

        output=(outputn[0],t34/N)

    return xf,jf,output


def performance(tol):
    """
    ============================================================================
    Assesses the performance of Bracket Descent and Scipy L-BFGS-B Methods
    ============================================================================

    Parameters
    ------------
    tol : float
        Determines the tolerance for minimization

    Returns
    ------------

    This function will produce 4 figures.
    The first 3 will represent a comparison of the precison of each method while
    the 4th will represent a comparison of the timing.
    The first three show the location of the computed minima for initial guesses
    of [-100,-3], [-50,-3], [-10,-3] and [-1,-3]. These are overlayed onto the
    original cost function; the Scipy L-BFGS-B results are represented by red
    diamonds while the Bracket Descent results are represented by blue diamonds.
    The three figures represent the cases when the noise amplitude is set to 0,
    1, and 10.

    The final figure consists of four subplots, the upper row represents the
    computational time taken for convergence, given an initial x starting point,
    while the lower represents the number of iterations requried. In each case
    the Scipy L-BFGS-B method is shown on the left and the Bracket descent is
    shown on the right. A legend on each plot differentiates the cases when the
    Noise Ampplitude is set to 0, 1, and 10.


    Trends Observed
    ----------------
    For all cases, the Scipy minimization function appears to be more consistent
    (to rely less on the initial guess) than the fortran Bracket Descent method.
    This is seen in figures hw241-hw243, where the B-D results are seen to cover
    a broader spead of final coordinated. These figures also illustrate that as
    the level of noise of the cost function is increased, the Scipy L-BFGS-B
    method becomes increasingly favourable over the Bracket descent approach,
    producing more precise results each time.
    This is a result of the lack of consideration for noise within the Bracket
    Descent method; that is to say that any random fluctations which result in
    two neighbouring points (along the convergence path) lying within the
    tolerance limit will be assumed to be the true minimum of the function as
    defined by the B-D method. However, it is likely that the Scipy L-BFGS-B
    method is adapted to smooth out noisy functions and hence find the true
    minimum more reliably.

    A consideration of figure hw244, however, demonstrates an advantage of the
    B-D method over the Scipy L-BGFS-B minimization in the form of timing. It can
    be seen that despite requiring more iterations before converging to within a
    set tolerance, the total computational time is less to within a factor of 10.



    """
    plt.close('all')

    count=0
    hw2.tol=tol
    nintb=[];nintl=[]; tlbfgsb=[]; txfbd=[]; lbfgsx=[]; lbfgsy=[]; xfbdx=[]; xfbdy=[];
    cost.c_noise=True

    for cost.c_noise_amp in [0., 1., 10.]:
        count=count+1

        for [x,y] in [[-100.,-3.],[-50.,-3.],[-10.,-3.],[-1.,-3.]]:
            t12=0;t34=0
            for i in range(0,1000):
                t1=time()
                scipy.optimize.minimize(cost.costj, [x,y], method='L-BFGS-B' ,tol=tol)
                t2=time()
                t12=t12+(t2-t1)

                t3=time()
                hw2.bracket_descent([x,y])
                t4=time()
                t34=t34+(t4-t3)

            tlbfgsb.append(t12/1000); txfbd.append(t34/1000)
            info=scipy.optimize.minimize(cost.costj, [x,y], method='L-BFGS-B' ,tol=tol)
            xfbd,jfbd,i2=hw2.bracket_descent([x,y])

            # print('method:              ', 'Fortran Bracket Descent')
            # print('Value:               ', jfbd)
            # print('number of iterations:', i2)
            # print('x:                   ', xfbd)
            # print('c_noise:             ', cost.c_noise)
            # print('   ')
            #print(info)

            x=info.x
            lbfgsx.append(x[0])
            lbfgsy.append(x[1])
            xfbdx.append(xfbd[0])
            xfbdy.append(xfbd[1])

            nint=info.nit
            nintl.append(nint)
            nintb.append(i2)

        Minx=1+(min([min(xfbdx[(count-1)*4:count*4]),min(lbfgsx[(count-1)*4:count*4])])-1)*1.1
        Maxx=1+(max([max(xfbdx[(count-1)*4:count*4]),max(lbfgsx[(count-1)*4:count*4])])-1)*1.1
        Miny=1+(min([min(xfbdy[(count-1)*4:count*4]),min(lbfgsy[(count-1)*4:count*4])])-1)*1.1
        Maxy=1+(max([max(xfbdy[(count-1)*4:count*4]),max(lbfgsy[(count-1)*4:count*4])])-1)*1.1
        [X,Y]=np.linspace(Minx,Maxx,200),np.linspace(Miny,Maxy,200)
        #calculate noiseless cost function at each point on 2D grid
        j=[[cost.costj([xi,yi]) for xi in X] for yi in Y]
        #create contour plots of cost functions with and without noise
        fig, p4 = plt.subplots()
        cp = p4.contourf(X, Y, j, locator=ticker.LogLocator(), cmap=cm.GnBu)
        cbar = fig.colorbar(cp)
        BD,=p4.plot(xfbdx[(count-1)*4:count*4],xfbdy[(count-1)*4:count*4],'b',linestyle='None',marker='d',markersize=6)
        Scipy,=p4.plot(lbfgsx[(count-1)*4:count*4],lbfgsy[(count-1)*4:count*4],'r',linestyle='None',marker='d',markersize=6)
        BD.set_label('Fortran Bracket Descent')
        Scipy.set_label('Scipy optimize L-BFGS-B')
        plt.legend(loc='upper left', fontsize='small')
        plt.suptitle('Rosemary Teague, performance \n Comparison of converged values, Noise='+str(int(cost.c_noise_amp)))
        #plt.tight_layout(pad=5)
        plt.savefig('hw24'+str(count), dpi=700)
        print(tlbfgsb)

    plt.close('all')
    f4, (p414,p424) = plt.subplots(2,2,sharey=True)
    one,=p414[0].plot(tlbfgsb[:4],[np.abs(-100.),np.abs(-50.),np.abs(-10.),np.abs(-1.)],'r',marker='x',markersize=12)
    two,=p414[0].plot(tlbfgsb[4:8],[np.abs(-100.),np.abs(-50.),np.abs(-10.),np.abs(-1.)],'m',marker='x',markersize=12)
    three,=p414[0].plot(tlbfgsb[8:],[np.abs(-100.),np.abs(-50.),np.abs(-10.),np.abs(-1.)],'#c79fef',marker='x',markersize=12)
    one.set_label('No Noise')
    two.set_label('Noise = 1.0')
    three.set_label('Noise = 10.0')
    p414[0].set_title('Scipy Optimise L-BFGS-B')
    p414[0].set_xlabel('Time Taken')
    p414[0].legend( loc = 'upper right', fontsize = 'x-small')
    p414[0].xaxis.set_ticks(np.linspace(min(tlbfgsb),max(tlbfgsb),3))
    p414[0].ticklabel_format(useOffset=False)
    uno,=p414[1].plot(txfbd[:4],[np.abs(-100.-xfbdx[0]),np.abs(-50.-xfbdx[1]),np.abs(-10.-xfbdx[2]),np.abs(-1.-xfbdx[3])],'b',marker='x',markersize=12)
    dos,=p414[1].plot(txfbd[4:8],[np.abs(-100.-xfbdx[4]),np.abs(-50.-xfbdx[5]),np.abs(-10.-xfbdx[6]),np.abs(-1.-xfbdx[7])],'g',marker='x',markersize=12)
    tres,=p414[1].plot(txfbd[8:],[np.abs(-100.-xfbdx[8]),np.abs(-50.-xfbdx[9]),np.abs(-10.-xfbdx[10]),np.abs(-1.-xfbdx[11])],'c',marker='x',markersize=12)
    uno.set_label('No Noise')
    dos.set_label('Noise = 1.0')
    tres.set_label('Noise = 10.0')
    p414[1].set_title('Fortran Bracket Descent')
    p414[1].set_xlabel('Time Taken')
    p414[1].legend(loc = 'upper left', fontsize = 'x-small')
    p414[1].xaxis.set_ticks(np.linspace(min(txfbd),max(txfbd),3))
    one1,=p424[0].plot(nintl[:4],[np.abs(-100.-lbfgsx[0]),np.abs(-50.-lbfgsx[1]),np.abs(-10.-lbfgsx[2]),np.abs(-1.-lbfgsx[3])],'r',marker='x',markersize=12)
    two2,=p424[0].plot(nintl[4:8],[np.abs(-100.-lbfgsx[4]),np.abs(-50.-lbfgsx[5]),np.abs(-10.-lbfgsx[6]),np.abs(-1.-lbfgsx[7])],'m',marker='x',markersize=12)
    three3,=p424[0].plot(nintl[8:],[np.abs(-100.-lbfgsx[8]),np.abs(-50.-lbfgsx[9]),np.abs(-10.-lbfgsx[10]),np.abs(-1.-lbfgsx[11])],'#c79fef',marker='x',markersize=12)
    one1.set_label('No Noise')
    two2.set_label('Noise = 1.0')
    three3.set_label('Noise = 10.0')
    p424[0].set_xlabel('Number of Iterations')
    p424[0].legend( loc = 'upper left', fontsize = 'x-small')
    p424[0].ticklabel_format(useOffset=False)
    uno1,=p424[1].plot(nintb[:4],[np.abs(-100.-xfbdx[0]),np.abs(-50.-xfbdx[1]),np.abs(-10.-xfbdx[2]),np.abs(-1.-xfbdx[3])],'b',marker='x',markersize=12)
    dos2,=p424[1].plot(nintb[4:8],[np.abs(-100.-xfbdx[4]),np.abs(-50.-xfbdx[5]),np.abs(-10.-xfbdx[6]),np.abs(-1.-xfbdx[7])],'g',marker='x',markersize=12)
    tres3,=p424[1].plot(nintb[8:],[np.abs(-100.-xfbdx[8]),np.abs(-50.-xfbdx[9]),np.abs(-10.-xfbdx[10]),np.abs(-1.-xfbdx[11])],'c',marker='x',markersize=12)
    uno1.set_label('No Noise')
    dos2.set_label('Noise = 1.0')
    tres3.set_label('Noise = 10.0')
    p424[1].set_xlabel('Number of Iterations')
    p424[1].legend(loc = 'upper left', fontsize = 'x-small')
    f4.text(0.04, 0.5, 'Initial x-distance from Converged minimum', va='center', rotation='vertical')
    plt.suptitle('Rosemary Teague, performance \n Time taken for values to converge',fontsize='large')
    plt.tight_layout(pad=3.5, h_pad=1,w_pad=1)
    plt.savefig('hw244', dpi=700)



if __name__ == '__main__':
    #
    # visualize(200,200)
    #
    # newton_test([10.,10.],display=True,i=1)
    # newton_test([5.,5.],display=True,i=2)
    # newton_test([2.,2.],display=True,i=3)
    #
    # bracket_descent_test([10.,10.],display=True,compare=True,i=1)
    # bracket_descent_test([5.,5.],display=True,compare=True,i=2)
    # bracket_descent_test([2.,2.],display=True,compare=True,i=3)

    performance(10**(-6))
