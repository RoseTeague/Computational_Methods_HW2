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
from hw2mod2 import cost
from hw2mod2 import hw2
from time import time

def visualize(Nx,Ny,xrange=10,yrange=10):
    """Display cost function with and without noise on an Ny x Nx grid
    """
    #Create 2D array of points
    [X,Y]=np.linspace(-xrange,xrange,Nx),np.linspace(-yrange,yrange,Ny)

    #calculate noiseless cost function at each point on 2D grid
    cost.c_noise=False
    j=[[cost.costj([xi,yi]) for yi in Y] for xi in X]

    #calculate noisey cost function at each point in 2D grid.
    cost.c_noise = True
    cost.c_noise_amp = 1
    jn=[[cost.costj([xi,yi]) for yi in Y] for xi in X]

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
    plt.title('Rosemary Teague, Visualize \n 2D cost function, noise amplitude='+str(cost.c_noise_amp))
    plt.savefig('hw212', dpi = 700)
    #plt.show()



def newton_test(xg,display=False,i=1):
    """ Use Newton's method to minimize cost function defined in cost module
    Input variable xg is initial guess for location of minimum. When display
    is true, a figure illustrating the convergence of the method should be
    generated

    Output variables: xf -- computed location of minimum, jf -- computed minimum
    Further output can be added to the tuple, output, as needed. It may also
    be left empty.
    """
    cost.c_noise=False
    hw2.tol=10**(-6)
    hw2.itermax=1000
    hw2.newton(xg)
    X,Y=hw2.xpath
    xf=[X[-1],Y[-1]]
    jf=hw2.jpath[-1]


    if display:
        f, (p1,p2) = plt.subplots(1,2)
        p1.plot(X,Y)
        p1.set_xlabel('X1-location')
        p1.set_ylabel('X2-location')
        p2.plot(np.linspace(0,len(X)-1,len(X)),np.sqrt((X-xf[0])**2+(Y-xf[1])**2))
        p2.set_xlabel('Iteration number')
        p2.set_ylabel('distance from converged minimum')
        plt.suptitle('Rosemary Teague, Newton_test, initial guess ='+str(xg)+' \n Rate of convergence of a cost function')
        plt.tight_layout(pad=4)
        plt.savefig('hw22'+str(i), dpi=700)


    output = ()
    jf=cost.costj(xf)
    return xf,jf,output


def bracket_descent_test(xg,display=False,i=1):
    """ Use bracket-descent to minimize cost function defined in cost module
    Input variable xg is initial guess for location of minimum. When display
    is true, 1-2 figures comparing the B-D and Newton steps should be generated

    Output variables: xf -- computed location of minimum, jf -- computed minimum
    """
    cost.c_noise=False
    hw2.tol=10**(-6)
    hw2.itermax=1000
    hw2.bracket_descent(xg)
    X,Y=hw2.xpath
    xf=[X[-1],Y[-1]]
    jf=hw2.jpath[-1]


    if display:
        f, (p1,p2) = plt.subplots(1,2)
        p1.plot(X,Y)
        p1.set_xlabel('X1-location')
        p1.set_ylabel('X2-location')
        p2.plot(np.linspace(0,len(X)-1,len(X)),np.sqrt((X-xf[0])**2+(Y-xf[1])**2))
        p2.set_xlabel('Iteration number')
        p2.set_ylabel('distance from converged minimum')
        plt.suptitle('Rosemary Teague, bracket_descent_test, initial guess ='+str(xg)+' \n Rate of convergence of a cost function')
        plt.tight_layout(pad=4)
        plt.savefig('hw23'+str(i), dpi=700)




    jf=cost.costj(xf)
    return xf,jf


def performance():
    """ Assess performance of B-D and L-BFGS-B methods. Add input/output as
    needed
    """
    plt.close()
    lbfgsx=[]; lbfgsy=[]; xfbdx=[]; xfbdy=[]; tlbfgsb=[]; txfbd=[]

    hw2.tol=10**(-14)
    cost.c_noise=True
    for cost.c_noise_amp in [0., 1., 10.]:
        for [x,y] in [[-100.,-3.],[-50.,-3.],[-10.,-3.],[-1.,-3.]]:

            t1=time()
            info=scipy.optimize.minimize(cost.costj, [x,y], method='L-BFGS-B')
            t2=time()
            tlbfgsb.append(t2-t1)

            t3=time()
            xfbd,jfbd,i2=hw2.bracket_descent([x,y])
            t4=time()
            txfbd.append(t4-t3)
            # print('method:              ', 'Fortran Bracket Descent')
            # print('Value:               ', jfbd)
            # print('number of iterations:', i2)
            # print('x:                   ', xfbd)
            # print('c_noise:             ', cost.c_noise)
            # print('   ')
            # print(info)

            x=scipy.optimize.OptimizeResult.values(info)[4]
            lbfgsx.append(x[0]-1.0)
            lbfgsy.append(x[1]-1.0)
            xfbdx.append(xfbd[0]-1.0)
            xfbdy.append(xfbd[1]-1.0)

    plt.close()
    f, (p41,p42) = plt.subplots(1,2)
    p41.plot(lbfgsx[0:4],lbfgsy[0:4],'r',marker='x',markersize=12)
    p41.set_title('Scipy Optimise L-BFGS-B')
    p41.set_xlabel('x-1')
    p41.set_ylabel('y-1')
    p41.ticklabel_format(useOffset=False)
    p42.plot(xfbdx[0:4],xfbdy[0:4],'b',marker='x',markersize=12)
    p42.set_title('Fortran Bracket Descent')
    p42.set_xlabel('x-1')
    p42.set_ylabel('y-1')
    p42.yaxis.set_ticks(np.linspace(min(xfbdy[0:4]),max(xfbdy[0:4]),3))
    p42.xaxis.set_ticks(np.linspace(min(xfbdx[0:4]),max(xfbdx[0:4]),3))
    p42.ticklabel_format(useOffset=False)
    plt.suptitle('Rosemary Teague, performance \n Comparison of converged values')
    plt.tight_layout(pad=4)
    plt.savefig('hw241', dpi=700)

    plt.close()
    f2, (p412,p422) = plt.subplots(1,2)
    p412.plot(lbfgsx[4:8],lbfgsy[4:8],'m',marker='x',markersize=12)
    p412.set_title('Scipy Optimise L-BFGS-B')
    p412.set_xlabel('x-1')
    p412.set_ylabel('y-1')
    p412.ticklabel_format(useOffset=False)
    p422.plot(xfbdx[4:8],xfbdy[4:8],'g',marker='x',markersize=12)
    p422.set_title('Fortran Bracket Descent')
    p422.set_xlabel('x-1')
    p422.set_ylabel('y-1')
    plt.suptitle('Rosemary Teague, performance \n Comparison of converged values with noise=1')
    plt.tight_layout(pad=3.5)
    plt.savefig('hw242', dpi=700)


    plt.close()
    f3, (p413,p423) = plt.subplots(1,2)
    p413.plot(lbfgsx[8:],lbfgsy[8:],'#c79fef',marker='x',markersize=12)
    p413.set_title('Scipy Optimise L-BFGS-B')
    p413.set_xlabel('x')
    p413.set_ylabel('y')
    p413.ticklabel_format(useOffset=False)
    p423.plot(xfbdx[8:],xfbdy[8:],'c',marker='x',markersize=12)
    p423.set_title('Fortran Bracket Descent')
    p423.set_xlabel('x')
    p423.set_ylabel('y')
    plt.suptitle('Rosemary Teague, performance \n Comparison of converged values with noise=10')
    plt.tight_layout(pad=3.5)
    plt.savefig('hw243', dpi=700)

    plt.close()
    f4, (p414,p424) = plt.subplots(1,2)
    p414.plot(tlbfgsb[:4],[-100.,-50.,-10.,-1.],'r',marker='x',markersize=12)
    p414.plot(tlbfgsb[4:8],[-100.,-50.,-10.,-1.],'m',marker='x',markersize=12)
    p414.plot(tlbfgsb[8:],[-100.,-50.,-10.,-1.],'#c79fef',marker='x',markersize=12)
    p414.set_title('Scipy Optimise L-BFGS-B')
    p414.set_xlabel('Time Taken')
    p414.set_ylabel('Initial guess')
    p414.xaxis.set_ticks(np.linspace(min(tlbfgsb),max(tlbfgsb),3))
    p414.ticklabel_format(useOffset=False)
    p424.plot(txfbd[:4],[-100.,-50.,-10.,-1.],'b',marker='x',markersize=12)
    p424.plot(txfbd[4:8],[-100.,-50.,-10.,-1.],'g',marker='x',markersize=12)
    p424.plot(txfbd[8:],[-100.,-50.,-10.,-1.],'c',marker='x',markersize=12)
    p424.set_title('Fortran Bracket Descent')
    p424.set_xlabel('Time Taken')
    p424.xaxis.set_ticks(np.linspace(min(txfbd),max(txfbd),3))
    plt.suptitle('Rosemary Teague, performance \n Time taken for values to converge')
    plt.tight_layout(pad=3.5)
    plt.savefig('hw244', dpi=700)



if __name__ == '__main__':
    #Add code here to call newton_test, bracket_descent_test, performance
    performance()
    visualize(200,200)
    newton_test([10.,10.],display=True,i=1)
    newton_test([5.,5.],display=True,i=2)
    newton_test([2.,2.],display=True,i=3)

    bracket_descent_test([10.,10.],display=True,i=1)
    bracket_descent_test([5.,5.],display=True,i=2)
    bracket_descent_test([2.,2.],display=True,i=3)
