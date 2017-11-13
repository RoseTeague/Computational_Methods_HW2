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
        plt.savefig('hw32'+str(i), dpi=700)




    jf=cost.costj(xf)
    return xf,jf


def performance():
    """ Assess performance of B-D and L-BFGS-B methods. Add input/output as
    needed
    """
    plt.close()
    lbfgsx=[]
    lbfgsy=[]
    xfbdx=[]
    xfbdy=[]

    for [x,y] in [[-100.,3.],[-50.,-3.],[-10.,-3.],[-1.,-3.]]:
        info=scipy.optimize.minimize(cost.costj, [x,y], method='L-BFGS-B')

        xfbd,jfbd,i2=hw2.bracket_descent([x,y])
        print('method:              ', 'Fortran Bracket Descent')
        print('Value:               ', jfbd)
        print('number of iterations:', i2)
        print('x:                   ', xfbd)
        print('   ')
        print(info)

        x=scipy.optimize.OptimizeResult.values(info)[4]
        lbfgsx.append(x[0])
        lbfgsy.append(x[1])
        xfbdx.append(xfbd[0])
        xfbdy.append(xfbd[1])

    f, (p41,p42) = plt.subplots(1,2)
    p41.plot(lbfgsx,lbfgsy,'rx')
    p41.set_ylim(0.9999987,0.9999997)
    p41.xaxis.set_ticks(np.arange(0.9999993,0.9999999,0.0000003))
    p41.ticklabel_format(useOffset=False)
    p42.plot(xfbdx,xfbdy,'bx')
    p42.set_ylim(0.997,1.0003)
    p42.xaxis.set_ticks(np.arange(0.9985000,1.00005,0.0005))
    plt.suptitle('Rosemary Teague, performance \n Comparison of converged values')
 #plt.xlim([0.999,1.001])
    plt.savefig('hw241', dpi=700)
    plt.tight_layout(pad=3.5)

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
