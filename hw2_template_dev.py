"""
Rosemary Teague
00828351

Assumes cost.f90 has been compiled with f2py to generate the module,
hw2mod.so (filename may also be of form hw2mod.xxx.so where xxx is system-dependent text) using
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from hw2mod import cost
from hw2mod import hw2

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
    plt.show()



def newton_test(xg,display=False):
    """ Use Newton's method to minimize cost function defined in cost module
    Input variable xg is initial guess for location of minimum. When display
    is true, a figure illustrating the convergence of the method should be
    generated

    Output variables: xf -- computed location of minimum, jf -- computed minimum
    Further output can be added to the tuple, output, as needed. It may also
    be left empty.
    """
    output = ()


    return xf,jf,output


def bracket_descent_test(xg,display=False):
    """ Use bracket-descent to minimize cost function defined in cost module
    Input variable xg is initial guess for location of minimum. When display
    is true, 1-2 figures comparing the B-D and Newton steps should be generated

    Output variables: xf -- computed location of minimum, jf -- computed minimum
    """

    return xf,jf


def performance():
    """ Assess performance of B-D and L-BFGS-B methods. Add input/output as
    needed
    """


if __name__ == '__main__':
    #Add code here to call newton_test, bracket_descent_test, performance
    performance()
    visualize(200,200)
