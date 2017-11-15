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

def visualize(Nx,Ny,xmin=-10,xmax=10,ymin=-10,ymax=10):
    """Display cost function with and without noise on an Ny x Nx grid
    """
    #Create 2D array of points
    [X,Y]=np.linspace(xmin,xmax,Nx),np.linspace(ymin,ymax,Ny)

    #calculate noiseless cost function at each point on 2D grid
    cost.c_noise=False
    j=[[cost.costj([xi,yi]) for xi in X] for yi in Y]

    #calculate noisey cost function at each point in 2D grid.
    cost.c_noise = True
    cost.c_noise_amp = 1
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
        p2.plot(np.linspace(0,len(X)-1,len(X)),np.sqrt((X-xf[0])**2+(Y-xf[1])**2))
        p2.set_xlabel('Iteration number')
        p2.set_ylabel('distance from converged minimum')
        p2.set_title('Rate')
        plt.suptitle('Rosemary Teague, Newton_test, initial guess ='+str(xg)+' \n Convergence of a cost function')
        plt.tight_layout(pad=4)
        plt.savefig('hw22'+str(i), dpi=700)


    output = ()
    jf=cost.costj(xf)
    return xf,jf,output


def bracket_descent_test(xg,display=False,compare=False,i=1):
    """ Use bracket-descent to minimize cost function defined in cost module
    Input variable xg is initial guess for location of minimum. When display
    is true, 1-2 figures comparing the B-D and Newton steps should be generated

    Output variables: xf -- computed location of minimum, jf -- computed minimum
    """
    cost.c_noise=False
    hw2.tol=10**(-6)
    hw2.itermax=1000
    t1=0;t2=0;t3=0;t4=0
    t1 = time()
    hw2.bracket_descent(xg)
    t2 = time()
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
        p2.plot(np.linspace(0,len(X)-1,len(X)),d1)
        p2.set_xlabel('Iteration number')
        p2.set_ylabel('distance from converged minimum')
        p2.set_title('Rate')
        plt.suptitle('Rosemary Teague, bracket_descent_test, initial guess ='+str(xg)+' \n Rate of convergence of a cost function')
        plt.tight_layout(pad=4)
        plt.savefig('hw231_'+str(i), dpi=700)

    if compare:
        plt.close('all')
        t3= time()
        xf2,jf2,i2=hw2.newton(xg)
        t4 = time()
        X2,Y2=hw2.xpath
        d2=np.sqrt((X2-xf2[0])**2+(Y2-xf2[1])**2)
        One,=plt.loglog(np.linspace(0,len(X)-1,len(X)),d1)
        Two,=plt.loglog(np.linspace(0,len(X2)-1,len(X2)),d2)
        One.set_label('Bracket Descent')
        Two.set_label('Newton')
        plt.xlabel('Iteration number')
        plt.ylabel('Distance from converged minimum')
        plt.legend()
        plt.title('Rosemary Teague, bracket_descent_test, initial guess ='+str(xg)+' \n Comparison of Newton and Bracket Descent Methods')
        plt.savefig('hw232_'+str(i), dpi=700)

    jf=cost.costj(xf)
    return xf,jf,t2-t1,t4-t3


def performance():
    """ Assess performance of B-D and L-BFGS-B methods. Add input/output as
    needed
    """
    plt.close()

    count=0
    hw2.tol=10**(-14)
    nintb=[];nintl=[]; tlbfgsb=[]; txfbd=[];lbfgsx=[]; lbfgsy=[]; xfbdx=[]; xfbdy=[];
    cost.c_noise=True
    for cost.c_noise_amp in [0., 1., 10.]:
        count=count+1

        for [x,y] in [[-100.,-3.],[-50.,-3.],[-10.,-3.],[-1.,-3.]]:
            t12=0;t34=0
            for i in range(0,10):
                t1=time()
                info=scipy.optimize.minimize(cost.costj, [x,y], method='L-BFGS-B')
                t2=time()
                t12=t12+(t2-t1)

                t3=time()
                xfbd,jfbd,i2=hw2.bracket_descent([x,y])
                t4=time()
                t34=t34+(t4-t3)

            tlbfgsb.append(t12/10); txfbd.append(t34/10)

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
        plt.tight_layout(pad=5)
        plt.savefig('hw24'+str(count), dpi=700)

    plt.close('all')
    f4, (p414,p424) = plt.subplots(2,2,sharey=True)
    one,=p414[0].plot(tlbfgsb[:4],[np.abs(-100.-lbfgsx[0]),np.abs(-50.-lbfgsx[1]),np.abs(-10.-lbfgsx[2]),np.abs(-1.-lbfgsx[3])],'r',marker='x',markersize=12)
    two,=p414[0].plot(tlbfgsb[4:8],[np.abs(-100.-lbfgsx[4]),np.abs(-50.-lbfgsx[5]),np.abs(-10.-lbfgsx[6]),np.abs(-1.-lbfgsx[7])],'m',marker='x',markersize=12)
    three,=p414[0].plot(tlbfgsb[8:],[np.abs(-100.-lbfgsx[0]),np.abs(-50.-lbfgsx[1]),np.abs(-10.-lbfgsx[2]),np.abs(-1.-lbfgsx[3])],'#c79fef',marker='x',markersize=12)
    one.set_label('No Noise')
    two.set_label('Noise = 1.0')
    three.set_label('Noise = 10.0')
    p414[0].set_title('Scipy Optimise L-BFGS-B')
    p414[0].set_xlabel('Time Taken')
    p414[0].legend( loc = 'upper left', fontsize = 'x-small')
    p414[0].xaxis.set_ticks(np.linspace(min(tlbfgsb),max(tlbfgsb),3))
    p414[0].ticklabel_format(useOffset=False)
    uno,=p414[1].plot(txfbd[:4],[np.abs(-100.-xfbdx[0]),np.abs(-50.-xfbdx[1]),np.abs(-10.-xfbdx[2]),np.abs(-1.-xfbdx[3])],'b',marker='x',markersize=12)
    dos,=p414[1].plot(txfbd[4:8],[np.abs(-100.-xfbdx[4]),np.abs(-50.-xfbdx[5]),np.abs(-10.-xfbdx[6]),np.abs(-1.-xfbdx[7])],'g',marker='x',markersize=12)
    tres,=p414[1].plot(txfbd[8:],[np.abs(-100.-xfbdx[0]),np.abs(-50.-xfbdx[1]),np.abs(-10.-xfbdx[2]),np.abs(-1.-xfbdx[3])],'c',marker='x',markersize=12)
    uno.set_label('No Noise')
    dos.set_label('Noise = 1.0')
    tres.set_label('Noise = 10.0')
    p414[1].set_title('Fortran Bracket Descent')
    p414[1].set_xlabel('Time Taken')
    p414[1].legend(loc = 'upper left', fontsize = 'x-small')
    p414[1].xaxis.set_ticks(np.linspace(min(txfbd),max(txfbd),3))
    one1,=p424[0].plot(nintl[:4],[np.abs(-100.-lbfgsx[0]),np.abs(-50.-lbfgsx[1]),np.abs(-10.-lbfgsx[2]),np.abs(-1.-lbfgsx[3])],'r',marker='x',markersize=12)
    two2,=p424[0].plot(nintl[4:8],[np.abs(-100.-lbfgsx[4]),np.abs(-50.-lbfgsx[5]),np.abs(-10.-lbfgsx[6]),np.abs(-1.-lbfgsx[7])],'m',marker='x',markersize=12)
    three3,=p424[0].plot(nintl[8:],[np.abs(-100.-lbfgsx[0]),np.abs(-50.-lbfgsx[1]),np.abs(-10.-lbfgsx[2]),np.abs(-1.-lbfgsx[3])],'#c79fef',marker='x',markersize=12)
    one1.set_label('No Noise')
    two2.set_label('Noise = 1.0')
    three3.set_label('Noise = 10.0')
    p424[0].set_xlabel('Number of Iterations')
    p424[0].legend( loc = 'upper left', fontsize = 'x-small')
    p424[0].ticklabel_format(useOffset=False)
    uno1,=p424[1].plot(nintb[:4],[np.abs(-100.-xfbdx[0]),np.abs(-50.-xfbdx[1]),np.abs(-10.-xfbdx[2]),np.abs(-1.-xfbdx[3])],'b',marker='x',markersize=12)
    dos2,=p424[1].plot(nintb[4:8],[np.abs(-100.-xfbdx[4]),np.abs(-50.-xfbdx[5]),np.abs(-10.-xfbdx[6]),np.abs(-1.-xfbdx[7])],'g',marker='x',markersize=12)
    tres3,=p424[1].plot(nintb[8:],[np.abs(-100.-xfbdx[0]),np.abs(-50.-xfbdx[1]),np.abs(-10.-xfbdx[2]),np.abs(-1.-xfbdx[3])],'c',marker='x',markersize=12)
    uno1.set_label('No Noise')
    dos2.set_label('Noise = 1.0')
    tres3.set_label('Noise = 10.0')
    p424[1].set_xlabel('Number of Iterations')
    p424[1].legend(loc = 'upper left', fontsize = 'x-small')
    f4.text(0.04, 0.5, 'Initial x-distance from Converged minimum', va='center', rotation='vertical')
    plt.suptitle('Rosemary Teague, performance \n Time taken for values to converge',fontsize='large')
    plt.tight_layout(pad=3.5, h_pad=0,w_pad=0)
    plt.savefig('hw244', dpi=700)



if __name__ == '__main__':
    #Add code here to call newton_test, bracket_descent_test, performance
    visualize(200,200)

    newton_test([10.,10.],display=True,i=1)
    newton_test([5.,5.],display=True,i=2)
    newton_test([2.,2.],display=True,i=3)

    bracket_descent_test([10.,10.],display=True,compare=True,i=1)
    bracket_descent_test([5.,5.],display=True,compare=True,i=2)
    bracket_descent_test([2.,2.],display=True,compare=True,i=3)

    performance()
