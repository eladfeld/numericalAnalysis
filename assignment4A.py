"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""
import math

import torch


def create_matrix_of_size_d(d: int):
    def choose(n, r):
        return math.factorial(n) / math.factorial(r) / math.factorial(n - r)

    M = np.zeros((d, d))

    for j in range(0, d):
        for i in range(0, j + 1):
            M[i, d - 1 - j] = (1.0 if (j - i) % 2 == 0 else -1.0) * choose(d - 1 - i, j - i) \
                              * choose(d - 1, i)

    return M


def bezierD(Md, Cfit, d):
    P = torch.tensor(Cfit)

    def f(t):
        T = torch.DoubleTensor([t ** i for i in range(d, -1, -1)])
        T = T.view(1, len(T))
        T = T.mm(Md)
        return T.mm(P)

    return f


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        n = d
        Xs = np.linspace(a, b, n)
        Ys = f(Xs)

        Md = create_matrix_of_size_d(d + 1)

        P = torch.DoubleTensor([Xs, Ys])
        
        t = torch.tensor(np.linspace(0.0, 1.0, n))
        T = torch.stack([t ** i for i in range(d, -1, -1)]).T

        Mdtens = torch.tensor(Md)
        Cfit = Mdtens.inverse().mm((T.T.mm(T)).inverse()).mm(T.T).mm(P.T)
        gamma = bezierD(Mdtens, Cfit, d)

        coeff = [0.] * (d + 1)
        for i in range(d + 1):
            s = 0
            for j in range(d + 1):
                s += Cfit[j][0] * Md[i, j]
            coeff[i] = s

        def result(x):
            coeff[-1] -= x
            roots = np.roots(coeff)

            root = [x for x in roots if (0 <= x <= 1 and np.isreal(x))]
            realNum = np.real(root[0])
            point = gamma(realNum)
            return point[0][1]

        return result


##########################################################################


from sampleFunctions import *


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
