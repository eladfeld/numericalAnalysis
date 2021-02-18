"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

import numpy as np
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, deltas):
        self.deltas = deltas
        pass

    def contour(self, n: int):
        Xs = [0.] * n
        Ys = [0.] * n
        for i in range(n):
            fraction = i * len(self.deltas) / n
            place = int(fraction)
            x, y = self.deltas[place](fraction - place)
            Xs[i] = x
            Ys[i] = y

        return np.stack((Xs, Ys), axis=1)

    def area(self) -> np.float32:
        ass4 = Assignment4()
        return ass4.area(self.contour)


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        num_of_points = 10000
        points = contour(num_of_points)
        area = 0.0
        j = num_of_points - 1

        for i in range(num_of_points):
            area += (points[j, 0] + points[i, 0]) * (points[j, 1] - points[i, 1])
            j = i

        return abs(np.float32(area) / 2)

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        n = 10

        samples = [sample() for i in range(n)]

        x_sum = 0
        y_sum = 0
        for p in samples:
            x_sum += p[0]
            y_sum += p[1]
        x_center = x_sum / n
        y_center = y_sum / n

        def sort_by(point):
            x = point[0] - x_center
            y = point[0] - y_center
            return -np.arctan(y / x)

        samples.sort(key=sort_by)

        matrix = np.zeros((n - 1, n - 1))

        for i in range(1, len(matrix) - 1):
            matrix[i, i] = 4
        for i in range(0, len(matrix) - 2):
            matrix[i, i + 1] = 1
            matrix[i + 1, i] = 1

        matrix[0, 0] = 2
        matrix[-1, -1] = 7
        matrix[-1, -2] = 2
        matrix[-2, -1] = 1

        sol = [(0., 0.)] * (n - 1)

        sol[0] = np.array(samples[0]) + 2 * np.array(samples[1])
        sol[n - 2] = 8 * np.array(samples[n - 2]) + np.array(samples[n - 1])

        for i in range(1, n - 2):
            sol[i] = 4 * np.array(samples[i]) + 2 * np.array(samples[i + 1])

        As = np.linalg.solve(matrix, sol)
        Bs = [2 * np.array(samples[i]) - np.array(As[i]) for i in range(1, n - 1)]
        Bs += [np.array(As[-2]) + 2 * np.array(As[-1]) - 2 * np.array(Bs[-2])]

        Ds = [
            lambda t: np.array(samples[i]) * (1 - t) ** 3 + 3 * np.array(As[i]) * (1 - t) ** 2 + np.array(
                Bs[i]) * 3 * (1 - t) * t ** 2 + np.array(samples[i + 1]) * (t ** 3) for i in range(n - 1)]

        result = MyShape(Ds)
        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_area_from_contour(self):
        circ = Circle(cx=1, cy=1, radius=1, noise=0.0)
        ass4 = Assignment4()
        T = time.time()
        a_computed = ass4.area(contour=circ.contour, maxerr=0.1)
        T = time.time() - T
        a_true = circ.area()
        self.assertLess(abs((a_true - a_computed) / a_true), 0.1)


if __name__ == "__main__":
    unittest.main()
