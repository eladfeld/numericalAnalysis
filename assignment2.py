"""
In this assignment you should find the intersection points for two functions.
"""

from collections.abc import Iterable


def find_roots(f: callable, a, b, max_error):
    diff = (b - a) / 2
    if diff < max_error / 8:
        return []
    mid = (a + b) / 2
    f_of_mid = f(mid)
    if abs(f_of_mid) < max_error:
        return [mid]
    return find_roots(f, a, mid, max_error) + find_roots(f, mid, b, max_error)


def elads_method_to_find_roots(f, a, b, max_error):
    result = []
    crosser = a
    last_x = a
    last_y = a
    slope = 1
    found_root_point = 1

    while crosser < b:
        f_of_the_crosser = f(crosser)
        if abs(f_of_the_crosser) < max_error:
            result.append(crosser)
            found_root_point += 1
        elif found_root_point > 1:
            found_root_point -= 1
        last_x = crosser
        crosser += min(max(max_error / 2, max_error / slope * found_root_point), max_error * 30)
        slope = abs((last_y - f_of_the_crosser) / (last_x - crosser))
        last_y = f_of_the_crosser
    return result


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution
        diff_func = lambda x: f1(x) - f2(x)
        X = elads_method_to_find_roots(diff_func, a, b, maxerr)

        return X


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_find_avarage_time(self):
        number_of_times = 10
        ass2 = Assignment2()

        times = [0.] * number_of_times
        for i in range(number_of_times):
            t = time.time()
            f1, f2 = randomIntersectingPolynomials(100)
            X = ass2.intersections(f1, f2, -10, 10)
            times[i] = time.time() - t
            for x in X:
                self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        print(sum(times) / len(times))


if __name__ == "__main__":
    unittest.main()
