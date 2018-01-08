import numpy as np
import random
from scipy.misc import comb
import matplotlib.pyplot as plt

def bernoulli(p):
	r = random.random()
	if r < p:
		return 1
	return 0

def gen(sample_ps, s):
	n = len(sample_ps)
	samples = np.zeros((n, s))
	for i in xrange(n):
		for j in xrange(s):
			samples[i][j] = bernoulli(sample_ps[i])
	return samples

def uniform():
	return random.random()

def moments(samples, k):
	n, s = samples.shape
	counts = np.sum(samples, axis=1)
	return [np.mean(np.asarray([float(comb(counts[j], i)) / float(comb(s, i)) for j in xrange(n)])) for i in xrange(1, k + 1)]

def empirical_moments(samples, k):
	n, s = samples.shape
	counts = np.sum(samples, axis=1)
	return [np.mean(np.asarray([(float(counts[j]) / float(s)) ** i for j in xrange(n)])) for i in xrange(1, k + 1)]

def interpolate_poly(moments):
	k = len(moments)
	moments = [1] + moments
	return np.dot(np.linalg.inv(np.asarray([[1.0 / (i + j + 1) for i in xrange(k + 1)] for j in xrange(k + 1)])), np.transpose(np.asarray(moments)))

def graph_f(c):
	f = np.poly1d(c[::-1])
	x_mesh = np.linspace(0, 1, 100)
	y_mesh = f(x_mesh)
	plt.plot(x_mesh, y_mesh)
	plt.axis([0, 1, 0, max(y_mesh) * 1.2])
	plt.show()

def interpolate_histog(moments, m):
    k = len(moments)
    from cvxopt import matrix, solvers
    A = np.zeros((m + 2 * k + 3, m + k + 1))
    B = np.zeros(m + 2 * k + 3)
    C = np.zeros(m + k + 1)
    # set C (optimization)
    for i in xrange(m + 1, m + k + 1):
        C[i] = 1.0
    # set B (constraints)
    B[0] = 1.0
    B[1] = -1.0
    for i in xrange(k):
        B[m + 3 + (2 * i)] = moments[i] # reweight this
        B[m + 3 + (2 * i) + 1] = -1.0 * moments[i] # reweight this
    # set A (linear coefficients)
    for i in xrange(m + 1):
        A[0][i] = 1.0
        A[1][i] = -1.0
        A[2 + i][i] = -1.0

    for j in xrange(k):
        A[m + 3 + (2 * j)][m + 1 + j] = -1.0
        A[m + 3 + (2 * j) + 1][m + 1 + j] = -1.0
        for i in xrange(m + 1):
            A[m + 3 + (2 * j)][i] = (((float(i) / float(m))) ** (j + 1)) # reweight this
            A[m + 3 + (2 * j) + 1][i] = (-1.0 * ((float(i) / float(m))) ** (j + 1)) # reweight this
    # print A
    A = np.transpose(A)
    # print B
    # print C
    sol = solvers.lp(matrix(np.ndarray.tolist(C)), matrix(np.ndarray.tolist(A)), matrix(np.ndarray.tolist(B)))
    print np.dot(C, np.asarray(sol['x']))
    return sol['x']