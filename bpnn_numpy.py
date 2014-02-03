# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>
#
# Conversion to numpy implementation with assorted other modifications 
# by Arun Luthra.

import numpy as np
import string

def rand(a,b):
    return (b-a)*np.random.random_sample() + a

def sigmoid(x):
    # Symmetrical sigmoid
    return np.tanh(x)
    # return 1./(1.+np.exp(-x)) # normal sigmoid, range 0 to 1

vsigmoid = np.vectorize(sigmoid)

def dsigmoid(y):
    return 1.0 - y**2

vdsigmoid = np.vectorize(dsigmoid)

class NN:
    def __init__(self, ni, nh, no):
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # Initialize numpy arrays of ones with default dtype float
        self.ai = np.ones((self.ni,1), dtype=float)
        self.ah = np.ones((self.nh,1), dtype=float)
        self.ao = np.ones((self.no,1), dtype=float)

        # initialize weights
        # Make random matrix with values in range [-0.2, 0.2)
        self.wi = (np.random.random_sample((self.ni, self.nh)) - 0.5) * 0.4
        # Make random matrix with values in range [-2., 2.)
        self.wo = (np.random.random_sample((self.nh, self.no)) - 0.5) * 4.
                
        # last change in weights for momentum
        self.ci = np.zeros((self.ni, self.nh))
        self.co = np.zeros((self.nh, self.no))

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError("Wrong number of inputs")

        # input activations
        for i in range(self.ni-1):
            self.ai[i,0] = inputs[i]

        # hidden activations
        # shapes: (nh,1) = (nh,sni) x (sni,1)
        self.ah = vsigmoid( self.wi.T.dot(self.ai) )

        # output activations
        # shapes: (no,1) = (no,nh) x (nh,1)
        self.ao = vsigmoid( self.wo.T.dot(self.ah) )

        return self.ao

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('Wrong number of target values')

        # Assume 'targets' has shape (no,1)
        error = targets - self.ao
        # vectorized sigmoid followed by element-wise multiplication with errors
        # (no,1) * constant:
        output_deltas = vdsigmoid(self.ao) * error

        # (nh,1) = (nh,no) matrix times (no,1) column vector:
        error = self.wo.dot(output_deltas)
        # (nh,1) * constant:
        hidden_deltas = vdsigmoid(self.ah) * error

        # (nh,no) matrix = (nh,1) column vector times (1,no) row vector:
        change = self.ah.dot(output_deltas.T)
        self.wo += N*change + M*self.co
        self.co = change

        # (self.ni,nh) = (self.ni,1) x (1,nh):
        change = self.ai.dot(hidden_deltas.T)
        # Each of these objects (except N,M) has shape (self.ni,nh):
        self.wi += N*change + M*self.ci
        # (self.ni,nh):
        self.ci = change

        # Vector subtraction, element-wise exponentiation, then sum over [self.no] elements 
        error = np.sum(0.5*(targets - self.ao)**2)
        return error

    def test(self, patterns):
        for p in patterns:
            a = p[1]
            b = self.update(p[0])
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])

        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        errs = []
        for i in range(iterations):
            error = 0.0
            # Do backpropagation training one data point at time:
            for p in patterns:
                inputs = p[0]
                if type(p[1]) == list:
                    targets = p[1]
                else: targets = list([p[1]])
                self.update(inputs)
                error = error + self.backPropagate(np.array(targets).reshape((len(targets),1)), N, M) 
            errs.append(error)
            if i % 100 == 0:
                print('error %-0.5f' % error)
        # return errs # Uncomment this to get training error as function of iteration

def normalize(data):
    # A slightly hacky normalization ncode
    colmin = np.min(data, axis=0)
    colmax = np.max(data, axis=0)
    colmean = np.mean(data, axis=0)
    colrange = colmax-colmin+0.001 # the hacky part
    #print(colmin,colmax,colmean,colrange)
    data_norm = data - colmean
    # if colrange > 0.001:
    data_norm = data_norm / colrange
    return data_norm

def irisdemo():
    from sklearn import datasets
    iris = datasets.load_iris()
    pattern = []

    data_norm = normalize(iris.data)

    for i,x in enumerate(data_norm):
        # Classify 3 different iris types with two tanh output nodes:
        # Encode class 0 as [0,0]
        # Encode class 1 as [1,0]
        # Encode class 2 as [0,1]
        targetpattern = [0.,0.]
        if iris.target[i] == 1:
            targetpattern = [1.,0.]
        elif iris.target[i] == 2:
            targetpattern = [0.,1.]

        pattern.append([ x, targetpattern])

    n = NN(4,9,2)
    n.train(pattern,iterations=200,N=0.03,M=0.3)
    n.test(pattern)

def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)

if __name__ == '__main__':
    pattern = demo()

