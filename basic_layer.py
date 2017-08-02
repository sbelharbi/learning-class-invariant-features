# Based on: https://github.com/caglar/autoencoders.git
# http://www-etud.iro.umontreal.ca/~gulcehrc/
from __future__ import division
import numpy as np
import theano
from theano import tensor as T
import warnings


from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from non_linearities import NonLinearity, CostType, relu, get_non_linearity_str


def sharedX_value(value, name=None, borrow=None, dtype=None):
    """Share a single value after transforming it to floatX type.

    value: a value
    name: variable name (str)
    borrow: boolean
    dtype: the type of the value when shared. default: theano.config.floatX
    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(
        theano._asarray(value, dtype=dtype), name=name, borrow=borrow)


class Layer(object):
    """
    A general base layer class for neural network.
    for training, the layer takes a pair of samples (input1, input2).
    input1 and input2 belong to the same class.
    input: sample for the supervised part.
    input1: first samples
    input2: second sample
    intended_to_be_corrupted: boolean. If True, we create a corruptor
        for the input. This indicates that may be at some point in the
        future the inputs of this layer may be corrupted.
    corrupt_input_l: Float. If !=0., only the input1 and input2 will be
        corrupted.
    NOTE:
        Basically, only the input of the first layer is corrupted. There is
            no interest/reason in corrupting the intermediate inputs.
    """
    def __init__(self,
                 input,
                 input1,
                 input2,
                 input_vl,
                 n_in,
                 n_out,
                 activation=T.nnet.sigmoid,
                 sparse_initialize=False,
                 num_pieces=1,
                 non_zero_units=25,
                 rng=None,
                 hint="l1mean",
                 use_hint=False,
                 intended_to_be_corrupted=False,
                 corrupt_input_l=0.,
                 use_sparsity=False,
                 use_sparsity_in_pred=False,
                 use_batch_normalization=False):

        assert hint is not None
        self.num_pieces = num_pieces
        self.corrupt_input_l = sharedX_value(corrupt_input_l, name="cor_l")
        self.intended_to_be_corrupted = intended_to_be_corrupted
        self.rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))
        self.input = input
        self.input1 = input1  # x1
        self.input2 = input2  # x2
        self.input_vl = input_vl  # bn input used for validation.
        self.n_in = n_in
        self.n_out = n_out
        self.rng = rng
        self.sparse_initialize = sparse_initialize
        self.non_zero_units = non_zero_units
        self.W = None
        self.b = None
        self.sparser = None
        self.activation = activation
        self.hint = hint
        self.use_hint = use_hint
        self.use_sparsity = use_sparsity
        self.use_sparsity_in_pred = use_sparsity_in_pred
        self.use_batch_normalization = use_batch_normalization
        self.bn = None

    def reset_layer(self):
        """
        initailize the layer's parameters to random.
        """
        if self.W is None:
            if self.sparse_initialize:
                W_values = self.sparse_initialize_weights()
            else:
                if self.activation == theano.tensor.tanh:
                    born = np.sqrt(6. / (self.n_in + self.n_out))
                else:
                    born = 4 * np.sqrt(6. / (self.n_in + self.n_out))
                W_values = np.asarray(self.rng.uniform(
                    low=-born,
                    high=born,
                    size=(self.n_in, self.n_out)),
                    dtype=theano.config.floatX)

            self.W = theano.shared(value=W_values, name='W', borrow=True)

        if self.b is None:
            b_values = np.zeros(int(self.n_out/self.num_pieces),
                                dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)

        if self.sparser is None:
            s_values = np.ones(
                int(self.n_out/self.num_pieces), dtype=theano.config.floatX)
            self.sparser = theano.shared(value=s_values, name='sparser',
                                         borrow=True)
        # The layer parameters
        self.params = [self.W, self.b]

    def get_corrupted_input(self, input):
        """This function keeps 1-self.corruption_input_l entries of the inputs
        the  same and zero-out randomly selected subset of size
        self.coruption_input_l.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - self.corrupt_input_l,
                                        dtype=theano.config.floatX) * input

    def sparse_initialization_weights(self):
        """
        Implement the sparse initialization technique as described in
        J. Marten, 'Deep learning via Hessian-free optimization', ICML, 2010.
        http://icml2010.haifa.il.ibm.com/papers/458.pdf
        """
        W = []
        mu, sigma = 0, 1/self.non_zero_units

        for i in xrange(self.n_in):
            row = np.zeros(self.n_out)
            non_zeros = self.rng.normal(mu, sigma, self.non_zero_units)
            # non_zeros /= non_zeros.sum()
            non_zero_idxs = self.rng.permutation(
                self.n_out)[0:self.non_zero_units]
            for j in xrange(self.non_zero_units):
                row[non_zero_idxs[j]] = non_zeros[j]
            W.append(row)
        W = np.asarray(W, dtype=theano.config.floatX)
        return W
