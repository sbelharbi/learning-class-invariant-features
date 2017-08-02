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
from basic_layer import Layer
from ae import Autoencoder
from non_linearities import NonLinearity, CostType, relu, get_non_linearity_str
from normalization import BatchNormLayer


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


class HiddenLayer(Layer):
    def __init__(self, input, input1, input2, input_vl, n_in, n_out, W=None,
                 b=None,
                 activation=T.tanh, rng=None, hint=None, use_hint=False,
                 intended_to_be_corrupted=False, corrupt_input_l=0.,
                 use_sparsity=False, use_sparsity_in_pred=False,
                 use_unsupervised=False, use_batch_normalization=False):
        """
        Typical hidden layer of an MLP: units are fully connected and have
        tangente hyperbolic activation function. Weight matrix (W) is of shape
        (n_in, n_out) and the bias vector (b) is of shape (nout,).

        Hidden unit activation is given by: tanh(dot(input, w)+ b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initiaze the weights.

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimension of the input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation:  Non linearity to be applied in the hidden layer.
        """
        if rng is None:
            rng = np.random.RandomState()

        super(HiddenLayer, self).__init__(
            input, input1, input2, input_vl, n_in, n_out,
            activation=activation,
            rng=rng, hint=hint, use_hint=use_hint,
            intended_to_be_corrupted=intended_to_be_corrupted,
            corrupt_input_l=corrupt_input_l,
            use_sparsity=use_sparsity,
            use_sparsity_in_pred=use_sparsity_in_pred,
            use_batch_normalization=use_batch_normalization)
        self.reset_layer()

        if W is not None:
            self.W = W

        if b is not None:
            self.b = b

        self.params = [self.W, self.b]
        if self.use_batch_normalization:
            # we normalize the output of the layer, not its input.
            # it does not matter the size of the minibatch (10).
            self.bn = BatchNormLayer([100, n_out])

        self.setup_outputs(input)
        self.setup_outputs_vl(input_vl)
        self.setup_outputs1(input1)
        self.setup_outputs2(input2)
        # Create the associated auto-encoder: tied-wights AE.
        self.use_unsupervised = use_unsupervised
        self.ae = Autoencoder(
            input=input, nvis=n_in, nhid=n_out, cost_type=CostType.MeanSquared,
            nonlinearity=get_non_linearity_str(activation), W=self.W, b=self.b,
            tied_weights=True, reverse=False)

    def setup_outputs(self, input):
        # lin_output = T.dot(input, self.W) + self.b
        if self.intended_to_be_corrupted:
            warnings.warn("Input 1 Will be corrupted!!!!!!")
            lin_output = T.dot(
                self.get_corrupted_input(input), self.W) + self.b
        else:
            lin_output = T.dot(input, self.W) + self.b

        # Normalize the linear transformation, (if there is bn)
        if self.use_batch_normalization:
            assert self.bn is not None
            lin_output = self.bn.get_output_for(
                lin_output, deterministic=False, batch_norm_use_averages=False,
                batch_norm_update_averages=True)
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output))
        if self.use_sparsity_in_pred:
            assert self.use_sparsity
            self.output = self.output * self.sparser

    def setup_outputs_vl(self, input):
        """Setup the outputs for the test.
        Specifically for the batch normalization output.
        """
        lin_output = T.dot(input, self.W) + self.b
        # Normalize the linear transformation.
        if self.use_batch_normalization:
            assert self.bn is not None
            lin_output = self.bn.get_output_for(
                lin_output, deterministic=False, batch_norm_use_averages=False,
                batch_norm_update_averages=True)
        self.output_vl = (
            lin_output if self.activation is None
            else self.activation(lin_output))
        if self.use_sparsity_in_pred:
            assert self.use_sparsity
            self.output = self.output * self.sparser

    def setup_outputs1(self, input):
        if self.intended_to_be_corrupted:
            warnings.warn("Input 1 Will be corrupted!!!!!!")
            lin_output = T.dot(
                self.get_corrupted_input(input), self.W) + self.b
        else:
            lin_output = T.dot(input, self.W) + self.b
        # Batch normalization
        if self.use_batch_normalization:
            assert self.bn is not None
            lin_output = self.bn.get_output_for(
                lin_output, deterministic=False,
                batch_norm_use_averages=False,
                batch_norm_update_averages=False)
        # We compute the distance over the linear transformation.
#        self.output1 = lin_output
        self.output1 = (
            lin_output if self.activation is None
            else self.activation(lin_output))
        if self.use_sparsity_in_pred:
            assert self.use_sparsity
            self.output1 = self.output1 * self.sparser

    def setup_outputs2(self, input):
        if self.intended_to_be_corrupted:
            warnings.warn("Input 2 Will be corrupted!!!!!!")
            lin_output = T.dot(
                self.get_corrupted_input(input), self.W) + self.b
        else:
            lin_output = T.dot(input, self.W) + self.b
        # Batch normalization
        if self.use_batch_normalization:
            assert self.bn is not None
            lin_output = self.bn.get_output_for(
                lin_output, deterministic=False,
                batch_norm_use_averages=False,
                batch_norm_update_averages=False)
        # We compute the distance over the linear transformation.
#        self.output2 = lin_output
        self.output2 = (
            lin_output if self.activation is None
            else self.activation(lin_output))
        if self.use_sparsity_in_pred:
            assert self.use_sparsity
            self.output2 = self.output2 * self.sparser

    def get_outputs(self, input):
        self.setup_outputs(input)
        return self.output

    def get_outputs1(self, input):
        self.setup_outputs1(input)
        return self.output1

    def get_outputs2(self, input):
        self.setup_outputs2(input)
        return self.output2

    def _squared_magn(self, x):
        """Returns the sum of the squared values of an array."""
        return (x**2).sum(axis=1)

    def _magnitude(self, x):
        """Returns the magnitude of an array."""
        return T.sqrt(T.maximum(self._squared_magn(x), 1e-7))
        # np.finfo(theano.config.floatX).tiny))

    def get_arc_cosine_penalty(self):
        """Calculate the arccosine distance in [0, 1].
        0: the two vectors are very similar. (have the same orientation)
        1: the two vectors are very disimilar (have the opposite orientation).
        The cosine similarity does not take in consideration the magnitude
        of the vectors. It considers only thier orientation (angle).
        Therefore, two vectors are similar if they have the same angle.
        See: https://en.wikipedia.org/wiki/Cosine_similarity
        """
        # tiny value:
#        tiny = sharedX_value(np.finfo(dtype=theano.config.floatX).tiny,
#                             name="tiny")
        # the gradient of sqrt at 0 is undefinded (nan).
        # use a tiny value instead of 0.
        # OLD SOLUTION
#        denom = T.sqrt(
#            T.sum(self.output1**2, axis=1) * T.sum(self.output2**2, axis=1))
#        nomin = (self.output1 * self.output2).sum(1)
#        cosine = nomin/denom  # the cosine betwen the two vectors
#        pi = sharedX_value(np.pi, name="pi")
#        minus1 = sharedX_value(-1., name="minus1")
#        plus1 = sharedX_value(1. - np.finfo(dtype=theano.config.floatX).eps,
#                              name="plus1")
#        # Need to be clipped. accos() gives nan when sin is close to 1.
#        angle = T.arccos(T.clip(
#            cosine, minus1.get_value(), plus1.get_value()))/pi
        # OLD SOLUTION
#        plus1 = sharedX_value(1. - np.finfo(dtype=theano.config.floatX).eps,
#                              name="plus1")
        pi = sharedX_value(np.pi, name="pi")
        cosine = T.clip(((self.output1 * self.output2).sum(axis=1) / (
            self._magnitude(self.output1) * self._magnitude(self.output2))),
            -1, 1 - 1e-7)
        angle = T.clip(T.arccos(cosine) / pi, 0, 1)

        return angle

    def get_l2_penalty(self, ind=0):
        """calculate the Euclidean distance between the two outputs."""
        dif = (self.output1 - self.output2)
        if self.use_sparsity:
            dif = dif * self.sparser
        if ind == 0:
            return (dif**2).sum(1)
        elif ind == 1:
            return (dif**2).mean(1)
        else:
            raise ValueError("ind error.")

    def get_l1_penalty(self, ind=0):
        """calculate the Manhattan distance between the two outputs."""
        dif = (self.output1 - self.output2)
        if self.use_sparsity:
            dif = dif * self.sparser
        if ind == 0:
            return (abs(dif)).sum(1)
        elif ind == 1:
            return (abs(dif)).mean(1)
        else:
            raise ValueError("ind error.")

    def get_contrastive(self, sim, margin):
        distance = ((self.output1 - self.output2)**2).sum(1)
        converge = (1. - sim) * distance
        contraste = sim * T.maximum(0, margin - distance)

        return converge + contraste

    def get_divergence(self, sim, margin):
        distance = ((self.output1 - self.output2)**2).sum(1) ** (1/2.)
        contraste = sim * T.maximum(0, margin - distance)

        return contraste

    def insepct_get_l1_conv(self, sim, margin):
        return (1. - sim) * self.get_l1_penalty(ind=1)

    def inscpect_get_l1_div(self, sim, margin):
        distance = ((self.output1 - self.output2)**2).sum(1)
        contraste = sim * T.maximum(0, margin - distance)
        return contraste

    def inspect_get_l1_distance(self, sim, margin):
        distance = ((self.output1 - self.output2)**2).sum(1)
        d = sim * distance
        return d

    def get_penalty(self, sim, margin):
        if self.hint is "l1sum":
            return (1. - sim) * self.get_l1_penalty(ind=0)
        elif self.hint is "l1mean":
            return (1. - sim) * self.get_l1_penalty(ind=1)
        elif self.hint is "l2sum":
            return (1. - sim) * self.get_l2_penalty(ind=0)
        elif self.hint is "l2mean":
            return (1. - sim) * self.get_l2_penalty(ind=1)
        elif self.hint is "arccos":
            return (1. - sim) * self.get_arc_cosine_penalty()
        elif self.hint is "l1sumcos":
            return (1. - sim) * (
                self.get_l1_penalty(ind=0) + self.get_arc_cosine_penalty())
        elif self.hint is "l1meancos":
            return (1. - sim) * (
                self.get_l1_penalty(ind=1) + self.get_arc_cosine_penalty())
        elif self.hint is "l2sumcos":
            return (1. - sim) * (
                self.get_l2_penalty(ind=0) + self.get_arc_cosine_penalty())
        elif self.hint is "l2meancos":
            return (1. - sim) * (
                self.get_l2_penalty(ind=0) + self.get_arc_cosine_penalty())
        elif self.hint is "contrastive":
            return self.get_contrastive(sim, margin)
        elif self.hint is "divergence":
            return self.get_divergence(sim, margin)
        else:
            raise ValueError("self.hint uknonw!!!!")


class LeNetConvPoolLayer_hint(HiddenLayer):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, input1, input2, input_vl,
                 filter_shape, image_shape, poolsize=(2, 2),
                 activation=T.tanh, hint="l1mean",
                 use_hint=False,
                 intended_to_be_corrupted=False,
                 corrupt_input_l=0.,
                 use_sparsity=False,
                 use_sparsity_in_pred=False,
                 use_unsupervised=False,
                 use_batch_normalization=False):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert hint is not None
        assert image_shape[1] == filter_shape[1]
        self.corrupt_input_l = sharedX_value(corrupt_input_l, name="cor_l")
        self.intended_to_be_corrupted = intended_to_be_corrupted
        self.rng = np.random.RandomState(123)
        self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))
        self.input = input
        # keep track of model input
        self.input = input
        self.input1 = input1  # x1
        self.input2 = input2  # x2
        self.input_vl = input_vl  # bn input used for validation.
        self.sparser = None
        self.activation = activation
        self.hint = hint
        self.use_hint = use_hint
        self.use_sparsity = use_sparsity
        self.use_sparsity_in_pred = use_sparsity_in_pred
        self.use_unsupervised = use_unsupervised
        self.ae = None  # no need for cnn... for now.
        self.use_batch_normalization = use_batch_normalization
        self.bn = None
        # the bn is applied before the pooling. (and after the linear op.)
        # output_shape = [batch size, num output maps, img height, img width]
        map_size_h = (image_shape[2] - filter_shape[2] + 1)
        map_size_w = (image_shape[3] - filter_shape[3] + 1)
        output_shape = [image_shape[0], filter_shape[0], map_size_h,
                        map_size_w]
        if self.use_batch_normalization:
            self.bn = BatchNormLayer(output_shape)
        # assert self.use_batch_normalization is False

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            name="W",
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name="b", borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        conv_out1 = conv2d(
            input=self.input1,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        conv_out2 = conv2d(
            input=self.input2,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        conv_out_vl = conv2d(
            input=self.input_vl,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        # BN
        if self.bn is not None:
            conv_out = self.bn.get_output_for(
                conv_out, deterministic=False,
                batch_norm_use_averages=False,
                batch_norm_update_averages=True)
            conv_out1 = self.bn.get_output_for(
                conv_out1, deterministic=False,
                batch_norm_use_averages=False,
                batch_norm_update_averages=False)
            conv_out2 = self.bn.get_output_for(
                conv_out2, deterministic=False,
                batch_norm_use_averages=False,
                batch_norm_update_averages=False)
            conv_out_vl = self.bn.get_output_for(
                conv_out_vl, deterministic=False,
                batch_norm_use_averages=False,
                batch_norm_update_averages=True)
        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        pooled_out1 = pool.pool_2d(
            input=conv_out1,
            ds=poolsize,
            ignore_border=True
        )
        pooled_out2 = pool.pool_2d(
            input=conv_out2,
            ds=poolsize,
            ignore_border=True
        )
        pooled_out_vl = pool.pool_2d(
            input=conv_out_vl,
            ds=poolsize,
            ignore_border=True
        )
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.output1_non_fl = activation(
            pooled_out1 + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output2_non_fl = activation(
            pooled_out2 + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_vl = activation(
            pooled_out_vl + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.output1 = self.output1_non_fl.flatten(2)
        self.output2 = self.output2_non_fl.flatten(2)
        if self.sparser is None:
            dim_h = int((image_shape[2] - filter_shape[1] + 1) / poolsize[0])
            dim_w = int((image_shape[3] - filter_shape[1] + 1) / poolsize[1])
            dim_out = filter_shape[0] * dim_h * dim_w
            s_values = np.ones(
                (dim_out),
                dtype=theano.config.floatX)
            self.sparser = theano.shared(value=s_values, name='sparser',
                                         borrow=True)

        # store parameters of this layer
        self.params = [self.W, self.b]


class LogisticRegressionLayer(Layer):
    """
    Multi-class logistic regression layer.
    The logistic regression is fully described by a weight matrix ::math:`W`
    and a bias vector ::math: `b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probablity.
    """
    def __init__(self, input, n_in, n_out, is_binary=False, threshold=0.4,
                 rng=None):
        """
        Initialize the parameters of the logistic regression.
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which
        the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie (number of classes)
        """
        self.activation = T.nnet.sigmoid
        self.threshold = threshold
        super(LogisticRegressionLayer, self).__init__(
            input,
            n_in,
            n_out,
            self.activation,
            rng)

        self.reset_layer()

        self.is_binary = is_binary
        if n_out == 1:
            self.is_binary = True
        # The number of classes
        self.n_classes_seen = np.zeros(n_out)
        # The number of the wrong classification madefor the class i
        self.n_wrong_classif_made = np.zeros(n_out)

        self.reset_conf_mat()

        # Compute vector class-membership probablities in symbolic form
        # self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+ self.b)
        self.p_y_given_x = self.get_class_memberships(self.input)

        if not self.is_binary:
            # Compute prediction as class whose probability is maximal
            # in symbolic form
            self.y_decision = T.argmax(self.p_y_given_x, axis=1)
        else:
            # If the probability is greater than the specified threshold
            # assign to the class 1, otherwise it is 0. Which alos can be
            # checked if p(y=1|x) > threshold.
            self.y_decision = T.gt(T.flatten(self.p_y_given_x), self.threshold)

        self.params = [self.W, self.b]

    def reset_conf_mat(self):
        """
        Reset the confusion matrix.
        """
        self.conf_mat = np.zeros(shape=(self.n_out, self.n_out),
                                 dtype=np.dtype(int))

    def negative_log_likelihood(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
            \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                    \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example
            the correct label.
        Note: We use the mean instead of the sum so that the learning rate
            is less dependent of the batch size.
        """
        if self.is_binary:
            return -T.mean(T.log(self.p_y_given_x))
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def crossentropy_categorical(self, y):
        """
        Find the categorical cross entropy.
        """
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))

    def crossentropy(self, y):
        """
        use the theano nnet cross entropy function. Return the mean.
        Note: self.p_y_given_x is (batch_size, 1) but y is (batch_size,).
        In order to establish the compliance, we should flatten the
        p_y_given_x.
        """
        return T.mean(
            T.nnet.binary_crossentropy(T.flatten(self.p_y_given_x), y))

    def get_class_memberships(self, x):
        lin_activation = T.dot(x, self.W) + self.b
        if self.is_binary:
            # return the sigmoid value
            return T.nnet.sigmoid(lin_activation)
        # else retunr the softmax
        return T.nnet.softmax(lin_activation)

    def update_conf_mat(self, y, p_y_given_x):
        """
        Update the confusion matrix with the given true labels and estimated
        labels.
        """
        if self.n_out == 1:
            y_decision = (p_y_given_x > self.threshold)
        else:
            y_decision = np.argmax(p_y_given_x, axis=1)
        for i in xrange(y.shape[0]):
            self.conf_mat[y[i]][y_decision[i]] += 1

    def errors(self, y):
        """
        returns a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch. Zero one loss
        over the size of the minibatch.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
        correct label.
        """
        if y.ndim != self.y_decision.ndim:
            raise TypeError("y should have the same shape as self.y_decision",
                            ('y', y.type, "y_decision", self.y_decision.type))
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # The T.neq operator returns a vector of 0s and 1s, where:
            # 1 represents a mistake in classification
            return T.mean(T.neq(self.y_decision, y))
        else:
            raise NotImplementedError()

    def raw_prediction_errors(self, y):
        """
        Returns a binary array where each each element indicates if the
        corresponding sample has been correctly classified (0) or not (1) in
        the minibatch.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
        correct label.
        """
        if y.ndim != self.y_decision.ndim:
            raise TypeError("y should have the same shape as self.y_decision",
                            ('y', y.type, "y_decision", self.y_decision.type))
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            # The T.neq operator returns a vector of 0s and 1s, where:
            # 1 represents a mistake in classification
            return T.neq(self.y_decision, y)
        else:
            raise NotImplementedError()

    def error_per_calss(self, y):
        """
        Return an array where each value is the error for the corresponding
        classe in the minibatch.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
        correct label.
        """
        if y.ndim != self.y_decision.ndim:
            raise TypeError("y should have the same shape as self.y_decision",
                            ('y', y.type, "y_decision", self.y_decision.type))
        if y.dtype.startswith('int') or y.dtype.startswith('uint'):
            y_decision_res = T.neq(self.y_decision, y)
            for (i, y_decision_r) in enumerate(y_decision_res):
                self.n_classes_seen[y[i]] += 1
                if y_decision_r:
                    self.n_wrong_classif_made[y[i]] += 1
            pred_per_class = self.n_wrong_classif_made / self.n_classes_seen
            return T.mean(y_decision_res), pred_per_class
        else:
            raise NotImplementedError()
