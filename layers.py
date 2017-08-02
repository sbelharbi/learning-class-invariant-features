from theano import tensor as T
import theano
import numpy
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from layer import HiddenLayer
from layer import LeNetConvPoolLayer_hint


def relu(x):
    return T.switch(x > 0, x, 0)


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


class IdentityHiddenLayer(object):
    """
    This is the identity layer. It takes the input and give it back as output.
    We will be using this layer just after the last convolution layer to applay
    a dropout.
    """
    def __init__(self, rng, input):
        self.input = input
        self.W = None
        self.b = None
        self.params = []
        self.output = input


def dropout_from_layer(rng, layer_output, p):
    """
    p: float. The probablity of dropping a unit.
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(99999))
    one = T.constant(1)
    retain_prob = one - p
    mask = srng.binomial(n=1, p=retain_prob, size=layer_output.shape,
                         dtype=layer_output.dtype)
    output = layer_output * mask

    return output


def localResponseNormalizationCrossChannel(incoming, alpha=1e-4,
                                           k=2, beta=0.75, n=5):
    """
    Implement the local response normalization cross the channels described
    in <ImageNet Classification with Deep Convolutional Neural Networks>,
    A.Krizhevsky et al. sec.3.3.
    Reference of the code:
    https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/
    normalization.py
    https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/normalize.py
    Parameters:
    incomping: The feature maps. (output of the convolution layer).
    alpha: float scalar
    k: float scalr
    beta: float scalar
    n: integer: number of adjacent channels. Must be odd.
    """
    if n % 2 == 0:
        raise NotImplementedError("Works only with odd n")

    input_shape = incoming.shape
    half_n = n // 2
    input_sqr = T.sqr(incoming)
    b, ch, r, c = input_shape
    extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
    input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],
                                input_sqr)
    scale = k
    for i in range(n):
        scale += alpha * input_sqr[:, i:i+ch, :, :]
    scale = scale ** beta

    return incoming / scale


class LRNCCIdentityLayer(IdentityHiddenLayer):
    def __init__(self, input, alpha=1e-4, k=2, beta=0.75, n=5):
        super(LRNCCIdentityLayer, self).__init__(rng=None, input=input)
        self.output = localResponseNormalizationCrossChannel(
            incoming=self.output, alpha=alpha, k=k, beta=beta, n=n)


class DropoutIdentityHiddenLayer(IdentityHiddenLayer):
    def __init__(self, rng, input, dropout_rate, rescale):
        """
        rescale: Boolean. Can be only used when applying dropout.
        """
        if rescale:
            one = T.constant(1)
            retain_prob = one - dropout_rate
            input /= retain_prob

        super(DropoutIdentityHiddenLayer, self).__init__(rng=rng, input=input)
        if dropout_rate > 0.:
            self.output = dropout_from_layer(rng, self.output, p=dropout_rate)


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, dropout_rate, rescale,
                 W=None, b=None, b_v=0., activation=None):
        """
        rescale: Boolean. Can be only used when applying dropout.
        """
        if rescale:
            one = T.constant(1)
            retain_prob = one - dropout_rate
            input /= retain_prob

        super(DropoutHiddenLayer, self).__init__(
            input=input, n_in=n_in, n_out=n_out, W=W, b=b,
            activation=activation, rng=rng)
        if dropout_rate > 0.:
            self.output = dropout_from_layer(rng, self.output, p=dropout_rate)


class LeNetConvPoolLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape,
                 poolsize=(2, 2), maxout=False, poolmaxoutfactor=2,
                 W=None, b=None, b_v=0., stride=(1, 1), LRN={
                     "app": False, "before": False, "alpha": 1e-4, "k": 2,
                     "beta": 0.75, "n": 5}):
        """
        Input:
            maxout: Boolean. Indicates if to do or not a maxout.
            poolmaxoutfactor: How many feature maps to maxout. The number of
                input feature maps must be a multiple of poolmaxoutfactor.
            allow_dropout_conv: Boolean. Allow or not the dropout in conv.
                layer. This maybe helpful when we want to use dropout only
                for fully connected layers.
            LRN: tuple (a, b) of booleans. a: apply or not the local response
                normalization. b: before (True) or after (False) the pooling.
            b_v: float. The initial value of the bias.
        """
        self.LRNCCIdentityLayer = None
        if maxout:
            assert poolmaxoutfactor == 2
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        if W is None:
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                name="w_conv",
                borrow=True
            )
        if b is None:
            b_v = (
                numpy.ones(
                    (filter_shape[0],)) * b_v).astype(theano.config.floatX)
            b = theano.shared(value=b_v, name="b_conv", borrow=True)

        self.W = W
        self.b = b
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            subsample=stride
        )
        # Local reponse normalization
        if LRN["app"] and LRN["before"]:
            self.LRNCCIdentityLayer = LRNCCIdentityLayer(
                conv_out, alpha=LRN["alpha"], k=LRN["k"], beta=LRN["beta"],
                n=LRN["n"])
            conv_out = self.LRNCCIdentityLayer.output
            print "LRN BEFORE pooling ..."

        if maxout:
            z = T.add(conv_out, self.b.dimshuffle('x', 0, 'x', 'x'))
            s = None
            for i in range(filter_shape[0]/poolmaxoutfactor):
                t = z[:, i::poolmaxoutfactor, :, :]
                if s is None:
                    s = t
                else:
                    s = T.maximum(s, t)
            z = s
            if poolsize not in [None, (1, 1)]:
                pooled_out = downsample.max_pool_2d(
                    input=z,
                    ds=poolsize,
                    ignore_border=True
                )
                self.output = pooled_out
            else:
                self.output = z
        else:
            if poolsize not in [None, (1, 1)]:
                pooled_out = downsample.max_pool_2d(
                    input=conv_out,
                    ds=poolsize,
                    ignore_border=True
                    )
                self.output = relu(
                    pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
                print "RELU..."
            else:
                # simple relu
                term = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
                self.output = T.switch(term > 0, term, 0 * term)
                print "RELU..."

        # Local reponse normalization
        if LRN["app"] and not LRN["before"]:
            self.LRNCCIdentityLayer = LRNCCIdentityLayer(
                self.output, alpha=LRN["alpha"], k=LRN["k"], beta=LRN["beta"],
                n=LRN["n"])
            self.output = self.LRNCCIdentityLayer.output
            print "LRN AFTER activation(of pooling)..."

        self.params = [self.W, self.b]


class DropoutLeNetConvPoolLayer(LeNetConvPoolLayer):
    def __init__(self, rng, input, filter_shape, image_shape, dropout_rate,
                 rescale, poolsize=(2, 2), stride=(1, 1),
                 LRN={
                     "app": False, "before": False, "alpha": 1e-4, "k": 2,
                     "beta": 0.75, "n": 5},
                 maxout=False, poolmaxoutfactor=2, W=None, b=None, b_v=0.):
        if rescale:
            one = T.constant(1)
            retain_prob = one - dropout_rate
            input /= retain_prob
        super(DropoutLeNetConvPoolLayer, self).__init__(
            rng=rng, input=input, filter_shape=filter_shape,
            image_shape=image_shape, poolsize=poolsize, stride=stride,
            LRN=LRN, maxout=maxout, poolmaxoutfactor=poolmaxoutfactor,
            W=W, b=b, b_v=b_v)
        if dropout_rate > 0.:
            self.output = dropout_from_layer(rng, self.output, p=dropout_rate)
