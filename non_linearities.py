import theano.tensor as T


def relu(x):
    return T.switch(x > 0, x, 0)


class NonLinearity:
    RELU = "rectifier"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


def softmax(x):
    return T.exp(x)/(T.exp(x).sum(1, keepdims=True))


def get_non_linearity_fn(nonlinearity):
        if nonlinearity == NonLinearity.SIGMOID:
            return T.nnet.sigmoid
        elif nonlinearity == NonLinearity.RELU:
            return relu
        elif nonlinearity == NonLinearity.TANH:
            return T.tanh
        elif nonlinearity == NonLinearity.SOFTMAX:
            return softmax  # T.nnet.softmax
        elif nonlinearity is None:
            return None


def get_non_linearity_str(nonlinearity):
        if nonlinearity == T.nnet.sigmoid:
            return NonLinearity.SIGMOID
        elif nonlinearity == relu:
            return NonLinearity.RELU
        elif nonlinearity == T.tanh:
            return NonLinearity.TANH
        elif nonlinearity == T.nnet.softmax:
            return None  # we do not use any non-linearity.
        elif nonlinearity == softmax:
            return None  # we do not use any non-linearity.
        elif nonlinearity is None:
            return None
        else:
            raise ValueError("Unknown non-linearity")


class CostType:
    MeanSquared = "MeanSquaredCost"
    CrossEntropy = "CrossEntropy"
    NegativeLogLikelihood = "NegativelogLikelihood"
