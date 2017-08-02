# Based on: https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/
# normalization.py#L120-L320
import theano
import numpy as np
from theano import tensor as T


class BatchNormLayer(object):
    """ Implementation of batch normalization from the paper:
    Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """
    def __init__(self, input_shape, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta=0., gamma=0, mean=0, inv_std=1):
        self.input_shape = input_shape
        if axes == 'auto':
            # default normalizationover lla but the not the second axis.
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha

        # create params
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input shape for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = 0.
        else:
            value = np.ones(shape, dtype=theano.config.floatX) * beta
            self.beta = theano.shared(value=value.astype(theano.config.floatX),
                                      name="beta", borrow=True)

        if gamma is None:
            self.gamma = 0.
        else:
            value = np.ones(shape, dtype=theano.config.floatX) * gamma
            self.gamma = theano.shared(
                value=value.astype(theano.config.floatX),
                name="gamma", borrow=True)

        value = np.ones(shape, dtype=theano.config.floatX) * mean
        self.mean = theano.shared(value=value.astype(theano.config.floatX),
                                  name="mean", borrow=True)

        value = np.ones(shape, dtype=theano.config.floatX) * inv_std
        self.inv_std = theano.shared(value=value.astype(theano.config.floatX),
                                     name="inv_std", borrow=True)
        self.params = [self.beta, self.gamma]
        self.stats = [self.mean, self.inv_std]

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # decide whether to use the sotred averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics.
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std
        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * inv_std) + beta
        return normalized
