# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 08:28:21 2015

@author: Soufiane Belharbi
"""
import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict

from tools import sharedX_value, sharedX_mtx
from tools import floatX


def norm_constraint(tensor_var, max_norm, norm_axes=None, epsilon=1e-7):
    """Max weight norm constraints and gradient clipping

    This takes a TensorVariable and rescales it so that incoming weight
    norms are below a specified constraint value. Vectors violating the
    constraint are rescaled so that they are within the allowed range.

    Parameters
    ----------
    tensor_var : TensorVariable
        Theano expression for update, gradient, or other quantity.
    max_norm : scalar
        This value sets the maximum allowed value of any norm in
        `tensor_var`.
    norm_axes : sequence (list or tuple)
        The axes over which to compute the norm.  This overrides the
        default norm axes defined for the number of dimensions
        in `tensor_var`. When this is not specified and `tensor_var` is a
        matrix (2D), this is set to `(0,)`. If `tensor_var` is a 3D, 4D or
        5D tensor, it is set to a tuple listing all axes but axis 0. The
        former default is useful for working with dense layers, the latter
        is useful for 1D, 2D and 3D convolutional layers.
        (Optional)
    epsilon : scalar, optional
        Value used to prevent numerical instability when dividing by
        very small or zero norms.

    Credit:
        https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py

    Returns
    -------
    TensorVariable
        Input `tensor_var` with rescaling applied to weight vectors
        that violate the specified constraints.


    Notes
    -----
    When `norm_axes` is not specified, the axes over which the norm is
    computed depend on the dimensionality of the input variable. If it is
    2D, it is assumed to come from a dense layer, and the norm is computed
    over axis 0. If it is 3D, 4D or 5D, it is assumed to come from a
    convolutional layer and the norm is computed over all trailing axes
    beyond axis 0. For other uses, you should explicitly specify the axes
    over which to compute the norm using `norm_axes`.
    """
    ndim = tensor_var.ndim

    if norm_axes is not None:
        sum_over = tuple(norm_axes)
    elif ndim == 2:  # DenseLayer
        sum_over = (0,)
    elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
        sum_over = tuple(range(1, ndim))
    else:
        raise ValueError(
            "Unsupported tensor dimensionality {}."
            "Must specify `norm_axes`".format(ndim)
        )

    dtype = np.dtype(theano.config.floatX).type
    norms = T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
    target_norms = T.clip(norms, 0, dtype(max_norm))
    constrained_output = \
        (tensor_var * (target_norms / (dtype(epsilon) + norms)))

    return constrained_output


class LearningRule():
    """ A `LearningRule` is a class that calculates the new parameters value
    using:
    a learning rate, the current parameters value and the current gradient.

    """
    def get_updates(self, learning_rate, params, grads, lr_scalers):
        """ Compute the current updates for the parameters.

        """

        raise NotImplementedError(
            str(type(self)) + " does not implement get_updates.")


class Momentum(LearningRule):
    """Implementation of the momentum as in the method described in section
    9 of [1]:'A Practical Guide to Training Restricted Boltzmann Machines',
    bu Geoffrey Hinton.(https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
    We implemented alos the formula presented in Imagenet paper:
    <ImageNet Classification with Deep Convolutional Neural Networks>,
    A.Krizhevsky et al. .
    More details in:
    [2]'On the importance of initialization and momentum in deep learning',
    I. Sutskever et al.
    [3]'Advances in optimizating recurrent networks', Y. Bengio et al.

    The model's parametes are updated such as:
    velocity_(t+1) := momentum * velocity_t -
        learning_rate * d cost / d param_t
    param_(t+1) := param_t + velocity_(t+1)

    Parameters:
        init_momentum: float
            Initial value of the momentum coefficient. It remains fisex unless
                used with 'MomentumAdjuster'.
        nesterov_momentum: boolean
            If True, uses the accelerated momentum technique described in [2,3]
        max_colm_norm: Boolean. The incoming weight vector corresponding to
            each hidden unit is constrained to have a maximum squared length of
            max_norm.
        max_norm: Float. The maximum norm.
    """
    def __init__(self, init_momentum, nesterov_momentum=False,
                 imagenet=False, imagenetDecay=5e-4, max_colm_norm=False,
                 max_norm=15.0):
        assert init_momentum >= 0., 'The initial momentum should be >=0.'
        assert init_momentum < 1., 'The initial momentum should be < 1.'

        self.momentum = sharedX_value(value=init_momentum, name="momentum",
                                      borrow=True)
        self.nesterov_momentum = nesterov_momentum
        self._first_time = True
        self.velocity = None  # tracks the velocity at the previous time
        self.imagenet = imagenet
        self.imagenetDecay = sharedX_value(value=imagenetDecay,
                                           name="imagenetDecay",
                                           borrow=True)
        self.max_colm_norm = max_colm_norm
        self.max_norm = max_norm

    def get_updates(self, learning_rate, params, grads, lr_scalers):
        """
        get the updates (params, and velocity)
        """
        # the initial velocity is zero.
        if self._first_time:
            self.velocity = [
                sharedX_mtx(
                    param.get_value() * 0.,
                    name='vel_'+param.name, borrow=True) for param in params]

        updates = []
        for (param, grad, vel, lr_sc) in zip(
                params, grads, self.velocity, lr_scalers):
            lr_scaled = learning_rate * lr_sc
            if self.imagenet:
                new_vel = self.momentum * vel -\
                    lr_scaled * self.imagenetDecay * param - lr_scaled * grad
            else:
                new_vel = self.momentum * vel - lr_scaled * grad

            updates.append((vel, new_vel))
            inc = new_vel
            # this is the equivalence of NAG in [3].3.5, eq [7].
            # It helps to avoid calculating the new grad(param+vel_(t-1)).
            # The only different from the paper is: momentum_(t)
            # which it's set to momentum_(t-1). If you develop the final inc,
            # you will find that it's equivalent to eq.[7] mentioned above.
            if self.nesterov_momentum:
                inc = self.momentum * new_vel - lr_scaled * grad

            new_param = param + inc
            if self.max_colm_norm and param.name in ["W", "w"]:
                new_param_final = norm_constraint(tensor_var=new_param,
                                                  max_norm=self.max_norm)
            else:
                new_param_final = new_param
            updates.append((param, new_param_final))

        # add the velocity updates to updates

        return updates


class MomentumLinearAdjusterOverEpoch(object):
    """A callback to adjust linearly the momentum on each frequence (EPOCH).
    It adjusts the momentum based on the number of the epochs seen.

    Parameters:
        final_momentum: float
            The momentum coefficient to use at the end of the learning.
        start: int
            The epoch on wich to start growing the momentum.
        saturate: int
            The epoch on wich to momentum should reach its final value.

    """
    def __init__(self, final_momentum, start, saturate):
        assert saturate >= start, "The momentum can not saturate before it "\
            "starts increasing. Please set a saturation value higher than the"\
            " start value."
        self._initialized = False
        self._count = 0
        self.saturate = saturate
        self.final_momentum = final_momentum
        self.start = start
        self.freq = 'epoch'  # it works only on epochs
        self._first_time = True

    def __call__(self, learning_rule, seen_epochs):
        """Update the momentum according to the number of the epochs already
        seen.

        Parameters:
            trainingAlgorithm: instance of
                training_algorithm.trainingAlgorithm,
                the current algorithm used for training the model.
        """
        # check
        if not hasattr(learning_rule, 'momentum'):
            raise ValueError(
                str(type(self))+' works only when the learning_rule '
                'specified in the training algorithm has the attribute '
                '<momentum>. For examples: "sarco.learning_rule.Momentum"')

        self._count = seen_epochs
        self._apply_momentum(learning_rule)

    def _apply_momentum(self, learning_rule):
        """Apply the momentum.
        """

        momentum = learning_rule.momentum
        if not self._initialized:
            self._init_momentum = momentum.get_value()
            self._initialized = True
        momentum.set_value(
            np.cast[theano.config.floatX](self.get_current_momentum()))

    def get_current_momentum(self):
        """Return the current momentum with the desired schedule.

        """
        w = self.saturate - self.start
        if w == 0:
            # saturate=start, jump straighforward to the final momentum value
            # if we exceeded the saturation, return the final momentum
            if self._count >= self.saturate:
                return self.final_momentum
            else:
                # else: (we didn't reach yet the saturation point),
                # return the initial momentum
                return self._init_momentum

        coef = float(self._count - self.start) / float(w)
        if coef < 0.:
            coef = 0.  # no effect
        if coef > 1.:
            coef = 1.

        cu_m = self._init_momentum * (1 - coef) + coef * self.final_momentum

        return cu_m


class AdaDelta(LearningRule):
    """Implement the ADADELTA algorithm of [1] to update the parameters
    of the model.
    Parameters:
        decay: float
            Decay rate in [1].

    Caution: the parameter 'epsilon' in [1] is the learning rate.
    So It would be better to use a small learning rate
        (maybe fixed all the learning process [we will see]
    [1]:'AdaDelta: An Adaptive Learning Rate Method', Zeiler M. )
    """
    def __init__(self, decay=0.95, max_colm_norm=False, max_norm=15.0):
        assert decay >= 0., 'The decay parameter in ' + str(type(self)) +\
            ' must be >= 0.'
        assert decay < 1., 'The decay parameter in ' + str(type(self)) +\
            ' must be < 1.'
        self.decay = decay
        self._first_time = True
        self.mean_square_grad = None
        self.mean_squar_dx = None
        self.max_colm_norm = max_colm_norm
        self.max_norm = max_norm

    def get_updates(self, learning_rate, params, grads, lr_scalers):
        """Compute the AdaDelta updates of the model's parameters.

        param_t := param_(t-1) + AdaDelta_update_t
        """
        if self._first_time:
            self.mean_square_grad = [
                sharedX_mtx(
                    param.get_value() * 0.,
                    name='mean_square_grad_'+param.name,
                    borrow=True) for param in params]
            self.mean_squar_dx = [
                sharedX_mtx(
                    param.get_value() * 0.,
                    name='mean_square_dx_'+param.name,
                    borrow=True) for param in params]
            self._first_time = False

        updates = []
        for (param, grad, mean_square_grad, mean_squar_dx, lr_sc) in zip(
                params, grads, self.mean_square_grad, self.mean_squar_dx,
                lr_scalers):
            # Calculate the running average gradient: E[g^2]_t
            new_mean_square_grad = (
                self.decay * mean_square_grad + (1 - self.decay) * T.sqr(grad))

            # The update: delta_x_t
            lr_scaled = learning_rate * lr_sc
            epsilon = lr_scaled
            rms_dx_t_1 = T.sqrt(mean_squar_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_square_grad + epsilon)
            delta_x_t = - (rms_dx_t_1 / rms_grad_t) * grad
            # Compute: E[delta_x^2]_t
            new_mean_square_dx = (
                self.decay * mean_squar_dx +
                (1 - self.decay) * T.sqr(delta_x_t))

            # update the params
            new_param = param + delta_x_t
            # Send for the update
            updates.append((mean_square_grad, new_mean_square_grad))
            updates.append((mean_squar_dx, new_mean_square_dx))
            if self.max_colm_norm and param.name in ["W", "w"]:
                new_param_final = norm_constraint(tensor_var=new_param,
                                                  max_norm=self.max_norm)
            else:
                new_param_final = new_param
            updates.append((param, new_param_final))

        return updates


class AdaGrad(LearningRule):
    """Implement the AdaGrad algorithm of [1] to update the parameters of
    the model.

    For more details on how to implement AdGrad, see [2], §2.
    [1]:'Adaptive subgradient methods for online learning and
    stochastic optimization.', Duchi et al.
    [2]:'Notes on AdaDrad', Chris Dyer.
    (link: http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf)

    """
    def __init__(self, max_colm_norm=False, max_norm=15.0):
        self._first_time = True
        self.sum_square_grad = None
        self.max_colm_norm = max_colm_norm
        self.max_norm = max_norm

    def get_updates(self, learning_rate, params, grads, lr_scalers):
        """Compute the AdaDelta updates of the model's parameters.

        param_t := param_(t-1) + AdaDelta_update_t
        """
        if self._first_time:
            self.sum_square_grad = [
                sharedX_mtx(
                    param.get_value() * 0.,
                    name='sum_square_grad_'+param.name,
                    borrow=True) for param in params]
            self._first_time = False

        updates = []
        for (param, grad, sum_square_grad, lr_sc) in zip(
                params, grads, self.sum_square_grad, lr_scalers):
            # Calculate the running average gradient: E[g^2]_t
            new_sum_square_grad = sum_square_grad + T.sqr(grad)

            # The update: delta_x_t
            lr_scaled = learning_rate * lr_sc
            epsilon = lr_scaled
            sqrt_sum_grad_t = T.sqrt(new_sum_square_grad)
            delta_x_t = - (epsilon / sqrt_sum_grad_t) * grad

            # update the params
            new_param = param + delta_x_t
            # Send for the update
            updates.append((sum_square_grad, new_sum_square_grad))
            if self.max_colm_norm and param.name in ["W", "w"]:
                new_param_final = norm_constraint(tensor_var=new_param,
                                                  max_norm=self.max_norm)
            else:
                new_param_final = new_param
            updates.append((param, new_param_final))

        return updates


class RMSProp(LearningRule):
    """Implements the RMSProp learning rule as described in [1].

    The RMSProp rule was described in [1]. The idea is similar to the
    AdaDelta,
    which consists of dividing the learning rate for a weight by a running
    average of the magintudes of recent graidients of that weight.
    Parameters:
        decay: float
            Decay constant similar to the one used in AdaDelta, and Momentum.
        max_scaling: float
            Restrict the RMSProp gradient scaling coefficient to values below
            'max_scaling' to avoid a learning rate too small (almost zero).

    [1]: 'Neural Networks for Machine Learning, Lecture 6a Overview of
        mini-­‐batch gradient descent', a lecture by Hinton et al.
    (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """
    def __init__(self, decay=0.9, max_scaling=1e5, max_colm_norm=False,
                 max_norm=15.0):
        assert 0. <= decay < 1., 'decay must be: 0. <= decay < 1'
        assert max_scaling > 0., 'max_scaling must be > 0.'
        self.decay = sharedX_value(decay, name='decay', borrow=True)
        self.epsilon = 1. / max_scaling
        self.mean_square_grads = None
        self._first_time = True
        self.max_colm_norm = max_colm_norm
        self.max_norm = max_norm

    def get_updates(self, learning_rate, params, grads, lr_scalers):
        """Compute the parameters' updates.

        """
        if self._first_time:
            self.mean_square_grads = [
                sharedX_mtx(
                    param.get_value() * 0.,
                    name='mean_square_grad_'+param.name,
                    borrow=True) for param in params]
            self._first_time = False
        updates = []
        for (param, grad, mean_square_grad, lr_sc) in zip(
                params, grads, self.mean_square_grads, lr_scalers):
            new_mean_square_grad = (
                self.decay * mean_square_grad + (1-self.decay) * T.sqr(grad))
            # the update
            rms_grad_t = T.sqrt(new_mean_square_grad)
            rms_grad_t = T.maximum(rms_grad_t, self.epsilon)
            lr_scaled = learning_rate * lr_sc
            delta_x_t = - lr_scaled * grad / rms_grad_t

            new_param = param + delta_x_t
            # updates
            if self.max_colm_norm and param.name in ["W", "w"]:
                new_param_final = norm_constraint(tensor_var=new_param,
                                                  max_norm=self.max_norm)
            else:
                new_param_final = new_param
            updates.append((param, new_param_final))
            updates.append((mean_square_grad, new_mean_square_grad))

        return updates


class Adam(LearningRule):
    """
    Implement Adaptive Moment Estimation.
    Adam updates implemented as in [1]_.
    Parameters:
        beta1 : float
            Exponential decay rate for the first moment estimates.
        beta2 : float
            Exponential decay rate for the second moment estimates.
        epsilon : float
            Constant for numerical stability.
    Credit:
        https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 max_colm_norm=False, max_norm=15.0):
        self.beta1 = sharedX_value(beta1, name='beta1', borrow=True)
        self.beta2 = sharedX_value(beta2, name='beta2', borrow=True)
        self.epsilon = sharedX_value(epsilon, name='epsilon', borrow=True)
        self.max_colm_norm = max_colm_norm
        self.max_norm = max_norm

    def get_updates(self, learning_rate, params, grads, lr_scalers):
        """Compute the parameters' updates.

        """
        t_prev = theano.shared(floatX(0.))
        updates = OrderedDict()

        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)

        t = t_prev + 1
        a_t = learning_rate*T.sqrt(one-self.beta2**t)/(one-self.beta1**t)

        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

            m_t = self.beta1*m_prev + (one-self.beta1)*g_t
            v_t = self.beta2*v_prev + (one-self.beta2)*g_t**2
            step = a_t*m_t/(T.sqrt(v_t) + self.epsilon)

            updates[m_prev] = m_t
            updates[v_prev] = v_t
            new_param = param - step
            if self.max_colm_norm and param.name in ["W", "w"]:
                new_param_final = norm_constraint(tensor_var=new_param,
                                                  max_norm=self.max_norm)
            else:
                new_param_final = new_param
            updates[param] = new_param_final

        updates[t_prev] = t

        return updates


class Adamax(LearningRule):
    """
    Adamax updates.
     Adamax updates implemented as in [1]_. This is a variant of of the Adam
    algorithm based on the infinity norm.
    Parameters:
        beta1 : float
            Exponential decay rate for the first moment estimates.
        beta2 : float
            Exponential decay rate for the weighted infinity norm estimates.
        epsilon : float
            Constant for numerical stability.
    Credit:
        https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py

    References
    ----------
    .. [1] Kingma, Diederik, and Jimmy Ba (2014):
           Adam: A Method for Stochastic Optimization.
           arXiv preprint arXiv:1412.6980.
    """
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 max_colm_norm=False, max_norm=15.0):
        self.beta1 = sharedX_value(beta1, name='beta1', borrow=True)
        self.beta2 = sharedX_value(beta2, name='beta2', borrow=True)
        self.epsilon = sharedX_value(epsilon, name='epsilon', borrow=True)
        self.max_colm_norm = max_colm_norm
        self.max_norm = max_norm

    def get_updates(self, learning_rate, params, grads, lr_scalers):
        """Compute the parameters' updates.

        """
        t_prev = theano.shared(floatX(0.))
        updates = OrderedDict()

        # Using theano constant to prevent upcasting of float32
        one = T.constant(1)

        t = t_prev + 1
        a_t = learning_rate/(one-self.beta1**t)

        for param, g_t in zip(params, grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

            m_t = self.beta1*m_prev + (one-self.beta1)*g_t
            u_t = T.maximum(self.beta2*u_prev, abs(g_t))
            step = a_t*m_t/(u_t + self.epsilon)

            updates[m_prev] = m_t
            updates[u_prev] = u_t
            new_param = param - step
            if self.max_colm_norm and param.name in ["W", "w"]:
                new_param_final = norm_constraint(tensor_var=new_param,
                                                  max_norm=self.max_norm)
            else:
                new_param_final = new_param
            updates[param] = new_param_final

        updates[t_prev] = t

        return updates
