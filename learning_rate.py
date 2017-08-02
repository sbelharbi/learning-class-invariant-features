import numpy as np
import theano


class AnnealedLearningRate(object):
    """A callback to adjust the learning rate on each freq (batch or epoch).

    The learning rate will be annealed by 1/t at each freq.
    Parameters:
        anneal_start: int
            the epoch when to start annealing.
    """
    def __init__(self, anneal_start, freq='epoch'):
        self._initialized = False
        self._count = 0.
        self._anneal_start = anneal_start
        self.freq = freq

    def __call__(self, learning_rate):
        """Updates the learning rate according to the annealing schedule.

        """
        if not self._initialized:
            self._base = learning_rate.get_value()
            self._initialized = True
        self._count += 1
        learning_rate.set_value(
            np.cast[theano.config.floatX](self.get_current_learning_rate()))

    def get_current_learning_rate(self):
        """Calculate the current learning rate according to the annealing
        schedule.

        """
        return self._base * min(1, self._anneal_start / self._count)


class ExponentialDecayLearningRate(object):
    """
    This anneals the learning rate by dviding it by decay_factor after
    each update (freq='batch').

    lr = lr * decay_factor**(-t)
    Parameters:
        decay_factor: float
            de the decay factor
        min_lr: float
            The lr will be fixed to min_lr when it's reached.
    """
    def __init__(self, decay_factor, min_lr):
        self._count = 0
        self._min_reached = False
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.freq = 'batch'

    def __call__(self, learning_rate):
        """Update the learning rate according to the exponential decay
        schedule.

        """
        if self._count == 0.:
            self._base_lr = learning_rate.get_vale()
        self._count += 1

        if not self._min_reached:
            new_lr = self._base_lr * (self.decay_factor ** (-self._count))
            if new_lr <= self.min_lr:
                self._min_reached = True
                new_lr = self._min_reached
        else:
            new_lr = self.min_lr

        learning_rate.set_value(np.cast[theano.config.floatX](new_lr))
