from layer import HiddenLayer
from layer import LeNetConvPoolLayer_hint

import theano.tensor as T
import theano
import numpy as np
import os
import sys
import datetime as DT
import matplotlib.pylab as plt
import math
import cPickle as pkl
from scipy.misc import imresize
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from random import shuffle
from theano.compile.nanguardmode import NanGuardMode
from non_linearities import NonLinearity, CostType, relu, get_non_linearity_str
from non_linearities import get_non_linearity_fn
from scipy import ndimage

floating = 10
prec2 = "%."+str(floating)+"f"

x_classes = 10  # number of classes by default.


class ModelMLP(object):
    def __init__(self, layers_infos, input, input1, input2,
                 trainx_sh, trainlabels_sh, trainy_sh,
                 validx_sh, validlabels_sh, margin, similair,
                 l1_reg=0., l2_reg=0., tag="",
                 reg_bias=False, dropout=None):
        """
        trainx_sh, trainlabels_sh, trainy_sh,
        validx_sh, validlabels_sh, margin, similair: theano.shared variable.
        """
        self.layers_infos = [
            {"n_in": l["n_in"],
             "n_out": l["n_out"],
             "activation": l["activation"]} for l in layers_infos]

        self.tag = tag
        self.layers = []
        self.margin = margin
        self.similair_bases = similair
        self.similair = T.fvector("sim")
        self.trainx_sh = trainx_sh
        self.trainlabels_sh = trainlabels_sh
        self.trainy_sh = trainy_sh
        self.validx_sh = validx_sh
        self.validlabels_sh = validlabels_sh
        # catch the model's params in the memory WHITHOUT saving
        # them on disc because disc acces is so expensive on somme
        # servers.
        self.catched_params = []
        self.catched_sparser = []
        self.catch_params_bn = []
        self.x = input
        self.x1 = input1
        self.x2 = input2

        self.trg = T.fmatrix("y")
        self.params = []
        self.sparser = []
        self.params_bn = []
        input_lr = input
        input_vl = input
        input_lr1 = input1
        input_lr2 = input2

        for layer in layers_infos:
            intended_to_be_corrupted = False
            if "intended_to_be_corrupted" in layer.keys():
                intended_to_be_corrupted = layer["intended_to_be_corrupted"]
            corrupt_input_l = 0.
            if "corrupt_input_l" in layer.keys():
                corrupt_input_l = layer["corrupt_input_l"]
            self.layers.append(
                HiddenLayer(
                    input=input_lr,
                    input1=input_lr1,
                    input2=input_lr2,
                    input_vl=input_vl,
                    n_in=layer["n_in"],
                    n_out=layer["n_out"],
                    W=layer["W"],
                    b=layer["b"],
                    activation=get_non_linearity_fn(layer["activation"]),
                    rng=layer["rng"],
                    hint=layer["hint"],
                    use_hint=layer["use_hint"],
                    intended_to_be_corrupted=intended_to_be_corrupted,
                    corrupt_input_l=corrupt_input_l,
                    use_sparsity=layer["use_sparsity"],
                    use_sparsity_in_pred=layer["use_sparsity_in_pred"],
                    use_unsupervised=layer["use_unsupervised"],
                    use_batch_normalization=layer["use_batch_normalization"]
                    )
                )
            # Calculate the penalty
            input_lr = self.layers[-1].output
            input_vl = self.layers[-1].output_vl
            input_lr1 = self.layers[-1].output1
            input_lr2 = self.layers[-1].output2
            self.params += self.layers[-1].params
            if self.layers[-1].bn is not None:
                self.params_bn += self.layers[-1].bn.params

        self.sparser = [l.sparser for l in self.layers]
        self.output = self.layers[-1].output
        self.output_vl = self.layers[-1].output_vl
        self.output1 = self.layers[-1].output1
        self.output2 = self.layers[-1].output2
        self.l1 = 0.
        self.l2 = 0.

        nparams = np.sum(
            [p.get_value().flatten().shape[0] for p in self.params])

        print "model contains %i parameters" % nparams
        # dump params for debug
#        todump = []
#        for p in self.params:
#            todump.append(p.get_value())
#        with open("paramsmlp.pkl", "w") as f:
#            pkl.dump(todump, f, protocol=pkl.HIGHEST_PROTOCOL)

#        with open("init_keras_param.pkl", 'r') as f:
#            keras_params = pkl.load(f)
#            print "initialized using keras params."
#            for param, param_vl in zip(self.params, keras_params):
#                param.set_value(param_vl)
        for l in self.layers:
            if l1_reg != 0.:
                self.l1 += l1_reg * abs(l.W).sum()
                l.ae.set_regularization_l1(l1_reg)
                if reg_bias:
                    self.l1 += l1_reg * abs(l.b).sum()
            if l2_reg != 0.:
                self.l2 += l2_reg * (l.W**2).sum()
                l.ae.set_regularization_l2(l2_reg)
                if reg_bias:
                    self.l2 += l2_reg * (l.b**2).sum()

    def catch_params(self):
        self.catched_params = [param.get_value() for param in self.params]
        self.catched_sparser = [
            sparser.get_value() for sparser in self.sparser]
        if self.params_bn != []:
            self.catch_params_bn = [p.get_value() for p in self.params_bn]

    def set_model_to_catched_params(self):
        if self.catched_params != []:
            for p, pv in zip(self.params, self.catched_params):
                p.set_value(pv)
        if self.catched_sparser != []:
            for p, pv in zip(self.sparser, self.catched_sparser):
                p.set_value(pv)
        if self.catch_params_bn != []:
            for p, pv in zip(self.params_bn, self.catch_params_bn):
                p.set_value(pv)

    def save_params(self, weights_file, catched=False):
        """Save the model's params."""
        params_bn = []
        with open(weights_file, "w") as f:
            if catched:
                if self.catched_params != []:
                    params_vl = self.catched_params
                    sparser_vl = self.catched_sparser
                    params_bn = self.catch_params_bn
                else:
                    raise ValueError(
                        "You asked to save catched params," +
                        "but you didn't catch any!!!!!!!")
            else:
                params_vl = [param.get_value() for param in self.params]
                sparser_vl = [sparser.get_value() for sparser in self.sparser]
                if self.params_bn != []:
                    params_bn = [pbn.get_value() for pbn in self.params_bn]

            stuff = {"layers_infos": self.layers_infos,
                     "params_vl": params_vl,
                     "sparser_vl": sparser_vl,
                     "params_bn": params_bn,
                     "tag": self.tag}
            pkl.dump(stuff, f, protocol=pkl.HIGHEST_PROTOCOL)

    def set_params_vals(self, weights_file):
        """Set the model's params."""
        with open(weights_file, "r") as f:
            stuff = pkl.load(f)
            layers_infos, params_vl = stuff["layers_infos"], stuff["params_vl"]
            sparser_vl = stuff["sparser_vl"]
            params_bn = stuff["params_bn"]
            # Stuff to check
            keys = ["n_in", "n_out", "activation"]
            assert all(
                [l1[k] == l2[k]
                    for k in keys
                    for (l1, l2) in zip(layers_infos, self.layers_infos)])
            for param, param_vl in zip(self.params, params_vl):
                param.set_value(param_vl)
            for sparser, spar_vl in zip(self.sparser, sparser_vl):
                sparser.set_value(spar_vl)
            if self.params_bn != []:
                for p, pv in zip(self.params_bn, params_bn):
                    p.set_value(pv)


class LeNet(ModelMLP):
    def __init__(self, layers_infos, input, input1, input2,
                 trainx_sh, trainlabels_sh, trainy_sh,
                 validx_sh, validlabels_sh, margin, similair,
                 l1_reg=0., l2_reg=0., tag="",
                 reg_bias=False, dropout=None, batch_size=None):
        """
        trainx_sh, trainlabels_sh, trainy_sh,
        validx_sh, validlabels_sh, margin, similair: theano.shared variable.
        """
        nkerns = [20, 50]
        self.layers_infos = [
            {"n_in": l["n_in"],
             "n_out": l["n_out"],
             "activation": l["activation"]} for l in layers_infos]

        self.tag = tag
        self.layers = []
        self.margin = margin
        self.similair_bases = similair
        self.similair = T.fvector("sim")
        self.trainx_sh = trainx_sh
        self.trainlabels_sh = trainlabels_sh
        self.trainy_sh = trainy_sh
        self.validx_sh = validx_sh
        self.validlabels_sh = validlabels_sh
        # catch the model's params in the memory WHITHOUT saving
        # them on disc because disc acces is so expensive on somme
        # servers.
        self.catched_params = []
        self.catched_sparser = []
        self.catch_params_bn = []
        self.x = input
        self.x1 = input1
        self.x2 = input2

        self.trg = T.fmatrix("y")
        self.params = []
        self.sparser = []
        self.params_bn = []
        input_lr = input
        input_vl = input
        input_lr1 = input1
        input_lr2 = input2
        # First conv layer
        layer = layers_infos[0]
        intended_to_be_corrupted = False
        if "intended_to_be_corrupted" in layer.keys():
            intended_to_be_corrupted = layer["intended_to_be_corrupted"]
        corrupt_input_l = 0.
        if "corrupt_input_l" in layer.keys():
            corrupt_input_l = layer["corrupt_input_l"]
        self.layers.append(
            LeNetConvPoolLayer_hint(
                rng=layer["rng"],
                input=input,
                input1=input_lr1,
                input2=input_lr2,
                input_vl=input_vl,
                filter_shape=(nkerns[0], 1, 5, 5),
                image_shape=(batch_size, 1, 28, 28),
                poolsize=(2, 2),
                activation=get_non_linearity_fn(layer["activation"]),
                hint=layer["hint"],
                use_hint=layer["use_hint"],
                intended_to_be_corrupted=intended_to_be_corrupted,
                corrupt_input_l=corrupt_input_l,
                use_sparsity=layer["use_sparsity"],
                use_sparsity_in_pred=layer["use_sparsity_in_pred"],
                use_unsupervised=layer["use_unsupervised"],
                use_batch_normalization=layer["use_batch_normalization"])
        )
        input_lr = self.layers[-1].output
        input_vl = self.layers[-1].output_vl
        input_lr1 = self.layers[-1].output1_non_fl
        input_lr2 = self.layers[-1].output2_non_fl
        self.params += self.layers[-1].params
        if self.layers[-1].bn is not None:
                self.params_bn += self.layers[-1].bn.params

        # Second conv layer
        layer = layers_infos[1]
        intended_to_be_corrupted = False
        if "intended_to_be_corrupted" in layer.keys():
            intended_to_be_corrupted = layer["intended_to_be_corrupted"]
        corrupt_input_l = 0.
        if "corrupt_input_l" in layer.keys():
            corrupt_input_l = layer["corrupt_input_l"]
        self.layers.append(
           LeNetConvPoolLayer_hint(
                rng=layer["rng"],
                input=input_lr,
                input1=input_lr1,
                input2=input_lr2,
                input_vl=input_vl,
                filter_shape=(nkerns[1], nkerns[0], 5, 5),
                image_shape=(batch_size, nkerns[0], 12, 12),
                poolsize=(2, 2),
                activation=get_non_linearity_fn(layer["activation"]),
                hint=layer["hint"],
                use_hint=layer["use_hint"],
                intended_to_be_corrupted=intended_to_be_corrupted,
                corrupt_input_l=corrupt_input_l,
                use_sparsity=layer["use_sparsity"],
                use_sparsity_in_pred=layer["use_sparsity_in_pred"],
                use_unsupervised=layer["use_unsupervised"],
                use_batch_normalization=layer["use_batch_normalization"])
        )
        input_lr = self.layers[-1].output.flatten(2)
        input_vl = self.layers[-1].output_vl.flatten(2)
        input_lr1 = self.layers[-1].output1
        input_lr2 = self.layers[-1].output2
        self.params += self.layers[-1].params
        if self.layers[-1].bn is not None:
                self.params_bn += self.layers[-1].bn.params

        # Fully connexted layer
        layer = layers_infos[2]
        intended_to_be_corrupted = False
        if "intended_to_be_corrupted" in layer.keys():
            intended_to_be_corrupted = layer["intended_to_be_corrupted"]
        corrupt_input_l = 0.
        if "corrupt_input_l" in layer.keys():
            corrupt_input_l = layer["corrupt_input_l"]
        self.layers.append(
                HiddenLayer(
                    input=input_lr,
                    input1=input_lr1,
                    input2=input_lr2,
                    input_vl=input_vl,
                    n_in=nkerns[1]*4*4,
                    n_out=500,
                    W=layer["W"],
                    b=layer["b"],
                    activation=get_non_linearity_fn(layer["activation"]),
                    rng=layer["rng"],
                    hint=layer["hint"],
                    use_hint=layer["use_hint"],
                    intended_to_be_corrupted=intended_to_be_corrupted,
                    corrupt_input_l=corrupt_input_l,
                    use_sparsity=layer["use_sparsity"],
                    use_sparsity_in_pred=layer["use_sparsity_in_pred"],
                    use_unsupervised=layer["use_unsupervised"],
                    use_batch_normalization=layer["use_batch_normalization"]
                    )
                )
        input_lr = self.layers[-1].output
        input_vl = self.layers[-1].output_vl
        input_lr1 = self.layers[-1].output1
        input_lr2 = self.layers[-1].output2
        self.params += self.layers[-1].params
        if self.layers[-1].bn is not None:
                self.params_bn += self.layers[-1].bn.params

        # Output layer
        layer = layers_infos[3]
        intended_to_be_corrupted = False
        if "intended_to_be_corrupted" in layer.keys():
            intended_to_be_corrupted = layer["intended_to_be_corrupted"]
        corrupt_input_l = 0.
        if "corrupt_input_l" in layer.keys():
            corrupt_input_l = layer["corrupt_input_l"]
        self.layers.append(
            HiddenLayer(
                    input=input_lr,
                    input1=input_lr1,
                    input2=input_lr2,
                    input_vl=input_vl,
                    n_in=500,
                    n_out=layer["n_out"],
                    W=layer["W"],
                    b=layer["b"],
                    activation=get_non_linearity_fn(layer["activation"]),
                    rng=layer["rng"],
                    hint=layer["hint"],
                    use_hint=layer["use_hint"],
                    intended_to_be_corrupted=intended_to_be_corrupted,
                    corrupt_input_l=corrupt_input_l,
                    use_sparsity=layer["use_sparsity"],
                    use_sparsity_in_pred=layer["use_sparsity_in_pred"],
                    use_unsupervised=layer["use_unsupervised"],
                    use_batch_normalization=layer["use_batch_normalization"]
                    )
        )
        input_lr = self.layers[-1].output
        input_vl = self.layers[-1].output_vl
        input_lr1 = self.layers[-1].output1
        input_lr2 = self.layers[-1].output2
        self.params += self.layers[-1].params
        if self.layers[-1].bn is not None:
                self.params_bn += self.layers[-1].bn.params

        self.sparser = [l.sparser for l in self.layers]
        self.output = self.layers[-1].output
        self.output_vl = self.layers[-1].output_vl
        self.output1 = self.layers[-1].output1
        self.output2 = self.layers[-1].output2
        self.l1 = 0.
        self.l2 = 0.

        nparams = np.sum(
            [p.get_value().flatten().shape[0] for p in self.params])

        print "model contains %i parameters" % nparams
        # dump params for debug
#        todump = []
#        for p in self.params:
#            todump.append(p.get_value())
#        with open("paramsmlp.pkl", "w") as f:
#            pkl.dump(todump, f, protocol=pkl.HIGHEST_PROTOCOL)

#        with open("init_keras_param.pkl", 'r') as f:
#            keras_params = pkl.load(f)
#            print "initialized using keras params."
#            for param, param_vl in zip(self.params, keras_params):
#                param.set_value(param_vl)
        for l in self.layers:
            if l1_reg != 0.:
                self.l1 += abs(l.W).sum()
                if reg_bias:
                    self.l1 += abs(l.b).sum()
            if l2_reg != 0.:
                self.l2 += (l.W**2).sum()
                if reg_bias:
                    self.l2 += (l.b**2).sum()


class CNNCIFAR(ModelMLP):
    def __init__(self, layers_infos, input, input1, input2,
                 trainx_sh, trainlabels_sh, trainy_sh,
                 validx_sh, validlabels_sh, margin, similair,
                 l1_reg=0., l2_reg=0., tag="",
                 reg_bias=False, dropout=None, batch_size=None):
        """
        trainx_sh, trainlabels_sh, trainy_sh,
        validx_sh, validlabels_sh, margin, similair: theano.shared variable.
        """
        nkerns = [96, 96, 192, 192, 192, 192, 10]
        filters = [(3, 3), (1, 1), (3, 3), (1, 1), (3, 3), (1, 1), (1, 1)]
        poolsizes = [(1, 1), (2, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        s = [1, 1]  # stride
        im_sz = [32, 32]
        dim_full = [1000]
        self.layers_infos = [
            {"n_in": l["n_in"],
             "n_out": l["n_out"],
             "activation": l["activation"]} for l in layers_infos]

        self.tag = tag
        self.layers = []
        self.margin = margin
        self.similair_bases = similair
        self.similair = T.fvector("sim")
        self.trainx_sh = trainx_sh
        self.trainlabels_sh = trainlabels_sh
        self.trainy_sh = trainy_sh
        self.validx_sh = validx_sh
        self.validlabels_sh = validlabels_sh
        # catch the model's params in the memory WHITHOUT saving
        # them on disc because disc acces is so expensive on somme
        # servers.
        self.catched_params = []
        self.catched_sparser = []
        self.catch_params_bn = []
        self.x = input
        self.x1 = input1
        self.x2 = input2

        self.trg = T.fmatrix("y")
        self.params = []
        self.sparser = []
        self.params_bn = []
        input_lr = input
        input_vl = input
        input_lr1 = input1
        input_lr2 = input2
        map_size_h, map_size_w = im_sz[0], im_sz[1]
        n_inf = 3
        for i in range(len(nkerns)):
            # First conv layer
            layer = layers_infos[i]
            intended_to_be_corrupted = False
            if "intended_to_be_corrupted" in layer.keys():
                intended_to_be_corrupted = layer["intended_to_be_corrupted"]
            corrupt_input_l = 0.
            if "corrupt_input_l" in layer.keys():
                corrupt_input_l = layer["corrupt_input_l"]
            print nkerns[i], n_inf, filters[i][0], filters[i][1]
            self.layers.append(
                LeNetConvPoolLayer_hint(
                    rng=layer["rng"],
                    input=input_lr,
                    input1=input_lr1,
                    input2=input_lr2,
                    input_vl=input_vl,
                    filter_shape=(nkerns[i], n_inf, filters[i][0],
                                  filters[i][1]),
                    image_shape=(batch_size, n_inf, map_size_h, map_size_w),
                    poolsize=poolsizes[i],
                    activation=get_non_linearity_fn(layer["activation"]),
                    hint=layer["hint"],
                    use_hint=layer["use_hint"],
                    intended_to_be_corrupted=intended_to_be_corrupted,
                    corrupt_input_l=corrupt_input_l,
                    use_sparsity=layer["use_sparsity"],
                    use_sparsity_in_pred=layer["use_sparsity_in_pred"],
                    use_unsupervised=layer["use_unsupervised"],
                    use_batch_normalization=layer["use_batch_normalization"])
            )
            input_lr = self.layers[-1].output
            input_vl = self.layers[-1].output_vl
            input_lr1 = self.layers[-1].output1_non_fl
            input_lr2 = self.layers[-1].output2_non_fl
            self.params += self.layers[-1].params
            if self.layers[-1].bn is not None:
                    self.params_bn += self.layers[-1].bn.params
            map_size_h = (map_size_h - filters[i][0] + 1)/s[0] / poolsizes[i][0]
            map_size_w = (map_size_w - filters[i][1] + 1)/s[1] / poolsizes[i][1]
            print map_size_h, "x", map_size_w
            n_inf = nkerns[i]

        # Take the flatten version
        input_lr = self.layers[-1].output.flatten(2)
        input_vl = self.layers[-1].output_vl.flatten(2)
        input_lr1 = self.layers[-1].output1
        input_lr2 = self.layers[-1].output2
        # Fully connexted layer
        layer = layers_infos[i+1]
        intended_to_be_corrupted = False
        if "intended_to_be_corrupted" in layer.keys():
            intended_to_be_corrupted = layer["intended_to_be_corrupted"]
        corrupt_input_l = 0.
        if "corrupt_input_l" in layer.keys():
            corrupt_input_l = layer["corrupt_input_l"]
        self.layers.append(
                HiddenLayer(
                    input=input_lr,
                    input1=input_lr1,
                    input2=input_lr2,
                    input_vl=input_vl,
                    n_in=nkerns[-1]*map_size_h*map_size_w,
                    n_out=dim_full[0],
                    W=layer["W"],
                    b=layer["b"],
                    activation=get_non_linearity_fn(layer["activation"]),
                    rng=layer["rng"],
                    hint=layer["hint"],
                    use_hint=layer["use_hint"],
                    intended_to_be_corrupted=intended_to_be_corrupted,
                    corrupt_input_l=corrupt_input_l,
                    use_sparsity=layer["use_sparsity"],
                    use_sparsity_in_pred=layer["use_sparsity_in_pred"],
                    use_unsupervised=layer["use_unsupervised"],
                    use_batch_normalization=layer["use_batch_normalization"]
                    )
                )
        input_lr = self.layers[-1].output
        input_vl = self.layers[-1].output_vl
        input_lr1 = self.layers[-1].output1
        input_lr2 = self.layers[-1].output2
        self.params += self.layers[-1].params
        if self.layers[-1].bn is not None:
                self.params_bn += self.layers[-1].bn.params

        # Output layer
        layer = layers_infos[i+2]
        intended_to_be_corrupted = False
        if "intended_to_be_corrupted" in layer.keys():
            intended_to_be_corrupted = layer["intended_to_be_corrupted"]
        corrupt_input_l = 0.
        if "corrupt_input_l" in layer.keys():
            corrupt_input_l = layer["corrupt_input_l"]
        self.layers.append(
            HiddenLayer(
                    input=input_lr,
                    input1=input_lr1,
                    input2=input_lr2,
                    input_vl=input_vl,
                    n_in=dim_full[0],
                    n_out=layer["n_out"],
                    W=layer["W"],
                    b=layer["b"],
                    activation=get_non_linearity_fn(layer["activation"]),
                    rng=layer["rng"],
                    hint=layer["hint"],
                    use_hint=layer["use_hint"],
                    intended_to_be_corrupted=intended_to_be_corrupted,
                    corrupt_input_l=corrupt_input_l,
                    use_sparsity=layer["use_sparsity"],
                    use_sparsity_in_pred=layer["use_sparsity_in_pred"],
                    use_unsupervised=layer["use_unsupervised"],
                    use_batch_normalization=layer["use_batch_normalization"]
                    )
        )
        input_lr = self.layers[-1].output
        input_vl = self.layers[-1].output_vl
        input_lr1 = self.layers[-1].output1
        input_lr2 = self.layers[-1].output2
        self.params += self.layers[-1].params
        if self.layers[-1].bn is not None:
                self.params_bn += self.layers[-1].bn.params

        self.sparser = [l.sparser for l in self.layers]
        self.output = self.layers[-1].output
        self.output_vl = self.layers[-1].output_vl
        self.output1 = self.layers[-1].output1
        self.output2 = self.layers[-1].output2
        self.l1 = 0.
        self.l2 = 0.

        nparams = np.sum(
            [p.get_value().flatten().shape[0] for p in self.params])

        print "model contains %i parameters" % nparams
        # dump params for debug
#        todump = []
#        for p in self.params:
#            todump.append(p.get_value())
#        with open("paramsmlp.pkl", "w") as f:
#            pkl.dump(todump, f, protocol=pkl.HIGHEST_PROTOCOL)

#        with open("init_keras_param.pkl", 'r') as f:
#            keras_params = pkl.load(f)
#            print "initialized using keras params."
#            for param, param_vl in zip(self.params, keras_params):
#                param.set_value(param_vl)
        for l in self.layers:
            if l1_reg != 0.:
                self.l1 += abs(l.W).sum()
                if reg_bias:
                    self.l1 += abs(l.b).sum()
            if l2_reg != 0.:
                self.l2 += (l.W**2).sum()
                if reg_bias:
                    self.l2 += (l.b**2).sum()


class StaticExponentialDecayWeightRate(object):
    """
    A callback to adjust the cost weight rate.
    The weight is used to weight the PRE-TRAINING sub-costs in the embedded
    pre-training.

    The weights are updated by: weight_sup = 1 - v, weight_in = v/2
    weight_out = v/2.
    v > 0, v = exp(-epochs_seen/slope)
    Parameters:
        anneal_start: int (default 0). The epoch when to start annealing.
        slope: float. The slope of the exp.
    """
    def __init__(self, slop, anneal_start=0):
        self._initialized = False
        self._anneal_start = anneal_start
        self.l_sup, self.l_in, self.l_out = None, None, None
        self.slop = float(slop)
        self.epochs_seen = 0

    def __call__(self, l_sup, l_in, l_out, epochs_seen, to_update):
        """Updates the weight rate according to the exp schedule.
        Input:
            l_sup: Theano shared variable.
            l_in: Theano shared variable.
            l_out: Theano shared variable.
            epochs_seen: Int. The number of seen epcohs.
            to_update: dict. indicate if to update l_in, l_out.
                {"l_in": True, "l_out": True}.
        """
        s_w = l_sup.get_value() + l_in.get_value() + l_out.get_value()
        assert s_w - 1 < 1e-3
        self.epochs_seen = epochs_seen

        if not self._initialized:
            self.l_sup, self.l_in, self.l_out = l_sup, l_in, l_out
            self._initialized = True

        if (self.epochs_seen >= self._anneal_start):
            self.l_sup.set_value(np.cast[theano.config.floatX](
                self.get_current_weight_rate()))
            val = 1. - self.l_sup.get_value()
            if to_update["l_in"] and to_update["l_out"]:
                self.l_in.set_value(np.cast[theano.config.floatX](val/2.))
                self.l_out.set_value(np.cast[theano.config.floatX](val/2.))
            elif to_update["l_in"] and not to_update["l_out"]:
                self.l_in.set_value(np.cast[theano.config.floatX](val))
            elif not to_update["l_in"] and to_update["l_out"]:
                self.l_out.set_value(np.cast[theano.config.floatX](val))

        s_w = l_sup.get_value() + l_in.get_value() + l_out.get_value()
        assert s_w - 1 < 1e-3

    def get_current_weight_rate(self):
        """Calculate the current weight cost rate according to the a
        schedule.
        """
        return np.max([0., 1. - np.exp(float(-self.epochs_seen)/self.slop)])


class StaticExponentialDecayWeightRateSingle(object):
    """
    A callback to adjust the cost weight rate.
    The weight is used to weight the PRE-TRAINING sub-costs in the embedded
    pre-training.

    The weights are updated by: weight_sup = 1 - v, weight_in = v/2
    weight_out = v/2.
    v > 0, v = exp(-epochs_seen/slope)
    Parameters:
        anneal_start: int (default 0). The epoch when to start annealing.
        slope: float. The slope of the exp.
    """
    def __init__(self, slop, anneal_start=0):
        self._initialized = False
        self._anneal_start = anneal_start
        self.w = None
        self.slop = float(slop)
        self.epochs_seen = 0

    def __call__(self, w, epochs_seen):
        """Updates the weight rate according to the exp schedule.
        Input:
            w: Theano shared variable.
            epochs_seen: Int. The number of seen epcohs.
        """
        self.epochs_seen = epochs_seen

        if not self._initialized:
            self.w = w
            self._initialized = True

        if (self.epochs_seen >= self._anneal_start):
            self.w.set_value(np.cast[theano.config.floatX](
                self.get_current_weight_rate()))

    def get_current_weight_rate(self):
        """Calculate the current weight cost rate according to the a
        schedule.
        """
        return np.max([0., 1. - np.exp(float(-self.epochs_seen)/self.slop)])


class StaticAnnealedWeightRate(object):
    """
    A callback to adjust the cost weight rate.
    The weight is used to weight the PRE-TRAINING sub-costs in the embedded
    pre-training.

    The weight is annealed by: weight_sup = weight_sup + v,
    weight_in = 1 - v/2, weight_sup = 1 - v/2.
    v > 0.
    The annealing process starts from the epoch T0 = 0 (by defautl) and
    finishes at some epoch T.
    At the epoch T, the weight must be 0. Hince, v = weight/(T-T0).
    Parameters:
        anneal_end: int. The epoch when to end the annealing
        anneal_start: int (default 0). The epoch when to start annealing.
    """
    def __init__(self, anneal_end, anneal_start=0):
        self._initialized = False
        self._anneal_start = anneal_start
        self._anneal_end = anneal_end
        self.l_sup, self.l_in, self.l_out = None, None, None
        self.v = 0.

    def __call__(self, l_sup, l_in, l_out, epochs_seen, to_update):
        """Updates the weight rate according to the annealing schedule.
        l_sup: Theano shared variable.
            l_in: Theano shared variable.
            l_out: Theano shared variable.
            epochs_seen: Int. The number of seen epcohs.
            to_update: dict. indicate if to update l_in, l_out.
                {"l_in": True, "l_out": True}.
        """
        s_w = l_sup.get_value() + l_in.get_value() + l_out.get_value()
        assert -1 < 1e-3

        if not self._initialized:
            self.l_sup, self.l_in, self.l_out = l_sup, l_in, l_out

            distance = float(self._anneal_end - self._anneal_start)
            if distance == 0:
                self.v = 0
            else:
                self.v = float(1. - self.l_sup.get_value()) / distance

            self._initialized = True

        if (epochs_seen >= self._anneal_start) and \
                (epochs_seen <= self._anneal_end):
            self.l_sup.set_value(np.cast[theano.config.floatX](
                self.get_current_weight_rate()))
            if epochs_seen == self._anneal_end:
                self.l_sup.set_value(np.cast[theano.config.floatX](1.))

            val = 1. - self.l_sup.get_value()
            if to_update["l_in"] and to_update["l_out"]:
                self.l_in.set_value(np.cast[theano.config.floatX](val/2.))
                self.l_out.set_value(np.cast[theano.config.floatX](val/2.))
            elif to_update["l_in"] and not to_update["l_out"]:
                self.l_in.set_value(np.cast[theano.config.floatX](val))
            elif not to_update["l_in"] and to_update["l_out"]:
                self.l_out.set_value(np.cast[theano.config.floatX](val))

#        print l_sup.get_value(), l_in.get_value(), l_out.get_value()

        s_w = l_sup.get_value() + l_in.get_value() + l_out.get_value()
        assert s_w - 1 < 1e-3

    def get_current_weight_rate(self):
        """Calculate the current learning rate according to the annealing
        schedule.
        """
        return np.max([0., self.l_sup.get_value() + self.v])


class StaticAnnealedWeightRateSingle(object):
    """
    A callback to adjust the cost weight rate.
    The weight is used to weight the PRE-TRAINING sub-costs in the embedded
    pre-training.

    The weight is annealed by: weight_sup = weight_sup + v,
    weight_in = 1 - v/2, weight_sup = 1 - v/2.
    v > 0.
    The annealing process starts from the epoch T0 = 0 (by defautl) and
    finishes at some epoch T.
    At the epoch T, the weight must be 0. Hince, v = weight/(T-T0).
    Parameters:
        anneal_end: int. The epoch when to end the annealing
        anneal_start: int (default 0). The epoch when to start annealing.
    """
    def __init__(self, anneal_end, down, init_vl, end_vl, anneal_start=0):
        self._initialized = False
        self._anneal_start = anneal_start
        self._anneal_end = anneal_end
        self.w = None
        self.v = 0.
        self.epochs_seen = 0
        self.down = down
        self.init_vl = init_vl
        self.end_vl = end_vl

    def __call__(self, w, epochs_seen):
        """Updates the weight rate according to the annealing schedule.
        l_sup: Theano shared variable.
            l_in: Theano shared variable.
            l_out: Theano shared variable.
            epochs_seen: Int. The number of seen epcohs.
            to_update: dict. indicate if to update l_in, l_out.
                {"l_in": True, "l_out": True}.
        """
        self.epochs_seen = epochs_seen
        if not self._initialized:
            self.w = w

            distance = float(self._anneal_end - self._anneal_start)
            if distance == 0:
                self.v = 0
            else:
                if self.down:
                    self.v = float(self.end_vl - self.init_vl) / distance
                else:
                    self.v = float(self.init_vl - self.end_vl) / distance

            self._initialized = True

        if self.epochs_seen == self._anneal_start:
            self.w.set_value(np.cast[theano.config.floatX](self.init_vl))
        if (self.epochs_seen >= self._anneal_start) and \
                (self.epochs_seen <= self._anneal_end):
            self.w.set_value(np.cast[theano.config.floatX](
                self.get_current_weight_rate()))
            if epochs_seen == self._anneal_end:
                if self.down:
                    self.w.set_value(np.cast[theano.config.floatX](0.))
                else:
                    self.w.set_value(np.cast[theano.config.floatX](1.))

    def get_current_weight_rate(self):
        """Calculate the current learning rate according to the annealing
        schedule.
        """
        return np.max([0., self.w.get_value() + self.v])


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    all_l = []
    for i in range(0, len(l), n):
        all_l.append(l[i:i + n])

    return all_l


def to_categorical(y, nbr_classes):
    """Convert an array of integer labels into a matrix
    (len(y), nbr_classes)."""
    y = np.array(y, dtype='int')
    Y = np.zeros((len(y), nbr_classes), dtype=theano.config.floatX)
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def plot_classes(y, cord, names, test_error, message=""):
    plt.close("all")
    cord = np.array(cord)
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    un = np.unique(y)
    fig, ax = plt.subplots()
    for u, col in zip(un, colors):
        ind = np.argwhere(y == u)
        x = cord[ind, :]
        x = x.reshape(x.shape[0], cord.shape[1])
        ax.scatter(x[:, 0], x[:, 1], label="class:" + str(u),
                   color=col)

    plt.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': 8})
    fig.suptitle(
        "Output prediction. Test error:" + str(test_error*100) + "%. " +
        message, fontsize=8)
    return fig


def duplicate_one_sample(x, y, nbr):
    """Duplicate one samples many times until  the number of row of x is
    nbr. Returns the weights of each sample (each original sample will have
    the weight 1 except the duplicated samples and the one used for duplication
    will be: 1/(nbr-x.shape[0])."""
    new_x = np.empty((nbr, x.shape[1]), dtype=x.dtype)
    new_y = np.empty((nbr,), dtype=y.dtype)
    weights = np.ones((nbr, ), dtype=theano.config.floatX)
    new_x[:x.shape[0]] = x
    new_x[x.shape[0]:] = x[-1]
    new_y[:y.shape[0]] = y
    new_y[y.shape[0]:] = y[-1]
    weights[y.shape[0]-1:] = 1./(nbr - x.shape[0] + 1)
    print np.sum(weights), x.shape[0]
    # assert int(np.sum(weights)) - x.shape[0] == 0

    return new_x, new_y, weights


def floatX(arr):
    """Converts data to a numpy array of dtype ``theano.config.floatX``.

    Parameters
    ----------
    arr : array_like
        The data to be converted.

    Returns
    -------
    numpy ndarray
        The input array in the ``floatX`` dtype configured for Theano.
        If `arr` is an ndarray of correct dtype, it is returned as is.
    """
    return np.asarray(arr, dtype=theano.config.floatX)


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


def sharedX_mtx(mtx, name=None, borrow=None, dtype=None):
    """Share a matrix value with type theano.confgig.floatX.
    Parameters:
        value: matrix array
        name: variable name (str)
        borrow: boolean
        dtype: the type of the value when shared. default: theano.config.floatX
    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(
        np.array(mtx, dtype=dtype), name=name, borrow=borrow)


def contains_nan(array):
    """Check whether a 'numpy.ndarray' contains any 'numpy.nan' values.

    array: a numpy.ndarray array

    Returns:
    contains_nan: boolean
        `True` if there is at least one 'numpy.nan', `False` otherwise.
    """
    return np.isnan(np.min(array))


def contains_inf(array):
    """ Check whether a 'numpy.ndarray' contains any 'numpy.inf' values.

    array: a numpy.ndarray array

    Returns:
    contains_inf: boolean
        `True` if there is a least one 'numpy.inf', `False` otherwise.
    """
    return np.isinf(np.nanmin(array)) or np.isinf(np.nanmax(array))


def isfinite(array):
    """Check if 'numpy.ndarray' contains any 'numpy.inf' or 'numpy.nan' values.

    array: a numpy.ndarray array

    Returns:
    isfinite: boolean
        `True` if there is no 'numpy.inf' and 'numpy.nan', `False` otherwise.
    """
    return np.isfinite(np.min(array)) and np.isfinite(np.max(array))


def get_net_pure_cost(model, cost_type, eye=True):
    """Get the train cost of the network."""
    cost = None
    if cost_type == CostType.MeanSquared:
        cost = T.mean(
            T.sqr(model.output_dropout - model.trg), axis=1)
    elif cost_type == CostType.CrossEntropy:
        cost = T.mean(
            T.nnet.binary_crossentropy(
                model.output_dropout, model.trg), axis=1)
    elif cost_type == CostType.NegativeLogLikelihood:
        # This causes INF in cost (UNSTABLE) when used with the hint.
        cost = -T.log(T.sum(model.output * model.trg, axis=1))
        # cost = T.nnet.categorical_crossentropy(model.output, model.trg)
    else:
        raise ValueError("cost type unknow.")
    return cost


def normalize_grad(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (T.sqrt(T.sum(T.square(x))))


def theano_fns(model,
               learning_rate,
               h_w, s_w, lambda_sparsity,
               updater=None,
               tag="noHint",
               max_colm_norm=False, max_norm=15.0, norm_gsup=False,
               norm_gh=False):
    """
    Params:
        norm_gsup: Boolean. If True, the supervised gradient will be normalized
        (g/l2(g)) before it is fed to the get_updates().
        norm_gh: Boolean. If True, the hint gradient will be normalized
        (g/l2(g)) before it is fed to the get updates() or the accumulator.
        see:
        https://goo.gl/dbQqeN
        https://goo.gl/89ywtn
        https://goo.gl/1U21d0
    """
    cost_type = CostType.NegativeLogLikelihood
    # Get pure cost net
    cost_net_pure = get_net_pure_cost(model, cost_type)
    hint = True
    sparsity = None
    if tag == "noHint":
        print "Train: NO hint"
        hint = False
    elif tag == "hint":
        print "Train: hint"
        penalty = None
        sparsity = None
        for l in model.layers:
            if l.use_hint:
                if penalty is None:
                    penalty = l.get_penalty(model.similair, model.margin)
                else:
                    penalty += l.get_penalty(model.similair, model.margin)
                # Sparsity
                if l.use_sparsity:
                    assert l.use_sparsity is None
                    if sparsity is None:
                        sparsity = abs(l.sparser).sum()
                    else:
                        sparsity += abs(l.sparser).sum()

    cost_hint = None
    cost_sup = s_w * T.mean(cost_net_pure)
    if hint:
        if sparsity is not None:
            cost_hint = h_w * T.mean(penalty) + lambda_sparsity * sparsity
        else:
            cost_hint = h_w * T.mean(penalty)
        if model.l1 != 0.:
            cost_hint += model.l1
        if model.l2 != 0.:
            cost_hint += model.l2

#    if model.l1 != 0.:
#        cost_sup += model.l1
#    if model.l2 != 0.:
#        cost_sup += model.l2

    assert sparsity is None
    # Gradients:
    params = model.params  # + [l.sparser for l in model.layers]

    grad_sup = T.grad(cost_sup, params, disconnected_inputs='warn')
    # Normalize the gradient: g <--- g / l2(g)
    if norm_gsup:
        for ig in range(len(grad_sup)):
            grad_sup[ig] = normalize_grad(grad_sup[ig])
    # "warn" instead of "raise")
    grad_hint = None
    if hint:
        grad_hint = T.grad(cost_hint, params,
                           disconnected_inputs='warn')  # instead of "raise")
        if norm_gh:
            for ig in range(len(grad_hint)):
                grad_hint[ig] = normalize_grad(grad_hint[ig])
    # http://deeplearning.net/software/theano/library/gradient.html#theano.gradient.grad
    # make shared zones to collect the gradient of each term.
    grad_sup_sh = [sharedX_mtx(
        param.get_value() * 0., name="grad_sup"+param.name, borrow=True)
        for param in params]

    grad_hint_sh = [sharedX_mtx(
        param.get_value() * 0., name="grad_hint"+param.name, borrow=True)
        for param in params]

    if hint:
        all_grads = [gs+gh for (gs, gh) in zip(grad_sup_sh, grad_hint_sh)]
    else:
        all_grads = grad_sup_sh
#    # Normalize the sum of the gradient
#    if norm_gh:
#        for ig in range(len(all_grads)):
#            all_grads[ig] = normalize_grad(all_grads[ig])
    # updates for shared grad
    updates_sh_gr_sup = [(gsh, gr) for (gsh, gr) in zip(grad_sup_sh, grad_sup)]
    updates_sh_gr_hint = None
    if hint:
        updates_sh_gr_hint = [
            (gsh, gr) for (gsh, gr) in zip(grad_hint_sh, grad_hint)]
    # reset to 0 the shared grad
    zero = sharedX_value(0.)
    update_reset_grad_sup_sh = [(gsh, gsh*zero) for gsh in grad_sup_sh]
    update_reset_grad_hint_sh = None
    if hint:
        update_reset_grad_hint_sh = [(gsh, gsh*zero) for gsh in grad_hint_sh]
    # SGD: updates
    lr_sc = list(
        [sharedX_value(1.) for i in xrange(len(params))])
    lr = learning_rate
    if updater is not None:
        updates = updater.get_updates(lr, params, all_grads, lr_sc)
    else:
        updates = [
            (param, param - lr * grad)
            for (param, grad) in zip(params, all_grads)]

    # Train function
    # sup
    # because the size pairs can differ from mini-batch to another
    # we decide to use indexes instead.
    i_x = T.lvector("ix")  # vector index of sup. points
    i_x1 = T.lvector("ix1")  # vector index of the first element of (x1, x2)
    i_x2 = T.lvector("ix2")  # vector index of the 2nd element of (x1, x2)
    i_sim = T.lvector("isim")  # vector index of the of the similarties
    # Create list of functions to grab the gradient value (norm)
    fn_grad_frob_hint = None
    fn_grad_frob_hint = None
    fn_grad_frob_sup = theano.function(
        [i_x], [(grad**2).sum()**(1/2.) for grad in grad_sup],
        givens={
                model.x: model.trainx_sh[i_x],
                model.trg: model.trainy_sh[i_x]},
        on_unused_input="ignore")
#    for grad in grad_sup:
#        list_fns_grad_frob_sup.append(
#            theano.function([i_x], (grad**2).sum()**(1/2.),
#                            givens={
#                                model.x: model.trainx_sh[i_x],
#                                model.trg: model.trainy_sh[i_x]},
#            on_unused_input="ignore")
#        )
    if hint:
        fn_grad_frob_hint = theano.function(
            [i_x1, i_x2, i_sim],
            [(grad**2).sum()**(1/2.) for grad in grad_hint],
            givens={model.x1: model.trainx_sh[i_x1],
                    model.x2: model.trainx_sh[i_x2],
                    model.similair: model.similair_bases[i_sim]},
            on_unused_input="ignore")
#        for grad in grad_hint:
#            list_fns_grad_frob_hint.append(
#                theano.function(
#                    [i_x1, i_x2, i_sim], (grad**2).sum()**(1/2.),
#                    givens={model.x1: model.trainx_sh[i_x1],
#                            model.x2: model.trainx_sh[i_x2],
#                            model.similair: model.similair_bases[i_sim]},
#                    on_unused_input="ignore")
#            )
    # Prepare functions to calculate the penality value

    fn_hint_value = theano.function(
        [i_x1, i_x2, i_sim],
        [T.mean(
            l.get_penalty(model.similair, model.margin))
            for l in model.layers],
        givens={model.x1: model.trainx_sh[i_x1],
                model.x2: model.trainx_sh[i_x2],
                model.similair: model.similair_bases[i_sim]},
        on_unused_input="ignore")
    # Inspection:
    # Compute the convergence value.
#    fn_conv_value = theano.function(
#        [i_x1, i_x2, i_sim],
#        [T.mean(
#            l.insepct_get_l1_conv(model.similair, model.margin))
#            for l in model.layers],
#        givens={model.x1: model.trainx_sh[i_x1],
#                model.x2: model.trainx_sh[i_x2],
#                model.similair: model.similair_bases[i_sim]},
#        on_unused_input="ignore")
#    for l in model.layers:
#        if l.hint is not None:
#            l_fns_hint_value.append(
#                theano.function(
#                    [i_x1, i_x2, i_sim],
#                    T.mean(l.get_penalty(model.similair, model.margin)),
#                    givens={model.x1: model.trainx_sh[i_x1],
#                            model.x2: model.trainx_sh[i_x2],
#                            model.similair: model.similair_bases[i_sim]},
#                    on_unused_input="ignore"))
    # i_sim will index model.similarity=[0, 1].
    # fn for sup. term
    fn_sup = theano.function(
        [i_x], [cost_sup],
        givens={
            model.x: model.trainx_sh[i_x],
            model.trg: model.trainy_sh[i_x]},
        updates=updates_sh_gr_sup,
        on_unused_input="ignore")

    # This option slows down the execution!!!!!!!
#        mode=NanGuardMode(
#            nan_is_error=True, inf_is_error=True, big_is_error=True)

    fn_hint = None
    if hint:
        fn_hint = theano.function(
            [i_x1, i_x2, i_sim], [cost_hint],
            givens={model.x1: model.trainx_sh[i_x1],
                    model.x2: model.trainx_sh[i_x2],
                    model.similair: model.similair_bases[i_sim]},
            updates=updates_sh_gr_hint,
            on_unused_input="ignore")
#            mode=NanGuardMode(
#                nan_is_error=True, inf_is_error=True, big_is_error=True))

    fn_update_params = theano.function(
        [], [], updates=updates,
        on_unused_input="ignore")
#        mode=NanGuardMode(
#                nan_is_error=True, inf_is_error=True, big_is_error=True))

    fn_reste_sh_grad_sup = theano.function(
        [], [], updates=update_reset_grad_sup_sh,
        on_unused_input="ignore")
#        mode=NanGuardMode(
#                nan_is_error=True, inf_is_error=True, big_is_error=True))

    fn_reste_sh_grad_hint = None
    if hint:
        fn_reste_sh_grad_hint = theano.function(
            [], [], updates=update_reset_grad_hint_sh,
            on_unused_input="ignore")
#            mode=NanGuardMode(
#                    nan_is_error=True, inf_is_error=True, big_is_error=True))

    # Eval function
    # Get the model classification error
    i_x_vl = T.lvector("ixvl")
    y_vl = T.vector("y")
    error = T.mean(T.neq(T.argmax(model.output, axis=1), y_vl))

    output_fn_vl = [error]

    eval_fn = theano.function(
        [i_x_vl], output_fn_vl,
        givens={model.x: model.validx_sh[i_x_vl],
                y_vl: model.validlabels_sh[i_x_vl]})

    eval_fn_tr = theano.function(
        [i_x_vl], output_fn_vl,
        givens={model.x: model.trainx_sh[i_x_vl],
                y_vl: model.trainlabels_sh[i_x_vl]})
    fns = {"fn_sup": fn_sup, "fn_hint": fn_hint,
           "fn_update_params": fn_update_params,
           "fn_reste_sh_grad_hint": fn_reste_sh_grad_hint,
           "fn_reste_sh_grad_sup": fn_reste_sh_grad_sup,
           "eval_fn": eval_fn,
           "eval_fn_tr": eval_fn_tr,
           "fn_grad_frob_sup": fn_grad_frob_sup,
           "fn_grad_frob_hint": fn_grad_frob_hint,
           "fn_hint_value": fn_hint_value}
    return fns


def theano_fns_double_up(model,
                         learning_rate,
                         h_w, s_w, unsup_w, lambda_sparsity,
                         updater=None,
                         tag="noHint",
                         max_colm_norm=False, max_norm=15.0, norm_gsup=False,
                         norm_gh=False):
    """
    Params:
        norm_gsup: Boolean. If True, the supervised gradient will be normalized
        (g/l2(g)) before it is fed to the get_updates().
        norm_gh: Boolean. If True, the hint gradient will be normalized
        (g/l2(g)) before it is fed to the get updates() or the accumulator.
        updater: dict. Dict of updaters, one for the supervised term and
            the second for the hint term.
        see:
        https://goo.gl/dbQqeN
        https://goo.gl/89ywtn
        https://goo.gl/1U21d0
    """
    cost_type = CostType.NegativeLogLikelihood
    # Get pure cost net
    cost_net_pure = get_net_pure_cost(model, cost_type)
    hint = True
    sparsity = None
    if tag == "noHint":
        print "Train: NO hint"
        hint = False
    elif tag == "hint":
        print "Train: hint"
        penalty = None
        sparsity = None
        penalty_unsup = None
        for l in model.layers:
            if l.use_hint:
                if penalty is None:
                    penalty = l.get_penalty(model.similair, model.margin)
                else:
                    penalty += l.get_penalty(model.similair, model.margin)
                if l.use_unsupervised:
                    if penalty_unsup is None:
                        penalty_unsup = l.ae.get_train_cost_clean()
                    else:
                        penalty_unsup += l.ae.get_train_cost_clean()
                # Sparsity
                if l.use_sparsity:
                    assert l.use_sparsity is None
                    if sparsity is None:
                        sparsity = abs(l.sparser).sum()
                    else:
                        sparsity += abs(l.sparser).sum()

    cost_hint = None
    cost_unsup = None
    cost_sup = s_w * T.mean(cost_net_pure)
    if hint:
        if sparsity is not None:
            cost_hint = h_w * T.mean(penalty) + lambda_sparsity * sparsity
        else:
            cost_hint = h_w * T.mean(penalty)
        if penalty_unsup is not None:
            cost_unsup = unsup_w * T.mean(penalty_unsup)
        if model.l1 != 0.:
            cost_hint += model.l1
        if model.l2 != 0.:
            cost_hint += model.l2

#    if model.l1 != 0.:
#        cost_sup += model.l1
#    if model.l2 != 0.:
#        cost_sup += model.l2

    assert sparsity is None
    # Gradients:
    params = model.params  # + [l.sparser for l in model.layers]
    params_sup = model.params
    for l in model.layers:
        if l.bn is not None:
            params_sup += l.bn.params

    params = params_sup  # include the bn params into the hint params.
    params_unsup = []
    if hint and cost_unsup is not None:
        for l in model.layers:
            if l.ae is not None:
                params_unsup.extend(l.ae.params)

    grad_sup = T.grad(cost_sup, params_sup, disconnected_inputs='warn')
    # Normalize the gradient: g <--- g / l2(g)
    if norm_gsup:
        for ig in range(len(grad_sup)):
            grad_sup[ig] = normalize_grad(grad_sup[ig])
    # "warn" instead of "raise")
    grad_hint = None
    grad_unsup = None
    if hint:
        grad_hint = T.grad(cost_hint, params,
                           disconnected_inputs='warn')  # instead of "raise")
        if norm_gh:
            for ig in range(len(grad_hint)):
                grad_hint[ig] = normalize_grad(grad_hint[ig])
        if cost_unsup is not None:
            # For now, we calculate the gradient with rexpect to ae params ONLY
            grad_unsup = T.grad(cost_unsup, params_unsup,
                                disconnected_inputs='warn')
    # http://deeplearning.net/software/theano/library/gradient.html#theano.gradient.grad

    # SGD: updates
    lr_sc_sup = list(
        [sharedX_value(1.) for i in xrange(len(params_sup))])
    lr_sc_hint = list(
        [sharedX_value(1.) for i in xrange(len(params))])
    lr_sc_unsup = list(
        [sharedX_value(1.) for i in xrange(len(params_unsup))])
    lr = learning_rate
    # The difference of the double updaters starts here.
    updates_sup, updates_hint, updates_unsup = None, None, None
    if updater is not None:
        updates_sup = updater['sup'].get_updates(
            lr, params_sup, grad_sup, lr_sc_sup)
        if hint:
            updates_hint = updater['hint'].get_updates(
                lr, params, grad_hint, lr_sc_hint)
        if cost_unsup is not None:
            updates_unsup = updater['unsup'].get_updates(
                lr, params_unsup, grad_unsup, lr_sc_unsup)
    else:
        updates_sup = [
            (param, param - lr * grad)
            for (param, grad) in zip(params_sup, grad_sup)]
        if hint:
            updates_hint = [
                (param, param - lr * grad)
                for (param, grad) in zip(params, grad_hint)]
        if cost_unsup is not None:
            updates_unsup = [
                (param, param - lr * grad)
                for (param, grad) in zip(params, grad_unsup)]

    # Train function
    # sup
    # because the size pairs can differ from mini-batch to another
    # we decide to use indexes instead.
    i_x = T.lvector("ix")  # vector index of sup. points
    i_x1 = T.lvector("ix1")  # vector index of the first element of (x1, x2)
    i_x2 = T.lvector("ix2")  # vector index of the 2nd element of (x1, x2)
    i_sim = T.lvector("isim")  # vector index of the of the similarties
    # Create list of functions to grab the gradient value (norm)
    fn_grad_frob_hint = None
    fn_grad_frob_hint = None
    fn_grad_frob_sup = theano.function(
        [i_x], [(grad**2).sum()**(1/2.) for grad in grad_sup],
        givens={
                model.x: model.trainx_sh[i_x],
                model.trg: model.trainy_sh[i_x]},
        on_unused_input="ignore")
#    for grad in grad_sup:
#        list_fns_grad_frob_sup.append(
#            theano.function([i_x], (grad**2).sum()**(1/2.),
#                            givens={
#                                model.x: model.trainx_sh[i_x],
#                                model.trg: model.trainy_sh[i_x]},
#            on_unused_input="ignore")
#        )
    if hint:
        fn_grad_frob_hint = theano.function(
            [i_x1, i_x2, i_sim],
            [(grad**2).sum()**(1/2.) for grad in grad_hint],
            givens={model.x1: model.trainx_sh[i_x1],
                    model.x2: model.trainx_sh[i_x2],
                    model.similair: model.similair_bases[i_sim]},
            on_unused_input="ignore")
#        for grad in grad_hint:
#            list_fns_grad_frob_hint.append(
#                theano.function(
#                    [i_x1, i_x2, i_sim], (grad**2).sum()**(1/2.),
#                    givens={model.x1: model.trainx_sh[i_x1],
#                            model.x2: model.trainx_sh[i_x2],
#                            model.similair: model.similair_bases[i_sim]},
#                    on_unused_input="ignore")
#            )
    # Prepare functions to calculate the penality value

    fn_hint_value = theano.function(
        [i_x1, i_x2, i_sim],
        [T.mean(
            l.get_penalty(model.similair, model.margin))
            for l in model.layers],
        givens={model.x1: model.trainx_sh[i_x1],
                model.x2: model.trainx_sh[i_x2],
                model.similair: model.similair_bases[i_sim]},
        on_unused_input="ignore")
    # Inspection:
    # Compute the convergence value.
#    fn_conv_value = theano.function(
#        [i_x1, i_x2, i_sim],
#        [T.mean(
#            l.insepct_get_l1_conv(model.similair, model.margin))
#            for l in model.layers],
#        givens={model.x1: model.trainx_sh[i_x1],
#                model.x2: model.trainx_sh[i_x2],
#                model.similair: model.similair_bases[i_sim]},
#        on_unused_input="ignore")
#    for l in model.layers:
#        if l.hint is not None:
#            l_fns_hint_value.append(
#                theano.function(
#                    [i_x1, i_x2, i_sim],
#                    T.mean(l.get_penalty(model.similair, model.margin)),
#                    givens={model.x1: model.trainx_sh[i_x1],
#                            model.x2: model.trainx_sh[i_x2],
#                            model.similair: model.similair_bases[i_sim]},
#                    on_unused_input="ignore"))
    # i_sim will index model.similarity=[0, 1].
    # fn for sup. term
    fn_sup = theano.function(
        [i_x], [cost_sup],
        givens={
            model.x: model.trainx_sh[i_x],
            model.trg: model.trainy_sh[i_x]},
        updates=updates_sup,
        on_unused_input="ignore")

    # This option slows down the execution!!!!!!!
#        mode=NanGuardMode(
#            nan_is_error=True, inf_is_error=True, big_is_error=True)

    fn_hint = None
    if hint:
        fn_hint = theano.function(
            [i_x1, i_x2, i_sim], [cost_hint],
            givens={model.x1: model.trainx_sh[i_x1],
                    model.x2: model.trainx_sh[i_x2],
                    model.similair: model.similair_bases[i_sim]},
            updates=updates_hint,
            on_unused_input="ignore")
#            mode=NanGuardMode(
#                nan_is_error=True, inf_is_error=True, big_is_error=True))

    fn_update_params_sup = None
#    fn_update_params_sup = theano.function(
#        [], [], updates=updates_sup,
#        on_unused_input="ignore")
    fn_update_params_hint = None
#    if hint:
#        fn_update_params_hint = theano.function(
#            [], [], updates=updates_hint,
#            on_unused_input="ignore")
    fn_update_params_unsup = None
    if cost_unsup is not None:
        fn_update_params_unsup = theano.function(
            [i_x], [cost_unsup], updates=updates_unsup,
            givens={
                model.x: model.trainx_sh[i_x]},
            on_unused_input="ignore")
#        mode=NanGuardMode(
#                nan_is_error=True, inf_is_error=True, big_is_error=True))

    fn_reste_sh_grad_sup = None
#    fn_reste_sh_grad_sup = theano.function(
#        [], [], updates=update_reset_grad_sup_sh,
#        on_unused_input="ignore")
#        mode=NanGuardMode(
#                nan_is_error=True, inf_is_error=True, big_is_error=True))

    fn_reste_sh_grad_hint = None
#    if hint:
#        fn_reste_sh_grad_hint = theano.function(
#            [], [], updates=update_reset_grad_hint_sh,
#            on_unused_input="ignore")
#            mode=NanGuardMode(
#                    nan_is_error=True, inf_is_error=True, big_is_error=True))

    # Eval function
    # Get the model classification error
    i_x_vl = T.lvector("ixvl")
    y_vl = T.vector("y")
    error = T.mean(T.neq(T.argmax(model.output_vl, axis=1), y_vl))

    output_fn_vl = [error]

    eval_fn = theano.function(
        [i_x_vl], output_fn_vl,
        givens={model.x: model.validx_sh[i_x_vl],
                y_vl: model.validlabels_sh[i_x_vl]})

    eval_fn_tr = theano.function(
        [i_x_vl], output_fn_vl,
        givens={model.x: model.trainx_sh[i_x_vl],
                y_vl: model.trainlabels_sh[i_x_vl]})
    fns = {"fn_sup": fn_sup, "fn_hint": fn_hint,
           "fn_update_params_sup": fn_update_params_sup,
           "fn_update_params_hint": fn_update_params_hint,
           "fn_update_params_unsup": fn_update_params_unsup,
           "fn_reste_sh_grad_hint": fn_reste_sh_grad_hint,
           "fn_reste_sh_grad_sup": fn_reste_sh_grad_sup,
           "eval_fn": eval_fn,
           "eval_fn_tr": eval_fn_tr,
           "fn_grad_frob_sup": fn_grad_frob_sup,
           "fn_grad_frob_hint": fn_grad_frob_hint,
           "fn_hint_value": fn_hint_value}
    return fns


def get_eval_fn(model):
    """Compile the evaluation function of the model."""
    # Get the model classification error
    x = T.fmatrix("x")
    y = T.vector('y')
    error = T.mean(T.neq(T.argmax(model.output, axis=1), y))

    theano_arg_vl = [x, y]
    output_fn_vl = [error]

    eval_fn = theano.function(
        theano_arg_vl, output_fn_vl,
        givens={model.x: x})

    return eval_fn


def evaluate_model(list_minibatchs_vl, eval_fn):
    """Evalute the model over a set."""
    error = None
    for mn_vl in list_minibatchs_vl:
        x = theano.shared(
            mn_vl['x'], borrow=True).get_value(borrow=True)
        y = theano.shared(mn_vl['y'], borrow=True).get_value(borrow=True)
        [error_mn] = eval_fn(x, y)
        if error is None:
            error = error_mn
        else:
            error = np.vstack((error, error_mn))

    return error


def plot_fig(values, title, x_str, y_str, path, best_iter, std_vals=None):
    """Plot some values.
    Input:
         values: list or numpy.ndarray of values to plot (y)
         title: string; the title of the plot.
         x_str: string; the name of the x axis.
         y_str: string; the name of the y axis.
         path: string; path where to save the figure.
         best_iter: integer. The epoch of the best iteration.
         std_val: List or numpy.ndarray of standad deviation values that
             corresponds to each value in 'values'.
    """
    floating = 6
    prec = "%." + str(floating) + "f"

    if best_iter >= 0:
        if isinstance(values, list):
            if best_iter >= len(values):
                best_iter = -1
        if isinstance(values, np.ndarray):
            if best_iter >= np.size:
                best_iter = -1

        v = str(prec % np.float(values[best_iter]))
    else:
        v = str(prec % np.float(values[-1]))
        best_iter = -1
    if best_iter == -1:
        best_iter = len(values)
    fig = plt.figure()
    plt.plot(
        values,
        label="lower val: " + v + " at " + str(best_iter) + " " +
        x_str)
    plt.xlabel(x_str)
    plt.ylabel(y_str)
    plt.title(title, fontsize=8)
    plt.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': 8})
    plt.grid(True)
    fig.savefig(path, bbox_inches='tight')
    plt.close('all')
    del fig


def plot_stats(train_stats, ax, fold_exp, tag):
    """Plot what happened during training.
    Input:
        ax: string. "mb": minbatch, "ep": epoch.
    """
    if ax is "mn":
        fd = fold_exp + "/minibatches"
        extra1 = "_mn"
        extra2 = "minibatches"
        extra3 = train_stats["best_mn"]
    else:
        fd = fold_exp + "/epochs"
        extra1 = "_ep"
        extra2 = "epochs"
        extra3 = train_stats["best_epoch"]
    if not os.path.exists(fd):
        os.makedirs(fd)

    if train_stats["tr_error" + extra1] != []:
        plot_fig(train_stats["tr_error" + extra1],
                 "Train error: [" + extra2 + "]. Case: " + tag,
                 extra2, "Class.error",
                 fd+"/train-class-error-" + extra2 + "_" + tag + ".png",
                 extra3)

    if train_stats["vl_error" + extra1] != []:
        plot_fig(train_stats["vl_error" + extra1],
                 "Valid error: [" + extra2 + "]. Case: " + tag,
                 extra2, "Class.error",
                 fd+"/valid-class-error-" + extra2 + "_" + tag + ".png",
                 extra3)

    if train_stats["tr_cost" + extra1] != []:
        plot_fig(train_stats["tr_cost" + extra1],
                 "Train cost: [" + extra2 + "]. Case: " + tag,
                 extra2, "Train cost",
                 fd+"/train-cost-" + extra2 + "_" + tag + ".png",
                 extra3)


def print_stats_train(train_stats, epoch, ext, mb):
    in_cost, out_cost, all_cost, tr_pure_cost = 0., 0., 0., 0.
    error_vl, error_tr, code_cost = 0., 0., 0.

    if train_stats["code_cost"+ext] != []:
        code_cost = str(prec2 % train_stats["code_cost"+ext][-1])
    if train_stats["in_cost"+ext] != []:
        in_cost = str(prec2 % train_stats["in_cost"+ext][-1])
    if train_stats["out_cost"+ext] != []:
        out_cost = str(prec2 % train_stats["out_cost"+ext][-1])
    if train_stats["tr_pure_cost"+ext] != []:
        tr_pure_cost = str(prec2 % train_stats["tr_pure_cost"+ext][-1])
    if train_stats["error_tr"+ext]:
        error_tr = str(prec2 % train_stats["error_tr"+ext][-1])
    all_cost = str(prec2 % train_stats["all_cost"+ext][-1])
    error_vl = str(prec2 % train_stats["error_vl"+ext][-1])
    min_vl_err = str(prec2 % min(train_stats["error_vl"+ext]))

    if ext is "":
        print "\r Epoch [", str(epoch), "] Train infos:",
    else:
        print "\r Epoch [", str(epoch), "] mb [", str(mb), "] Train infos:",
#    print "\t all cost:", all_cost, " \t tr pure cost:", tr_pure_cost
#    print "\t in cost:", in_cost, " \t out cost:", out_cost
#    print "\t error tr:", error_tr, " \t error vl:", error_vl, " min vl:",\
#        min_vl_err
#    print "\t all cost: %s ,\t tr pure cost: %s, \t in cost: %s,"\
#        "\t out cost: %s,\t error tr: %s, \t error vl: %s, \t min vl: %s"\
#        % (all_cost, tr_pure_cost, in_cost, out_cost, error_tr, error_vl,
#            min_vl_err)
    print "all cost:", all_cost, "\t tr pure cost:", tr_pure_cost,\
        "\t in cost:", in_cost, "\t  out cost:", out_cost, "\t error tr:",\
        error_tr, "\t code cost:", code_cost, "\t error vl:", error_vl,\
        "\t min vl:", min_vl_err,
    sys.stdout.flush()


def plot_debug_grad(debug, tag, fold_exp, trg):
    plt.close("all")
    # f = plt.figure(figsize=(15, 10.8), dpi=300)
    nbr_rows = int(len(debug["grad_sup"][0])/2)
    f, axs = plt.subplots(nbr_rows, 2, sharex=True, sharey=False,
                          figsize=(15, 12.8), dpi=300)

    if trg == "sup":
        grad = np.array(debug["grad_sup"])
    elif trg == "hint":
        grad = np.array(debug["grad_hint"])
    print grad.shape, trg
    j = 0
    for i in range(0, nbr_rows*2, 2):
        w_vl = grad[:, i]
        b_vl = grad[:, i+1]
        axs[j, 0].plot(w_vl, label=trg)
        axs[j, 0].set_title("w"+str(j))
        axs[j, 1].plot(b_vl, label=trg)
        axs[j, 1].set_title("b"+str(j))
        axs[j, 0].grid(True)
        axs[j, 1].grid(True)
        j += 1
    f.suptitle("Grad sup/hint:" + tag, fontsize=8)
    plt.legend()
    f.savefig(fold_exp+"/grad_" + trg + ".png", bbox_inches='tight')
    plt.close("all")
    del f


def plot_debug_ratio_grad(debug, fold_exp, r="h/s"):
    plt.close("all")
    # f = plt.figure(figsize=(15, 10.8), dpi=300)
    nbr_rows = int(len(debug["grad_sup"][0])/2)
    f, axs = plt.subplots(nbr_rows, 2, sharex=True, sharey=False,
                          figsize=(15, 12.8), dpi=300)

    grads = np.array(debug["grad_sup"])
    gradh = np.array(debug["grad_hint"])
    if gradh.size != grads.size:
        print "Can't calculate the ratio. It looks like you divided the " +\
            "hint batch..."
        return 0
    print gradh.shape, grads.shape
    j = 0
    for i in range(0, nbr_rows*2, 2):
        w_vls = grads[:, i]
        b_vls = grads[:, i+1]
        w_vl_h = gradh[:, i]
        b_vlh = gradh[:, i+1]
        if r == "h/s":
            ratio_w = np.divide(w_vl_h, w_vls)
            ratio_b = np.divide(b_vlh, b_vls)
        elif r == "s/h":
            ratio_w = np.divide(w_vls, w_vl_h)
            ratio_b = np.divide(b_vls, b_vlh)
        else:
            raise ValueError("Either h/s or s/h.")
        axs[j, 0].plot(ratio_w, label=r)
        axs[j, 0].set_title("w"+str(j))
        axs[j, 1].plot(ratio_b, label=r)
        axs[j, 1].set_title("b"+str(j))
        axs[j, 0].grid(True)
        axs[j, 1].grid(True)
        j += 1
    f.suptitle("Ratio gradient: " + r, fontsize=8)
    plt.legend()
    f.savefig(fold_exp+"/ratio_grad_" + r.replace("/", "-") + ".png",
              bbox_inches='tight')
    plt.close("all")
    del f


def plot_penalty_vl(debug, tag, fold_exp):
    plt.close("all")
    vl = np.array(debug["penalty"])
    fig = plt.figure(figsize=(15, 10.8), dpi=300)
    names = debug["names"]
    for i in range(vl.shape[1]):
        if vl.shape[1] > 1:
            plt.plot(vl[:, i], label="layer_"+str(names[i]))
        else:
            plt.plot(vl[:], label="layer_"+str(names[i]))
    plt.xlabel("mini-batchs")
    plt.ylabel("value of penlaty")
    plt.title(
        "Penalty value over layers:" + "_".join([str(k) for k in names]) +
        ". tag:" + tag)
    plt.legend(loc='upper right', fancybox=True, shadow=True, prop={'size': 8})
    plt.grid(True)
    fig.savefig(fold_exp+"/penalty.png", bbox_inches='tight')
    plt.close('all')
    del fig


def pair_samples(x, y):
    """Pair similar/dissimalr samples (x1, x2, yi).
    yi: 0: similar. 1: dissimilar.
    """
    failed = True
    y = y.astype(np.int)
    uniq = np.unique(y)
    # Build indices:
    l_ind1, l_ind2, l_sim = [], [], []
    for u in uniq:
        # Similair
        indu = np.where(y == u)[0]
        # Build the combination
        # if there is more than 1 element in the class: build combin.
        # else: ignore the sample; Too bad for you.
        if indu.size > 1:
            comb1, comb2, s = [], [], []
            for i in xrange(indu.size - 1):
                for j in xrange(i+1, indu.size):
                    comb1.append(indu[i])
                    comb2.append(indu[j])
                    s.append(0)  # similar
            l_ind1.extend(comb1)
            l_ind2.extend(comb2)
            l_sim.extend(s)
        # Dissimilar
        indu_others = np.where(y != u)[0]
        if indu_others.size > 1:
            comb1, comb2, s = [], [], []
            for i in xrange(indu.size):
                for j in xrange(indu_others.size):
                    comb1.append(indu[i])
                    comb2.append(indu_others[j])
                    s.append(1)  # dissimilar
            l_ind1.extend(comb1)
            l_ind2.extend(comb2)
            l_sim.extend(s)
    # check if we find at least a pair
    if l_ind1 != []:
        failed = False
        similair = np.array(l_sim, dtype=theano.config.floatX)
        x1 = x[l_ind1[:]]
        x2 = x[l_ind2[:]]
        print "from similair: ", x.shape[0], " to: ", x1.shape[0]
        return x1, x2, similair, failed
    else:
        return None, None, None, failed


def pair_samples_index(y, div):
    """Pair similar/dissimalr samples (x1, x2, yi), (x, y).
    yi: 0: similar. 1: dissimilar.
    Returns only the indexes.
    #positive == #negative.
    div: boolean. if True, take also the dissimilar points.
    """
    failed = True
    y = y.astype(np.int)
    uniq = np.unique(y)
    l_ind1, l_ind2, l_sim = [], [], []
    l_check = []
    sim, notsim = 0, 0
    for u in uniq:
        # Similair
        indu = np.where(y == u)[0]
        indu_others = np.where(y != u)[0]
        # Build the combination
        # if there is more than 1 element in the class: build combin.
        # else: ignore the sample; Too bad for you.
        if indu.size > 1:
            comb1, comb2, s = [], [], []
            for i in xrange(indu.size - 1):
                for j in xrange(i+1, indu.size):
                    comb1.append(indu[i])
                    comb2.append(indu[j])
                    sim += 1
                    s.append(0)  # similar
                    if div:
                        ind = np.random.randint(0, indu_others.size)
                        xx = [indu[i], indu_others[ind]]
                        if xx not in l_check and xx[::-1] not in l_check:
                            comb1.append(indu[i])
                            comb2.append(indu_others[ind])
                            s.append(1)  # dissimilar
                            notsim += 1
            l_ind1.extend(comb1)
            l_ind2.extend(comb2)
            l_sim.extend(s)

    # check if we find at least a pair

    if l_ind1 != []:
        failed = False
        print "from similair: ", y.shape[0], " to: ", len(l_ind1)
        return np.array(l_ind1, dtype=int), np.array(l_ind2, dtype=int),\
            l_sim, failed
    else:
        return None, None, None, failed


def pair_all_data(y, batch_size, div):
    """
    div: boolean. if True, take also the dissimilar points.
    """
    all_list = []
    for i in range(0, y.shape[0], batch_size):
        y_tmp = y[i:i+batch_size]
        l_ind1, l_ind2, l_sim, failed = pair_samples_index(y_tmp, div)
        if i+batch_size <= y.shape[0]:
            l_sup = np.arange(i, i+batch_size, dtype=int)
        else:
            l_sup = np.arange(i, y.shape[0], dtype=int)

        all_list.append([l_sup, l_ind1+i, l_ind2+i, l_sim, failed])

    return all_list


def train_one_epoch(model, fns,
                    epoch, fold_exp, train_stats, vl_err_start, tag,
                    train_batch_size, l_vl, l_tr, div,
                    stop=False, debug=None, debug_code=False):
    """ train the model's parameters for 1 epoch.
    div: boolean. if True, take also the dissimilar points.
    """
    # get the fns
    fn_sup, fn_hint = fns["fn_sup"], fns["fn_hint"]
    fn_update_params = fns["fn_update_params"]
    fn_reste_sh_grad_hint = fns["fn_reste_sh_grad_hint"]
    fn_reste_sh_grad_sup = fns["fn_reste_sh_grad_sup"]
    eval_fn, eval_fn_tr = fns["eval_fn"], fns["eval_fn_tr"]
    fn_grad_frob_sup = fns["fn_grad_frob_sup"]
    fn_grad_frob_hint = fns["fn_grad_frob_hint"]
    fn_hint_value = fns["fn_hint_value"]
    error_vl, error_tr, cost_train = [], [], []
    nb_mb, best_epoch, best_mb = train_stats["current_nb_mb"], None, None
    if epoch <= 1000:
        freq_vl = 2000
    elif epoch <= 1900 and epoch > 1000:
        freq_vl = 2000
    elif epoch > 1900:
        freq_vl = 1
    freq_vl = 8
    plot_freq = 20
    catchted_once = False
    # prepare the indexes
    y_train = model.trainlabels_sh.get_value()
    all_list = pair_all_data(y_train, train_batch_size, div)
    c_mn = 0
    for ch in all_list:
        print "Processing the ", c_mn, "/", len(all_list)
        c_mn += 1
        l_sup, l_ind1, l_ind2, l_sim, failed = ch
        cost_sup, cost_hint = 0., 0.
        tOO = DT.datetime.now()
        cost_sup = fn_sup(l_sup)[0]
        if debug_code:
            debug["grad_sup"].append(fn_grad_frob_sup(l_sup))
            # print "grad sup:", debug["grad_sup"][-1]

        print "train sup took", DT.datetime.now() - tOO
        # Pair samples
        if tag == "noHint":
            # Updates
            txx = DT.datetime.now()
            fn_update_params()
            print "update params took", DT.datetime.now() - txx
            txx = DT.datetime.now()
            fn_reste_sh_grad_sup()
            print "rest grad sup took", DT.datetime.now() - txx
        else:
            if failed:  # if there is no pairs!!!!!
                txx = DT.datetime.now()
                fn_update_params()
                print "update params took", DT.datetime.now() - txx
                txx = DT.datetime.now()
                fn_reste_sh_grad_sup()
                print "rest grad sup took", DT.datetime.now() - txx
                continue
            # Updates
            tOO = DT.datetime.now()
            cost_hint = fn_hint(l_ind1, l_ind2, l_sim)[0]
            if debug_code:
                debug["grad_hint"].append(
                    fn_grad_frob_hint(l_ind1, l_ind2, l_sim))
                # print "Grad hint:", debug["grad_hint"][-1]
                debug["penalty"].append(fn_hint_value(l_ind1, l_ind2, l_sim))
                # print "Penalty value:", debug["penalty"][-1]
            fn_update_params()
            fn_reste_sh_grad_sup()
            fn_reste_sh_grad_hint()
            print "train hint took", DT.datetime.now() - tOO
        if debug_code:
            debug["penalty"].append(fn_hint_value(l_ind1, l_ind2, l_sim))
            # print "Penalty value:", debug["penalty"][-1]
        cost = cost_sup + cost_hint
        print "Train cost:", cost
        cost_train.append(cost)
        train_stats["tr_cost_mn"].append(cost)
        # Eval over valid and train: if it's validation time!

        if (nb_mb % freq_vl == 0):
            error_tmp = np.mean(
                    [eval_fn(np.array(l_vl[kk])) for kk in range(len(l_vl))])
            error_vl.append(error_tmp)
            train_stats["vl_error_mn"].append(error_vl[-1])
            # Pick the best model according to the validation error.
            if len(train_stats["vl_error_mn"]) > 2:
                min_vl = np.min(train_stats["vl_error_mn"][:-1])
            else:
                min_vl = vl_err_start
            print "vl error", error_vl[-1], epoch, " best:", min_vl,\
                "min vl:", min(min_vl, error_vl[-1])
            if error_vl[-1] < min_vl:
                min_vl = error_vl[-1]
                train_stats["best_epoch"] = epoch
                train_stats["best_mn"] = nb_mb
                model.catch_params()
                catchted_once = True

        nb_mb += 1

    # Evaluate model over train set.
    # Do it only once at the end of the epoch. It's enough.
    # With large train set, this may be very time consuming.
    error_tmp = np.mean(
        [eval_fn_tr(np.array(l_tr[kk])) for kk in range(len(l_tr))])
    error_tr.append(error_tmp)
    print "Train error:", error_tr[-1]
    train_stats["tr_error_mn"].append(error_tr[-1])
    # save on disc best models.
    if (epoch % 200 == 0) or stop:
        if stop:
            print "Going to end the training, but before, I'm gonna go ahead "\
                " and save the model params."
        # Plot stats: mb
        plot_stats(train_stats, "mn", fold_exp, tag)
        if catchted_once:
            model.save_params(fold_exp + "/model.pkl", catched=True)

    # Stats
    stats = {"tr_error": error_tr,
             "vl_error": error_vl,
             "tr_cost": cost_train,
             "best_epoch": best_epoch,
             "best_mn": best_mb,
             "current_nb_mb": nb_mb}

    return stats


def shuffl_batch_hint(l_ind1, l_ind2, l_sim):
    new_l1 = np.array(l_ind1)
    new_l2 = np.array(l_ind2)
    new_sim = np.array(l_sim, dtype=np.int)
    new_l1 = new_l1.reshape(l_ind1.size, 1)
    new_l2 = new_l2.reshape(l_ind2.size, 1)
    new_sim = new_sim.reshape(new_sim.size, 1)
    mega_stuff = np.hstack((new_l1, new_l2, new_sim))
    for i in range(100):
        np.random.shuffle(mega_stuff)
    new_l1 = mega_stuff[:, 0]
    new_l2 = mega_stuff[:, 1]
    new_sim = list(mega_stuff[:, 2])
    return new_l1, new_l2, new_sim


def train_one_epoch_alter(model, fns,
                          epoch, fold_exp, train_stats, vl_err_start, tag,
                          train_batch_size, l_vl, l_tr, div,
                          stop=False, debug=None, debug_code=False, h_w=None):
    """ train the model's parameters for 1 epoch.
    Alternate between the hint term and the supervised term:
    - Make a gradient step toward the hint term.
    - Make a gradient step toward the supervised term.
    """
    splitit = False
    # get the fns
    fn_sup, fn_hint = fns["fn_sup"], fns["fn_hint"]
    fn_update_params_sup = fns["fn_update_params_sup"]
    fn_update_params_hint = fns["fn_update_params_hint"]
    fn_update_params_unsup = fns["fn_update_params_unsup"]
    fn_reste_sh_grad_hint = fns["fn_reste_sh_grad_hint"]
    fn_reste_sh_grad_sup = fns["fn_reste_sh_grad_sup"]
    eval_fn, eval_fn_tr = fns["eval_fn"], fns["eval_fn_tr"]
    fn_grad_frob_sup = fns["fn_grad_frob_sup"]
    fn_grad_frob_hint = fns["fn_grad_frob_hint"]
    fn_hint_value = fns["fn_hint_value"]
    error_vl, error_tr, cost_train = [], [], []
    nb_mb, best_epoch, best_mb = train_stats["current_nb_mb"], None, None
    if epoch <= 1000:
        freq_vl = 2000
    elif epoch <= 1900 and epoch > 1000:
        freq_vl = 2000
    elif epoch > 1900:
        freq_vl = 1
    freq_vl = 8
    plot_freq = 20
    catchted_once = False
    # prepare the indexes
    y_train = model.trainlabels_sh.get_value()
    all_list = pair_all_data(y_train, train_batch_size, div)
    # when to validate?
    # after: x%, x=90.
    freq_vl = int((95/100.) * y_train.size/float(train_batch_size))
    c_mn = 0
    for ch in all_list:
        print "Processing the ", c_mn, "/", len(all_list)
        c_mn += 1
        l_sup, l_ind1, l_ind2, l_sim, failed = ch

        cost_sup, cost_hint = 0., 0.
        tOO = DT.datetime.now()
        print "train sup took", DT.datetime.now() - tOO
        # Pair samples
        if tag == "noHint":
            # Updates
            txx = DT.datetime.now()
            cost_sup = fn_sup(l_sup)[0]
            if debug_code:
                debug["grad_sup"].append(fn_grad_frob_sup(l_sup))
            # print "grad sup:", debug["grad_sup"][-1]
#            fn_update_params_sup()
            print "update params took", DT.datetime.now() - txx
            txx = DT.datetime.now()
#            fn_reste_sh_grad_sup()
            print "rest grad sup took", DT.datetime.now() - txx
        else:
            if failed:  # if there is no pairs!!!!!
                txx = DT.datetime.now()
#                fn_update_params()
#                print "update params took", DT.datetime.now() - txx
#                txx = DT.datetime.now()
#                fn_reste_sh_grad_sup()
                print "rest grad sup took", DT.datetime.now() - txx
                continue
            # Updates: hint then sup.
            # 1. Step toward hint term.
            if not failed:
                tOO = DT.datetime.now()
                # Forward the whole hint mini-batch!
                if not splitit:
                    cost_hint = fn_hint(l_ind1, l_ind2, l_sim)[0]
                    if debug_code:
                        debug["grad_hint"].append(
                            fn_grad_frob_hint(l_ind1, l_ind2, l_sim))
                        # print "Grad hint:", debug["grad_hint"][-1]
                        debug["penalty"].append(
                            fn_hint_value(l_ind1, l_ind2, l_sim))
                        # print "Penalty value:", debug["penalty"][-1]
#                    fn_update_params_hint()
#                    fn_reste_sh_grad_hint()
                # Split the hint mini-batch into smaller mini-btaches...
                else:
                    # Shuffle the hint batch.
                    print "Going to split the hint batch ..."
                    l_ind1, l_ind2, l_sim = shuffl_batch_hint(l_ind1, l_ind2,
                                                              l_sim)
                    # Loop over the small mini-mini-batches!!!!!
                    # Use the same size of the mini-batch!
                    # how many times you update the params using the hint
                    # term before using the supervised term: nbr_updates.
                    nbr_updates = 1
                    b_size = l_ind1.size / nbr_updates
                    for kxy in range(0, l_ind1.size,  b_size):
                        ind1 = l_ind1[kxy:kxy+b_size]
                        ind2 = l_ind2[kxy:kxy+b_size]
                        sim = l_sim[kxy:kxy+b_size]
                        cost_hint = fn_hint(ind1, ind2, sim)[0]
                        if debug_code:
                            debug["grad_hint"].append(
                                fn_grad_frob_hint(ind1, ind2, sim))
                            # print "Grad hint:", debug["grad_hint"][-1]
                            debug["penalty"].append(
                                fn_hint_value(ind1, ind2, sim))
                            # print "Penalty value:", debug["penalty"][-1]
#                        fn_update_params_hint()
#                        fn_reste_sh_grad_hint()

                print "train hint took: ", DT.datetime.now() - tOO
            # 2. Step toward the unsupervised term
            if fn_update_params_unsup is not None:
                t_unsp = DT.datetime.now()
                cost_unsup = fn_update_params_unsup(l_sup)[0]
                print "Updating unsup params took:", DT.datetime.now() - t_unsp
                print "cost_unsup: ", cost_unsup
            # 3. Step toward sup. term.
            txx = DT.datetime.now()
            cost_sup = fn_sup(l_sup)[0]
            if debug_code:
                debug["grad_sup"].append(fn_grad_frob_sup(l_sup))
            # print "grad sup:", debug["grad_sup"][-1]
#            fn_update_params_sup()
            print "train sup took", DT.datetime.now() - txx
#            fn_reste_sh_grad_sup()

        if debug_code:
            print l_ind1.shape, "***"
            debug["penalty"].append(fn_hint_value(l_ind1, l_ind2, l_sim))
            # print "Penalty value:", debug["penalty"][-1]
        if np.isinf(cost_sup):
            raise ValueError("cost sup inf.")
        if np.isnan(cost_sup):
            raise ValueError("cost sup nan.")
        if np.isinf(cost_hint):
            raise ValueError("cost hint inf.")
        if np.isnan(cost_hint):
            raise ValueError("cost hint nan.")
        cost = cost_sup + cost_hint
        print "Train cost:", cost
        if train_stats["vl_error_mn"] != []:
            print "\t\t >>>>>>>>>>Min valid: ",\
                min(train_stats["vl_error_mn"]), " Epoch: ", epoch
            if error_vl != []:
                print "\t\t >>>>>>>>>>vl: ", error_vl[-1]
        cost_train.append(cost)
        train_stats["tr_cost_mn"].append(cost)
        # Eval over valid and train: if it's validation time!

        if (nb_mb % freq_vl == 0):
            error_tmp = np.mean(
                    [eval_fn(np.array(l_vl[kk])) for kk in range(len(l_vl))])
            error_vl.append(error_tmp)
            train_stats["vl_error_mn"].append(error_vl[-1])
            # Pick the best model according to the validation error.
            if len(train_stats["vl_error_mn"]) > 2:
                min_vl = np.min(train_stats["vl_error_mn"][:-1])
            else:
                min_vl = vl_err_start
            print "vl error", error_vl[-1], epoch, " best:", min_vl,\
                "min vl:", min(min_vl, error_vl[-1]), " epoch:", epoch
            if error_vl[-1] < min_vl:
                min_vl = error_vl[-1]
                train_stats["best_epoch"] = epoch
                train_stats["best_mn"] = nb_mb
                model.catch_params()
                catchted_once = True

        nb_mb += 1

    # Evaluate model over train set.
    # Do it only once at the end of the epoch. It's enough.
    # With large train set, this may be very time consuming.
    error_tmp = np.mean(
        [eval_fn_tr(np.array(l_tr[kk])) for kk in range(len(l_tr))])
    error_tr.append(error_tmp)
    print "Train error:", error_tr[-1]
    train_stats["tr_error_mn"].append(error_tr[-1])
    # save on disc best models.
    if (epoch % 200 == 0) or stop:
        if stop:
            print "Going to end the training, but before, I'm gonna go ahead "\
                " and save the model params."
        # Plot stats: mb
        plot_stats(train_stats, "mn", fold_exp, tag)
        if catchted_once:
            model.save_params(fold_exp + "/model.pkl", catched=True)

    # Stats
    stats = {"tr_error": error_tr,
             "vl_error": error_vl,
             "tr_cost": cost_train,
             "best_epoch": best_epoch,
             "best_mn": best_mb,
             "current_nb_mb": nb_mb}

    return stats


def collect_stats_epoch(stats, train_stats):
    if stats["tr_error"] != []:
        train_stats["tr_error_ep"].append(np.mean(stats["tr_error"]))
    if stats["vl_error"] != []:
        train_stats["vl_error_ep"].append(np.mean(stats["vl_error"]))
    if stats["tr_cost"] != []:
        train_stats["tr_cost_ep"].append(np.mean(stats["tr_cost"]))
    if stats["best_epoch"] is not None:
        train_stats["best_epoch"] = stats["best_epoch"]
    if stats["best_mn"] is not None:
        train_stats["best_mn"] = stats["best_mn"]
    train_stats["current_nb_mb"] = stats["current_nb_mb"]

    return train_stats


def split_mini_batch(x_train, y_train, share=True, borrow=True):
    x, y, xx, yy = None, None, None, None
    # split the supervised from the unsupervised data
    # Sup: (x, y)
    if len(x_train.shape) == 2:
        col_tr_x = x_train[:, 0]
    else:
        col_tr_x = x_train[:, 0, 0, 0]
    col_tr_y = y_train[:, 0]

    ind_sup = np.where(
        np.logical_and(np.invert(np.isnan(col_tr_x)),
                       np.invert(np.isnan(col_tr_y))))
    if len(ind_sup[0] > 0):
        x = x_train[ind_sup[0]]
        y = y_train[ind_sup[0]]

    # In: x_sup + x_nosup
    ind_xx = np.where(
        np.logical_and(np.invert(np.isnan(col_tr_x)), np.isnan(col_tr_y)))
    if len(ind_xx[0]) > 0:
        xx = x_train[ind_xx[0]]
    if x is not None:
        if xx is not None:
            xx = np.vstack((x, xx))
        else:
            xx = x_train[ind_sup[0]]
    # Out: y_sup + y_nonsup
    ind_yy = np.where(
        np.logical_and(np.isnan(col_tr_x), np.invert(np.isnan(col_tr_y))))
    if len(ind_yy[0] > 0):
        yy = y_train[ind_yy[0]]
    if y is not None:
        if yy is not None:
            yy = np.vstack((y, yy))
        else:
            yy = y_train[ind_sup[0]]

    if share:
        if xx is not None:
            c_batch_in = {
                'x': theano.shared(
                    np.asarray(xx, dtype=theano.config.floatX),
                    borrow=borrow)}
        else:
            c_batch_in = {'x': None}
        if yy is not None:
            c_batch_out = {
                'y': theano.shared(
                    np.asarray(yy, dtype=theano.config.floatX),
                    borrow=borrow)}
        else:
            c_batch_out = {'y': None}
        if x is not None:
            c_batch_sup = {
                'x': theano.shared(np.asarray(x, dtype=theano.config.floatX),
                                   borrow=borrow),
                'y': theano.shared(np.asarray(y, dtype=theano.config.floatX),
                                   borrow=borrow)}
        else:
            c_batch_sup = {'x': None, 'y': None}
    else:
        c_batch_in = {'x': xx}
        c_batch_out = {'y': yy}
        c_batch_sup = {'x': x, 'y': y}

    return c_batch_in, c_batch_out, c_batch_sup


def split_data_to_minibatchs_embed(data, batch_size, share=False, borrow=True,
                                   sharpe=False):
    """Split a dataset to minibatchs to:
        1. Control the case where the size of the set is not divided by the
            btahc_size.
        2. Allows to share batch by batch when the size of data is too big to
            fit in the GPU shared memory.

    data: dictionnary with two keys:
        x: x (numpy.ndarray) for the supervised data
            may contains rows with only numpy.nan as values.
        y: y (numpy.ndarray) for the superrvised data
            may contains rows with only numpy.nan as values.
    batch_size: the size of the batch
    sharpe: Boolean. If True, ignore the rest of data with size less than
        batch size.

    Returns:
        list_minibatchs: list
            a list of dic of mini-batchs, each one has the 3 keys: "in",
            "out", "sup".
    """
    nbr_samples = data['x'].shape[0]
    if batch_size > nbr_samples:
        batch_size = nbr_samples

    nbr_batchs = int(nbr_samples / batch_size)
    list_minibatchs = []
    for index in xrange(nbr_batchs):
        x_train = data['x'][index * batch_size:(index+1) * batch_size]
        y_train = data['y'][index * batch_size:(index+1) * batch_size]
        c_batch_in, c_batch_out, c_batch_sup = split_mini_batch(
            x_train.astype(theano.config.floatX),
            y_train.astype(theano.config.floatX), share=share, borrow=borrow)

        list_minibatchs.append(
            {'in': c_batch_in, 'out': c_batch_out, 'sup': c_batch_sup})
    if not sharpe:
        # in case some samples are left
        if (nbr_batchs * batch_size) < nbr_samples:
            x_train = data['x'][(index+1) * batch_size:]
            y_train = data['y'][(index+1) * batch_size:]
            c_batch_in, c_batch_out, c_batch_sup = split_mini_batch(
                x_train.astype(theano.config.floatX),
                y_train.astype(theano.config.floatX), share=share,
                borrow=borrow)
            list_minibatchs.append(
                {'in': c_batch_in, 'out': c_batch_out, 'sup': c_batch_sup})

    return list_minibatchs


def split_data_to_minibatchs_eval(data, batch_size, sharpe=False):
    """Split a dataset to minibatchs to:
        1. Control the case where the size of the set is not divided by the
            btahc_size.
        2. Allows to share batch by batch when the size of data is too big to
            fit in the GPU shared memory.

    data: dictionnary with two keys:
        x: x (numpy.ndarray) for the supervised data
            may contains rows with only numpy.nan as values.
        y: y (numpy.ndarray) for the superrvised data
            may contains rows with only numpy.nan as values.
    batch_size: the size of the batch

    Returns:
        list_minibatchs: list
            a list of dic of mini-batchs, each one has the 2 keys: "x", "y".
    """
    nbr_samples = data['x'].shape[0]
    if batch_size > nbr_samples:
        batch_size = nbr_samples

    nbr_batchs = int(nbr_samples / batch_size)
    list_minibatchs = []
    for index in xrange(nbr_batchs):
        x_train = data['x'][index * batch_size:(index+1) * batch_size]
        y_train = data['y'][index * batch_size:(index+1) * batch_size]
        list_minibatchs.append({'x': x_train.astype(theano.config.floatX),
                                'y': y_train.astype(theano.config.floatX)})
    if not sharpe:
        # in case some samples are left
        if (nbr_batchs * batch_size) < nbr_samples:
            x_train = data['x'][(index+1) * batch_size:]
            y_train = data['y'][(index+1) * batch_size:]
            list_minibatchs.append({'x': x_train.astype(theano.config.floatX),
                                    'y': y_train.astype(theano.config.floatX)})

    return list_minibatchs


def grab_train_data(x, y, nbr_sup, nbr_xx, nbr_yy):
    """Create a mixed train dataset.
    Input:
        x: numpy.ndarray. X matrix (all data)
        y: numpy.ndarray. Y matrix (all data)
        nbr_sup: int. The number of the supervised examples.
        nbr_xx: int. The number of the unsupervised x.
        nbr_yy: int. The number of the y without x.
    """
    assert nbr_sup + nbr_xx + nbr_yy <= x.shape[0]
    xtr = x[:nbr_sup]
    ytr = y[:nbr_sup]
    xxtr = x[nbr_sup:nbr_sup+nbr_xx]
    yytr = y[nbr_sup+nbr_xx: nbr_sup+nbr_xx+nbr_yy]
    # Mix them randomly.
    # Combine
    index_sup = range(nbr_sup)
    index_xx = range(nbr_sup, nbr_sup + nbr_xx)
    index_yy = range(nbr_sup + nbr_xx, nbr_sup + nbr_xx + nbr_yy)
    index = index_sup + index_xx + index_yy
    index_arr = np.array(index)
    for i in range(10000):
        np.random.shuffle(index_arr)
    if len(x.shape) == 2:
        mega_x = np.empty((len(index), x.shape[1]), dtype=np.float32)
    else:
        mega_x = np.empty((len(index), x.shape[1], x.shape[2], x.shape[3]),
                          dtype=np.float32)
    mega_y = np.empty((len(index), y.shape[1]), dtype=np.float32)
    for i, j in zip(index_arr, xrange(len(index))):
        if i in index_sup:
            mega_x[j] = xtr[i]
            mega_y[j] = ytr[i]
        elif i in index_xx:
            mega_x[j] = xxtr[i-nbr_sup]
            mega_y[j] = np.float32(np.nan)
        elif i in index_yy:
            mega_x[j] = np.float32(np.nan)
            mega_y[j] = yytr[i-nbr_sup-nbr_xx]
        else:
            raise ValueError("Something wrong.")

    return mega_x, mega_y


def reshape_data(y, s):
    """Reshape the output (debug).
    Input:
        y: numpy.ndarray. Matrix: row are samples.
        s: int. The new shape of the output samples (squared).
    """
    out = np.empty((y.shape[0], s*s), dtype=y.dtype)
    for i in xrange(y.shape[0]):
        im = imresize(y[i, :].reshape(28, 28), (s, s), 'bilinear').flatten()
        out[i, :] = im

    return out


def plot_x_y_yhat(x, y, y_hat, xsz, ysz, binz=False):
    """Plot x, y and y_hat side by side."""
    plt.close("all")
    f = plt.figure(figsize=(15, 10.8), dpi=300)
    gs = gridspec.GridSpec(1, 3)
    if binz:
        y_hat = (y_hat > 0.5) * 1.
    ims = [x, y, y_hat]
    tils = [
        "x:" + str(xsz) + "x" + str(xsz),
        "y:" + str(ysz) + "x" + str(ysz),
        "yhat:" + str(ysz) + "x" + str(ysz)]
    for n, ti in zip([0, 1, 2], tils):
        f.add_subplot(gs[n])
        if n == 0:
            plt.imshow(ims[n], cmap=cm.Greys_r)
        else:
            plt.imshow(ims[n], cmap=cm.Greys_r)
        plt.title(ti)

    return f


def plot_all_x_y_yhat(x, y, yhat, xsz, ysz, fd, binz=False):
    for i in xrange(x.shape[0]):
        if len(x[i].shape) <= 2:
            imx = x[i, :].reshape(xsz, xsz)
        else:
            imx = x[i].transpose((1, 2, 0))
        imy = y[i, :].reshape(ysz, ysz)
        imyhat = yhat[i, :].reshape(ysz, ysz)
        fig = plot_x_y_yhat(imx, imy, imyhat, xsz, ysz, binz)
        fig.savefig(fd+"/"+str(i)+".png", bbox_inches='tight')
        del fig


def add_noise(x):
    """Add random noise to images."""
    sz = x.shape
    mask = (x == 0) * 1.
    noise = np.random.rand(sz[0], sz[1])
    for i in range(x.shape[0]):
        noise[i] = ndimage.uniform_filter(noise[i].reshape(28, 28), size=3).flatten()
    noise = np.multiply(noise, mask)
    out = np.clip(x + noise, 0., 1.)
    return out


def add_cifar_10(x, x_cifar_10, sh=True):
    """Add cifar 10 as background."""
    sz = x.shape
    mask = (x == 0) * 1.
    # binarize cifar
    back = x_cifar_10.reshape(x_cifar_10.shape[0], 3, 32, 32).mean(1)
    back = back[:, 2:30, 2:30]  # take 28x28 from the center.
    back /= 255.
    back = back.astype(np.float32)

    # shuffle the index
    if sh:
        ind = np.random.randint(0, x_cifar_10.shape[0], sz[0])  # the index
        for i in range(10):
            np.random.shuffle(ind)
    else:
        # used only to plot images for paper.
        assert x_cifar_10.shape[0] == sz[0]
        ind = np.arange(0,  sz[0])  # the index
    back_sh = back[ind]
    back_sh = back_sh.reshape(back_sh.shape[0], -1)
    back_ready = np.multiply(back_sh, mask)
    out = np.clip(x + back_ready, 0., 1.)
    return out


def plot_representations(X, y, title):
    """Plot distributions and thier labels."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    f = plt.figure(figsize=(15, 10.8), dpi=300)
#    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

#    if hasattr(offsetbox, 'AnnotationBbox'):
#        # only print thumbnails with matplotlib > 1.0
#        shown_images = np.array([[1., 1.]])  # just something big
#        for i in range(digits.data.shape[0]):
#            dist = np.sum((X[i] - shown_images) ** 2, 1)
#            if np.min(dist) < 4e-3:
#                # don't show points that are too close
#                continue
#            shown_images = np.r_[shown_images, [X[i]]]
#            imagebox = offsetbox.AnnotationBbox(
#                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
#                X[i])
#            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    return f

if __name__ == "__main__":
    from keras import regularizers
    W_regularizer = regularizers.l2(1e-3)
    activity_regularizer = regularizers.activity_l2(0.0001)
    cnn = FeatureExtractorCNN("alexnet",
                              weights_path="../inout/weights/alexnet_weights.h5",
                              trainable=True, W_regularizer=W_regularizer,
                              activity_regularizer=activity_regularizer,
                              trained=False, dense=False,
                              just_the_features=True)
