import cPickle as pkl
import numpy as np
import theano.tensor as T
import os
import sys
import datetime as DT
import shutil
import inspect
import theano
import warnings
import yaml

from tools import ModelMLP
from tools import NonLinearity
from tools import split_data_to_minibatchs_eval
from tools import sharedX_value
from tools import theano_fns
from tools import theano_fns_double_up
from learning_rule import AdaDelta
from learning_rule import RMSProp
from learning_rule import Momentum
from tools import evaluate_model
from tools import collect_stats_epoch
from tools import plot_stats
from tools import train_one_epoch
from tools import train_one_epoch_alter
from tools import to_categorical
from tools import plot_classes
from tools import chunks
from tools import plot_penalty_vl
from tools import plot_debug_grad
from tools import plot_debug_ratio_grad
from sklearn import manifold
from tools import plot_representations


# Parse the yaml config.
config_path = "./config_yaml/"
with open(config_path + sys.argv[1], 'r') as fy:
    config_exp = yaml.load(fy)

x_classes = 10

debug_code = config_exp["debug_code"]

if debug_code:
    warnings.warn("YOU ARE IN DEBUG MODE! YOUR CODE WILL TAKE MORE TIME!!!!!")


def standerize(d, mu=None, sigma=None):
    if mu is None:
        mu = np.mean(d, axis=0)
        sigma = np.std(d, axis=0)
    if sigma.nonzero()[0].shape[0] == 0:
        raise Exception("std found to be zero!!!!")
    norm_d = (d - mu) / sigma

    return norm_d, mu, sigma


def get_inter_output(model, l_tst, testx_sh):
    i_x_vl = T.lvector("ixtst")

    eval_fn_tst = theano.function(
        [i_x_vl],
        [l.output for l in model.layers],
        givens={model.x: testx_sh[i_x_vl]})
    output_v = [
        eval_fn_tst(np.array(l_tst[kkk])) for kkk in range(len(l_tst))]
    nbr_layers = len(output_v[0])

    l_val = []
    for l in range(nbr_layers):
        tmp = None
        for k in output_v:
            if tmp is None:
                tmp = k[l]
            else:
                tmp = np.vstack((tmp, k[l]))
        l_val.append(tmp)

    return l_val


# create_tr_vl_ts_cb("data/2d")

# def knn1(model, l_tst, testx_sh, l_tr, trainx_sh):

# create_tr_vl_ts_nc("data/nestedcircle", 50000)

# DATA MNIST
# =============================================================================
path_data = "data/mnist.pkl"
f = open(path_data, 'r')
train, valid, test = pkl.load(f)
trainx, trainy = train[0], train[1]
validx, validy = valid[0], valid[1]
testx, testy = test[0], test[1]

# How much to take for training?
nbr_sup = config_exp["nbr_sup"]
run = config_exp["run"]
print "RUN:", run
print "SUP: ", nbr_sup
trainx, trainy = trainx[:nbr_sup], trainy[:nbr_sup]
# Prepare the pre-shuffling
if not os.path.exists("data/" + str(nbr_sup)):
    os.makedirs("data/" + str(nbr_sup))
trainx_tmp = trainx
trainy_tmp = trainy

print trainy.shape
big_mtx = np.hstack((trainx_tmp, trainy_tmp.reshape(trainy_tmp.size, 1)))
print "Going to shuffle the train data. It takes some time ..."
period = 200
i = 0
#for k in xrange(5000):
#    np.random.shuffle(big_mtx)
#    if k % period == 0:
#        trainx_tmp2 = big_mtx[:, 0:trainx_tmp.shape[1]]
#        trainy_tmp2 = big_mtx[:, -1]
#        stuff = {"x": trainx_tmp2, "y": trainy_tmp2}
#        print k
#        with open("data/"+str(nbr_sup) + "/" + str(i) + ".pkl", 'w') as f:
#            pkl.dump(stuff, f, protocol=pkl.HIGHEST_PROTOCOL)
#        i += 1

#with open("data/"+str(nbr_sup) + "/0.pkl") as f:
#    stuff = pkl.load(f)
#    trainx, trainy = stuff["x"], stuff["y"]
# share over gpu: we can store the whole mnist over the gpu.
# Train
trainx_sh = theano.shared(trainx.astype(theano.config.floatX),
                          name="trainx", borrow=True)
trainlabels_sh = theano.shared(trainy.astype(theano.config.floatX),
                               name="trainlabels", borrow=True)
trainy_sh = theano.shared(to_categorical(trainy, x_classes).astype(
    theano.config.floatX),  name="trainy", borrow=True)

# valid
validx_sh = theano.shared(validx.astype(theano.config.floatX),
                          name="validx", borrow=True)
validlabels_sh = theano.shared(validy.astype(theano.config.floatX),
                               name="validlabels", borrow=True)
#
input = T.fmatrix("x")
input1 = T.fmatrix("x1")
input2 = T.fmatrix("x2")
rng = np.random.RandomState(23455)
# Architecture
nhid_l0 = 1200
nhid_l1 = 1200
nhid_l2 = 200

nbr_classes = x_classes
use_batch_normalization = config_exp["use_batch_normalization"]
h_ind = config_exp["h_ind"]
h_ind = [int(tt) for tt in h_ind]

assert len(h_ind) == 4
h0, h1, h2, h3, h4, h5, h6, h7, h8 = None, None, None, None, None, None, None,\
    None, None
l_v = []
for xx in h_ind:
    print xx
    if int(xx) == 1:
        l_v.append(True)
    elif int(xx) == 0:
        l_v.append(False)
    else:
        raise ValueError("Error in applying hint: 0/1")

hint_type = "l2sum"  # "l1mean"
print l_v
corrupt_input_l = config_exp["corrupt_input_l"]
if corrupt_input_l != 0.:
    warnings.warn(
        "YOU ASKED TO USE DENOISING PROCESS OVER THE INPUTS OF THE FIRST LAYER"
        )
    if not config_exp["hint"]:
        raise ValueError(
            "You asked for densoing process but you are not using the penalty")
start_corrupting = config_exp["start_corrupting"]
warnings.warn(
    "CORRUPTION WILL START AFTER:" + str(start_corrupting) + " epochs!!!!!!")
use_sparsity = config_exp["use_sparsity"]
use_sparsity_in_pred = config_exp["use_sparsity_in_pred"]
print "Use sparsity: ", use_sparsity
print "Use sparsity in pred:", use_sparsity_in_pred
use_unsupervised = config_exp["use_unsupervised"]

layer0 = {
    "rng": rng,
    "n_in": trainx.shape[1],
    "n_out": nhid_l0,
    "W": None,
    "b": None,
    "activation": NonLinearity.SIGMOID,
    "hint": hint_type,
    "use_hint": l_v[0],
    "intended_to_be_corrupted": True,
    "corrupt_input_l": corrupt_input_l,
    "use_sparsity": use_sparsity,
    "use_sparsity_in_pred": use_sparsity_in_pred,
    "use_unsupervised": use_unsupervised,
    "use_batch_normalization": use_batch_normalization[0]
    }

layer1 = {
    "rng": rng,
    "n_in": nhid_l0,
    "n_out": nhid_l1,
    "W": None,
    "b": None,
    "activation": NonLinearity.SIGMOID,
    "hint": hint_type,
    "use_hint": l_v[1],
    "use_sparsity": use_sparsity,
    "use_sparsity_in_pred": use_sparsity_in_pred,
    "use_unsupervised": use_unsupervised,
    "use_batch_normalization": use_batch_normalization[1]
    }

layer2 = {
    "rng": rng,
    "n_in": nhid_l1,
    "n_out": nhid_l2,
    "W": None,
    "b": None,
    "activation": NonLinearity.SIGMOID,
    "hint": hint_type,
    "use_hint": l_v[2],
    "use_sparsity": use_sparsity,
    "use_sparsity_in_pred": use_sparsity_in_pred,
    "use_unsupervised": use_unsupervised,
    "use_batch_normalization": use_batch_normalization[2]
    }

#layer3 = {
#    "rng": rng,
#    "n_in": nhid_l2,
#    "n_out": nhid_l3,
#    "W": None,
#    "b": None,
#    "activation": NonLinearity.SIGMOID,
#    "hint": l_v[3]
#    }
#
#layer4 = {
#    "rng": rng,
#    "n_in": nhid_l3,
#    "n_out": nhid_l4,
#    "W": None,
#    "b": None,
#    "activation": NonLinearity.SIGMOID,
#    "hint": l_v[4]
#    }
#
#layer5 = {
#    "rng": rng,
#    "n_in": nhid_l4,
#    "n_out": nhid_l5,
#    "W": None,
#    "b": None,
#    "activation": NonLinearity.SIGMOID,
#    "hint": l_v[5]
#    }
#
#layer6 = {
#    "rng": rng,
#    "n_in": nhid_l5,
#    "n_out": nhid_l6,
#    "W": None,
#    "b": None,
#    "activation": NonLinearity.SIGMOID,
#    "hint": l_v[6]
#    }
#
#layer7 = {
#    "rng": rng,
#    "n_in": nhid_l6,
#    "n_out": nhid_l7,
#    "W": None,
#    "b": None,
#    "activation": NonLinearity.SIGMOID,
#    "hint": l_v[7]
#    }

output_layer = {
    "rng": rng,
    "n_in": nhid_l2,
    "n_out": nbr_classes,
    "W": None,
    "b": None,
    "activation": NonLinearity.SOFTMAX,
    "hint": hint_type,
    "use_hint": l_v[3],
    "use_sparsity": False,
    "use_sparsity_in_pred": False,
    "use_unsupervised": use_unsupervised,
    "use_batch_normalization": use_batch_normalization[3]
    }
layers = [layer0, layer1, layer2, output_layer]
l1, l2 = 0., 0.
reg_bias = True
margin = sharedX_value(1., name="margin")
similair = theano.shared(np.array([0, 1], dtype=theano.config.floatX),
                         name="sim")
model = ModelMLP(layers, input, input1, input2,
                 trainx_sh, trainlabels_sh, trainy_sh,
                 validx_sh, validlabels_sh, margin, similair,
                 l1_reg=l1, l2_reg=l2,
                 reg_bias=reg_bias)

size_model = str(trainx.shape[1]) +\
    '_'.join([str(l["n_in"]) for l in layers]) + "_" + str(nbr_classes)
path_model_init_params = "init_params/" + size_model + '_' +\
    str(config_exp["repet"]) + ".pkl"
if not os.path.isfile(path_model_init_params):
    model.save_params(path_model_init_params, catched=False)
else:
    model.set_params_vals(path_model_init_params)

train_batch_size = 100
valid_batch_size = 1000

max_epochs = config_exp["max_epochs"]
lr_vl = 1e-7
lr = sharedX_value(lr_vl, name="lr")
h_w = sharedX_value(config_exp["h_w"], name="hw")
s_w = sharedX_value(1., name="sw")
unsup_w = sharedX_value(1., name="unsw")
lambda_sparsity = sharedX_value(0., name="l_sparsity")

# Compile functions: train/valid
updater_sup = AdaDelta(decay=0.95)
updater_hint = AdaDelta(decay=0.95)
updater_unsup = AdaDelta(decay=0.95)
updater = {"sup": updater_sup, 'hint': updater_hint, "unsup": updater_unsup}

# updater = Momentum(0.9, nesterov_momentum=False, imagenet=False,
#                   imagenetDecay=5e-4, max_colm_norm=False)

hint = config_exp["hint"]
# "hint", "noHint"
if hint:
    tag = "hint"
else:
    tag = "noHint"

norm_gsup = config_exp["norm_gsup"]
norm_gh = config_exp["norm_gh"]
fns = theano_fns_double_up(
    model, learning_rate=lr,
    h_w=h_w, s_w=s_w, unsup_w=unsup_w, lambda_sparsity=lambda_sparsity,
    updater=updater, tag=tag,
    max_colm_norm=False, max_norm=15.0,
    norm_gsup=norm_gsup, norm_gh=norm_gh)

eval_fn, eval_fn_tr = fns["eval_fn"], fns["eval_fn_tr"]
# Things to track during training: epoch and minibatch
train_stats = {"tr_error_ep": [], "vl_error_ep": [], "tr_cost_ep": [],
               "tr_error_mn": [], "vl_error_mn": [], "tr_cost_mn": [],
               "current_nb_mb": 0, "best_epoch": 0, "best_mn": 0}

names = []
for l, i in zip(layers, range(len(layers))):
    if l["hint"] is not None:
        names.append(i)
debug = {"grad_sup": [], "grad_hint": [], "penalty": [], "names": names}
# Eval before start training
l_vl = chunks(range(validx.shape[0]), valid_batch_size)
l_tr = chunks(range(trainx.shape[0]), valid_batch_size)
vl_err_start = np.mean(
    [eval_fn(np.array(l_vl[kk])) for kk in range(len(l_vl))])
tr_err_start = np.mean(
    [eval_fn_tr(np.array(l_tr[kk])) for kk in range(len(l_tr))])
print vl_err_start, tr_err_start

# Exp stamp
time_exp = DT.datetime.now().strftime('%m_%d_%Y_%H_%M_%s')
tag_text = "_".join([str(l["hint"]) for l in layers])
h_exp = "_".join([str(e) for e in h_ind])
fold_exp = "exps/" + tag + "_" + str(nbr_sup) + "_" + h_exp + "_" +\
    size_model + "_" + time_exp
if not os.path.exists(fold_exp):
    os.makedirs(fold_exp)

shutil.copy(inspect.stack()[0][1], fold_exp)
shutil.copy(config_path+sys.argv[1], fold_exp)

# Start training
stop, i = False, 0
div = any([l["hint"] is "contrastive" for l in layers])
shuffle_period = 1   # epochs
do_shuffle = True
extreme_random = config_exp["extreme_random"]
if extreme_random:
    print "Extreme randomness."
else:
    print "Same shuffle."
kk = 1

# TEST BEFORE START TRAINING
testx_sh = theano.shared(testx.astype(theano.config.floatX),
                         name="testx", borrow=True)
testlabels_sh = theano.shared(testy.astype(theano.config.floatX),
                              name="testlabels", borrow=True)

i_x_vl = T.lvector("ixtst")
y_vl = T.vector("y")
error = T.mean(T.neq(T.argmax(model.output, axis=1), y_vl))

output_fn_test = [error, model.output, model.layers[-2].output]

eval_fn_tst = theano.function(
    [i_x_vl], output_fn_test,
    givens={model.x: testx_sh[i_x_vl],
            y_vl: testlabels_sh[i_x_vl]})
l_tst = chunks(range(testx.shape[0]), valid_batch_size)
test_error_l = [eval_fn_tst(np.array(l_tst[kkk])) for kkk in range(len(l_tst))]
print test_error_l[0][0]

test_error = np.mean([l[0] for l in test_error_l])
print "Test error:", test_error
prediction = None
for l in test_error_l:
    if prediction is None:
        prediction = l[1]
    else:
        prediction = np.vstack((prediction, l[1]))


with open(fold_exp+"/pred_before.pkl", "w") as fp:
    pkl.dump({"y": testy, "pred": prediction}, fp)

best_vl_error = np.finfo(np.float).max
start_hint_epoch = config_exp["start_hint"]

while i < max_epochs:
    if i >= start_corrupting:
        warnings.warn(
            "SETTING THE CORRUPTION LEVEL TO:" + str(corrupt_input_l))
        model.layers[0].corrupt_input_l.set_value(
            np.cast[theano.config.floatX](corrupt_input_l))
    else:
        warnings.warn("SETTING THE CORRUPTION LEVEL TO: 0")
        model.layers[0].corrupt_input_l.set_value(
              np.cast[theano.config.floatX](0.))
    stop = (i == max_epochs - 1)
    tx = DT.datetime.now()
    stats = train_one_epoch_alter(
        model, fns, i, fold_exp, train_stats, vl_err_start, tag,
        train_batch_size, l_vl, l_tr, div, stop=stop,
        debug=debug, debug_code=debug_code, h_w=h_w)
    txx = DT.datetime.now()
    print "CORRUPTION LEVEL VALUE: " +\
        str(model.layers[0].corrupt_input_l.get_value())
    print "One epoch", DT.datetime.now() - tx
    train_stats = collect_stats_epoch(stats, train_stats)
    if (i % 100 == 0 or stop) and debug_code:
        plot_debug_grad(debug, tag_text, fold_exp, "sup")
        plot_penalty_vl(debug, tag_text, fold_exp)
        if tag == "hint":
            plot_debug_grad(debug, tag_text, fold_exp, "hint")
            plot_debug_ratio_grad(debug, fold_exp, "h/s")
            plot_debug_ratio_grad(debug, fold_exp, "s/h")

    if stop:
        plot_stats(train_stats, "ep", fold_exp, tag)
        with open(fold_exp + "/train_stats.pkl", 'w') as f_ts:
                pkl.dump(train_stats, f_ts)
        with open(fold_exp + "/train_debug.pkl", 'w') as f_ts:
                pkl.dump(debug, f_ts)
    i += 1
    # shuffle the data

    print "Going to shuffle the train data."

    if do_shuffle and i % shuffle_period == 0 and not stop:
        if extreme_random:
            trainx_tmp = model.trainx_sh.get_value()
            trainy_tmp = model.trainlabels_sh.get_value()
            big_mtx = np.hstack(
                (trainx_tmp, trainy_tmp.reshape(trainy_tmp.size, 1)))
            for k in xrange(5):
                np.random.shuffle(big_mtx)
            trainx_tmp = big_mtx[:, 0:trainx_tmp.shape[1]]
            trainy_tmp = big_mtx[:, -1]
        else:
            with open("data/"+str(nbr_sup) + "/" + str(kk) + ".pkl") as f:
                stuff = pkl.load(f)
                trainx_tmp, trainy_tmp = stuff["x"], stuff["y"]
        model.trainlabels_sh.set_value(trainy_tmp.astype(theano.config.floatX))
        model.trainy_sh.set_value(
            to_categorical(
                trainy_tmp, nbr_classes).astype(theano.config.floatX))
        model.trainx_sh.set_value(trainx_tmp.astype(theano.config.floatX))
        kk += 1
        if kk > 240:
            kk = 0
        print "Finished loading shuffled data. Updated the train set on GPU."
    del stats
    print "This part took:", DT.datetime.now() - txx

    print "MIN VALID ", np.min(train_stats["vl_error_mn"]), " *********"
#    # If there was no improvement...
    if (i > start_hint_epoch) and hint:
        # new_v = min([1., h_w.get_value() + 0.01])
        new_v = 1.
        h_w.set_value(np.cast[theano.config.floatX](new_v))
#        print "NO IMPROV. PUSHING THE NET..............................."
#    best_vl_error = np.min(train_stats["vl_error_mn"])
    # Update the importance of the hint
#    if i >= 1:
#        # new_v = min([1., h_w.get_value() + 0.1])
#        h_w.set_value(np.cast[theano.config.floatX](1.))


# Perform the test
# Set the model's param to the best catched ones
model.set_model_to_catched_params()
# share test data

test_error_l = [eval_fn_tst(np.array(l_tst[kkk])) for kkk in range(len(l_tst))]
train_error_l = [eval_fn_tst(np.array(l_tr[kkk])) for kkk in range(len(l_tr))]

test_error = np.mean([l[0] for l in test_error_l])
print "Test error:", test_error
# Train

# Test
# last hidden layer representations.
with open(fold_exp+"/last_hidden_rep_test.pkl", "w") as fhr:
    stuff_hrep_tst = None
    for k in test_error_l:
        if stuff_hrep_tst is None:
            stuff_hrep_tst = l[2]
        else:
            stuff_hrep_tst = np.vstack((stuff_hrep_tst, l[2]))

    stuff_hrep_tr = None
    for k in train_error_l:
        if stuff_hrep_tr is None:
            stuff_hrep_tr = l[2]
        else:
            stuff_hrep_tr = np.vstack((stuff_hrep_tr, l[2]))
    pkl.dump(
        {"x_hint_repr_tst": stuff_hrep_tst, "y_tst": testy, "ximg_tst": testx,
         "x_hint_repr_tr": stuff_hrep_tr, "y_tr": trainy, "ximg_tr": trainx},
        fhr)
    # plot t-SNE of the opriginal images
    tx0 = DT.datetime.now()
    tsne_original = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_original = tsne_original.fit_transform(testx)
    fig_tsne_org = plot_representations(
        X_tsne_original, testy, "t-SNE embedding of mnist original images.")
    fig_tsne_org.savefig(fold_exp+"/original_rep_test.eps", format='eps',
                         dpi=1200, bbox_inches='tight')
    print "t-SNE of original images took:", DT.datetime.now() - tx0
    # plot t-SNE of the prediction
    tx0 = DT.datetime.now()
    tsne_lasthidden_rep = manifold.TSNE(n_components=2, init='pca',
                                        random_state=0)
    X_tsne_lhrep = tsne_original.fit_transform(stuff_hrep_tst)
    fig_tsne_lhrep = plot_representations(
        X_tsne_lhrep, testy,
        "t-SNE embedding of the last hidden representation of the MLP" +
        "applied over mnist.")
    fig_tsne_lhrep.savefig(fold_exp+"/lasth_rep_mlp_test.eps", format='eps',
                           dpi=1200, bbox_inches='tight')
    print "t-SNE of hidden representation took:", DT.datetime.now() - tx0
prediction = None
for l in test_error_l:
    if prediction is None:
        prediction = l[1]
    else:
        prediction = np.vstack((prediction, l[1]))


with open(fold_exp+"/pred_after.pkl", "w") as fp:
    pkl.dump({"y": testy, "pred": prediction}, fp)

##############################################################################
# GET INTERMEDIATE VALUE AND PLOT THEM. POSSIBLE ONLY WHEN THE INTERMEDIATE
# VALUES ARE 2D.
# inter_vl = get_inter_output(model, l_tst, testx_sh)
# plot the intermediate values
# ll = 0
# for vi in inter_vl:
#    fig = plot_classes(
#        testy_int, vi, "", test_error,
#        "pred. 2D: mnist 1/7. layer" + str(ll))
#    fig.savefig(
#        fold_exp + "/predinterlayer" + str(ll) + ".png", bbox_inches='tight')
#    ll += 1

###############################################################################

#    fig_scatter = plot_classes(y=testy_int, cord=prediction, names=cs,
#                               test_error=test_error, message="AFTER train")
#    fig_scatter.savefig(fold_exp+"/pred_after.png", bbox_inches='tight')
# save min valid
vl_pathfile = "exps/" + "run_" + str(run) + "_sup_" + str(nbr_sup) + "_" +\
    h_exp + "_c_l_" + str(corrupt_input_l) + "_start_at_" +\
    str(start_corrupting) + "_debug_" + str(debug_code) +\
    "_use_sparse_" + str(use_sparsity) + "_use_spar_pred_" +\
    str(use_sparsity_in_pred) + "_" + "norm_" + str(norm_gsup) + "_" +\
    str(norm_gh) + "_" + time_exp + ".txt"
with open(vl_pathfile, 'w') as f:
    f.write("Exp. folder: " + fold_exp + "\n")
    f.write(
        "valid error:" + str(
            np.min(train_stats["vl_error_mn"]) * 100.) + " % \n")
    f.write("Test error:" + str(test_error * 100.) + " % \n")
shutil.copy(vl_pathfile, fold_exp)
