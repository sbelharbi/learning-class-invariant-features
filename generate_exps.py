import yaml
import os


def get_name_exp_from_yaml(d):
    name = ""
    name = str(d["run"]) + "_"
    name += str(d["nbr_sup"]) + "_"
    name += str(len(d["h_ind"]) - 1) + "_"
    name += "_".join([str(int(k)) for k in d["h_ind"]]) + "_"
    name += str(d["repet"]) + "_"
    name += str(d["hint"]) + "_"
    name += str(d["norm_gsup"]) + "_"
    name += str(d["norm_gh"]) + "_"
    name += str(d["debug_code"]) + "_"
    name += str(d["use_unsupervised"])

    return name


def save_file(exp, rep, max_rep):
    # grad normalization
    # conf_norm = [(1, 0), (0, 1), (1, 1)]
    conf_norm = [(0, 0)]
    for c in conf_norm:
        exp["norm_gh"] = bool(c[0])
        exp["norm_gsup"] = bool(c[1])
        if rep == max_rep - 1:
            exp["debug_code"] = True
        print "Just forces debuge to TRUE *********"
        exp["debug_code"] = False
        name = get_name_exp_from_yaml(exp)
        with open(fold_exps+"/"+name+".yaml", "w") as fyaml:
            yaml.dump(exp, fyaml)
            f.write("python " + runner + " " + name + ".yaml \n")
# Default
nbr_layers = 3
use_unsupervised = False
exp = {"debug_code": False,
       "nbr_sup": 1000,
       "run": 45,
       "h_ind": [False for i in range(nbr_layers+1)],
       "use_batch_normalization": [False for i in range(nbr_layers+1)],
       "corrupt_input_l": 0.,
       "start_corrupting": 0,
       "use_sparsity": False,
       "use_sparsity_in_pred": False,
       "max_epochs": 400,
       "hint": False,
       "extreme_random": True,
       "norm_gsup": False,
       "norm_gh": False,
       "repet": 0,
       "use_unsupervised": use_unsupervised,
       "h_w": 1.,
       "start_hint": 5
       }
nbr_sup_ = [1000, 3000, 5000, 50000]
h_w_vls = [.0, .0, .0, .0]
start_hint_vl = [2, 2, 1, 1]
run = 0
fold_exps = "config_yaml"
bash_name = "job0.sh"
f = open(bash_name, "w")
f.write("#!/usr/bin/env bash \n")
runner = "train3_new_dup.py"
max_rep = 7
for nbr, h_w, start_hint in zip(nbr_sup_, h_w_vls, start_hint_vl):
    for rep in range(max_rep):
        print rep
        exp["nbr_sup"] = nbr
        # we need one run for an MLP without hint.
        if rep == 0:
            exp["debug_code"] = False
        else:
            exp["debug_code"] = False
        exp["h_ind"] = [False for k in range(nbr_layers+1)]
        exp["use_batch_normalization"] = [False for k in range(nbr_layers+1)]
        exp["use_batch_normalization"][-2] = False
        exp["hint"] = False
        exp["run"] = run
        exp["repet"] = rep
        exp["norm_gh"] = False
        exp["norm_gsup"] = False
        exp["max_epochs"] = 2000
        exp["start_hint"] = 0
        exp["h_w"] = h_w
        name = get_name_exp_from_yaml(exp)
        with open(fold_exps+"/"+name+".yaml", "w") as fyaml:
            yaml.dump(exp, fyaml)
            f.write("python " + runner + " " + name + ".yaml \n")

        exp["max_epochs"] = 400
        exp["debug_code"] = False
        # ******* Train inly the layer before the output.
        exp["h_ind"] = [False for k in range(nbr_layers+1)]
        exp["h_ind"][-2] = True
        exp["use_batch_normalization"] = [False for k in range(nbr_layers+1)]
        exp["use_batch_normalization"][-2] = True
        exp["hint"] = True
        exp["run"] = run
        exp["repet"] = rep
        exp["start_hint"] = start_hint
        save_file(exp, rep, max_rep)
        continue
        # *****
        # Exclusive layers
        for i in range(nbr_layers+1):
            exp["h_ind"] = [False for k in range(nbr_layers+1)]
            exp["h_ind"][i] = True
            exp["hint"] = True
            exp["run"] = run
            exp["repet"] = rep
            save_file(exp, rep, max_rep)

        # From input to output
#        exp["h_ind"] = [False for k in range(nbr_layers+1)]
#        exp["h_ind"][0] = True
#        for kk in range(1, nbr_layers+1):
#            exp["h_ind"][kk] = True
#            save_file(exp, rep, max_rep)
        # From output to input
        exp["h_ind"] = [False for k in range(nbr_layers+1)]
        exp["h_ind"][-1] = True
        for kk in range(-2, -(nbr_layers+2), -1):
            exp["h_ind"][kk] = True
            save_file(exp, rep, max_rep)
f.close()
os.system("chmod +x " + bash_name)
