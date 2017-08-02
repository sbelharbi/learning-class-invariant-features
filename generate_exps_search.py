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
    name += str(d["use_unsupervised"]) + "_"
    name += str(d["start_hint"])

    return name


def save_file(exp, rep, max_rep):
    # grad normalization
    # conf_norm = [(1, 0), (0, 1), (1, 1)]
    conf_norm = [(0, 0)]
    for c in conf_norm:
        exp["norm_gh"] = bool(c[0])
        exp["norm_gsup"] = bool(c[1])
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
gpu = "p100.sl"
nbr_sup_ = 1000
h_w_vls = .0
# start_hint_vl = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
start_hint_vl = [10, 10, 10, 10, 10, 10, 10]
run = 0
fold_exps = "config_yaml"
folder_jobs = "jobs"
bash_name = "submit.sh"
f = open(bash_name, "w")
f.write("#!/usr/bin/env bash \n")
runner = "train3_new_dup.py"
max_rep = 7
flags = "THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 "
rep = 0
fgpu = open(gpu, "r")
gpu_cont = fgpu.read()

for start_hint in start_hint_vl:
    exp["nbr_sup"] = nbr_sup_
    exp["run"] = run
    exp["norm_gh"] = False
    exp["norm_gsup"] = False
    exp["start_hint"] = 0
    exp["h_w"] = h_w_vls

    exp["max_epochs"] = 400
    exp["debug_code"] = False
    # ******* Train inly the layer before the output.
    exp["h_ind"] = [False for k in range(nbr_layers+1)]
    exp["h_ind"][-2] = True
    exp["use_batch_normalization"] = [False for k in range(nbr_layers+1)]
    exp["use_batch_normalization"][-2] = False
    exp["hint"] = True
    exp["run"] = run
    exp["repet"] = rep
    exp["start_hint"] = start_hint
    name = get_name_exp_from_yaml(exp)
    with open(fold_exps+"/"+name+".yaml", "w") as fyaml:
        yaml.dump(exp, fyaml)
    name_job = str(start_hint) + "_" + str(nbr_sup_) + "_" + str(rep) + ".sl"
    with open(folder_jobs + "/" + name_job, "w") as fjob:
        fjob.write(gpu_cont + "\n")
        fjob.write(flags + " python " + runner + " " + name + ".yaml \n")
        # save_file(exp, rep, max_rep)
    f.write("sbatch ./" + folder_jobs + "/" + name_job + " \n")
    rep += 1

f.close()
fgpu.close()
os.system("chmod +x " + bash_name)
