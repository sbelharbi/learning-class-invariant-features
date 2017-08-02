import yaml
import os
from operator import eq
import numpy as np
import sys


def get_vl_tst(f):
    with open(f, 'r') as ff:
        cont = ff.readlines()
    cont = [x.strip() for x in cont]
    # vl:
    vl = float(cont[1].split(":")[1].split(" ")[0])
    tst = float(cont[2].split(":")[1].split(" ")[0])
    return [vl, tst]


def get_all_yamls_perfs(folder, h_ind=[1, 1, 1, 1], hint=[True, False],
                        norm_gh=True,
                        norm_gsup=True, nbr_sup=1000):
    """Collect all the yaml files and the performance files."""
    h_ind_n = [str(i) for i in h_ind]
    h_ind = [bool(k) for k in h_ind]
    path_to_exps = folder
    list_exps = next(os.walk(path_to_exps))[1]
    list_exps = [e for e in list_exps if e.startswith("hint") or
                 e.startswith("no")]
    list_exps = [path_to_exps + e for e in list_exps]
    # Start filtering
    filtered_list = []
    list_start_hint = []
    for d in list_exps:
        # Get the yaml file
        for file in os.listdir(d):
            if file.endswith(".yaml"):
                yaml_file = os.path.join(d, file)
                # print yaml_file
        # Satrt filtering ...
        # Read the yaml file
        with open(yaml_file, 'r') as y:
            yaml_cont = yaml.load(y)
        if yaml_cont["hint"] not in hint:
            continue
        if yaml_cont["norm_gh"] != norm_gh:
            continue
        if yaml_cont["norm_gsup"] != norm_gsup:
            continue
        if yaml_cont["nbr_sup"] != nbr_sup:
            continue
        if not all(map(eq, yaml_cont["h_ind"], h_ind)):
            continue
        # Get the per file.
        for file in os.listdir(d):
            if file.endswith(".txt"):
                perf_file = os.path.join(d, file)
                filtered_list.append(perf_file)
        list_start_hint.append(yaml_cont["start_hint"])
    # No that you are done collecting the appropriate files.
    # COmpute the mean+-std
    vl, tst = [], []
    for file in filtered_list:
        [v, t] = get_vl_tst(file)
        vl.append(v)
        tst.append(t)
    # remove the largest and smallest value (test error)
    comb = zip(vl, tst, list_start_hint)
    sorted_comb = sorted(comb, key=lambda tup: tup[1])
    print "(vl, tst, start_hint)", len(comb)
    for el in sorted_comb:
        print el
    # remove the best and the worst.
    sorted_comb.pop(0)
    sorted_comb.pop(-1)
    vl, tst, list_start_hint = zip(*sorted_comb)
    # back to original lists.
    m_vl = np.mean(vl)
    std_vl = np.std(vl)
    m_tst = np.mean(tst)
    std_tst = np.std(tst)
    print str(len(filtered_list)), "_".join(h_ind_n), " norm_gh:",\
        str(norm_gh),\
        " norm_gsup:", str(norm_gsup),\
        " vl:", str(m_vl), "+-", str(std_vl), " tst:", str(m_tst), "+-",\
        str(std_tst), "\n"

inds = [[0, 0, 1, 0]]
norm_gh = False
norm_gsup = False
hint = [True, False]
nbr_sup = int(sys.argv[2])
path_exps = str(sys.argv[1])

for e in inds:
    get_all_yamls_perfs(path_exps, h_ind=e, hint=hint, norm_gh=norm_gh,
                        norm_gsup=norm_gsup, nbr_sup=nbr_sup)
