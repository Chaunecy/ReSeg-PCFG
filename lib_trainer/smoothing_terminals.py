import os
from collections import defaultdict


def smoothing_terminals(rule: str):
    for folder in ["Alpha", "Digits", "Others"]:
        for length in [4, 5, 6, 7]:
            fd_origin = open(os.path.join(rule, folder, f"{length}.txt"), "r")
            origin_counter = defaultdict(float)
            for line in fd_origin:
                line = line.strip("\r\n")
                pwd, prob = line.split("\t")
                origin_counter[pwd] = float(prob)
            threshold_prob = 5 * min(origin_counter.values())
            fd_origin.close()
            fd_left = open(os.path.join(rule, folder, f"{length / 2}.txt"), "r")
            opt_len = open(os.path.join(rule, folder, f"opt-{length}.txt"), "w")
            for line in fd_left:
                pwd, prob = line.strip("\r\n").split("\t")
                prob = float(prob)
                if prob < threshold_prob:
                    continue
                fd_right = open(os.path.join(rule, folder, f"{length - length / 2}.txt"))
                for line_r in fd_right:
                    pwd_r, prob_r = line_r.strip("\r\n").split("\t")
                    prob_r = float(prob_r)
                    final_prob = prob_r * prob
                    final_pwd = f"{pwd}{pwd_r}"
                    if final_prob > threshold_prob and final_pwd not in origin_counter:
                        opt_len.write(f"{pwd}\t{final_prob}\n")

            pass
        pass
    pass
