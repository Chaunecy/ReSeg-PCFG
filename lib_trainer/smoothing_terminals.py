import os
import shutil
import sys
from collections import defaultdict

from tqdm import tqdm


def smoothing_terminals(rule: str):
    folders = ["Alpha", "Digits", "Other"]
    lengths = [4, 5, 6, 7, 8]
    pbar = tqdm(total=len(folders) * len(lengths), desc="Smoothing: ")
    for folder in folders:
        for length in lengths:
            pbar.update()
            origin_filename = os.path.join(rule, folder, f"{length}.txt")
            origin_file_bak = f"{origin_filename}.bak"
            if not os.path.exists(origin_file_bak):
                shutil.copyfile(origin_filename, origin_file_bak)
            fd_origin_bak = open(origin_file_bak, "r")
            origin_counter = defaultdict(float)
            for line in fd_origin_bak:
                line = line.strip("\r\n")
                pwd, prob = line.split("\t")
                origin_counter[pwd] = float(prob)
            fd_origin_bak.close()
            min_origin_v = min(origin_counter.values())
            opt_len = open(os.path.join(rule, folder, f"opt-{length}.txt"), "w")
            for len_left in range(1, length):
                fd_left = open(os.path.join(rule, folder, f"{len_left}.txt"), "r")
                for line in fd_left:
                    pwd, prob_left = line.strip("\r\n").split("\t")
                    prob_left = float(prob_left)
                    fd_right = open(os.path.join(rule, folder, f"{length - len_left}.txt"), "r")
                    first_line = fd_right.readline()
                    first_right_prob = float(first_line.strip("\r\n").split("\t")[1])
                    fd_right.seek(0)
                    threshold_prob4left = min_origin_v / first_right_prob
                    if prob_left < threshold_prob4left:
                        break
                    threshold_prob4right = min_origin_v / prob_left
                    for line_r in fd_right:
                        pwd_r, prob_right = line_r.strip("\r\n").split("\t")
                        prob_right = float(prob_right)
                        if prob_right < threshold_prob4right:
                            break
                        final_prob = prob_right * prob_left
                        final_pwd = f"{pwd}{pwd_r}"
                        if final_pwd not in origin_counter or origin_counter[final_pwd] < final_prob:
                            opt_len.write(f"{final_pwd}\t{final_prob}\n")
                    fd_right.close()
                    opt_len.flush()
                fd_left.close()
            opt_len.close()
            fd_opt = open(opt_len.name, "r")
            for line in fd_opt:
                pwd, prob = line.strip("\r\n").split("\t")
                prob = float(prob)
                if pwd not in origin_counter or origin_counter[pwd] < prob:
                    origin_counter[pwd] = prob
            fd_opt.close()
            n_sum = sum(origin_counter.values())
            fd_origin = open(origin_filename, "w")
            for p, v in origin_counter.items():
                fd_origin.write(f"{p}\t{v / n_sum:.20f}\n")
            fd_origin.close()
            pass
        pass
    pass


if __name__ == '__main__':
    smoothing_terminals("/home/cw/Codes/Python/pcfg_cracker/Rules/csdnsrc3tar1")
