import argparse
import bisect
import random
import re
import sys
from collections import Counter, defaultdict
from math import log2, ceil
from typing import List, Tuple, TextIO, Any, Dict

import numpy
from tqdm import tqdm

from lib_scorer.grammar_io import load_grammar as load_grammar4scorer

"""
Note that OMEN scorer is not considered here
if prob for OMEN is not 0, use lib_scorer please
"""


def extend_dict(counter: Counter):
    items = list(counter.keys())
    cum_counts = numpy.array(list(counter.values())).cumsum()
    return counter, items, cum_counts
    pass


def pick_extend(extend: (Counter, List[str], [])):
    counter, items, cum_counts = extend  # type: (Counter, [], [])
    total = cum_counts[-1]
    idx = bisect.bisect_right(cum_counts, random.uniform(0, total))
    item = items[idx]
    return -log2(counter.get(item)), item
    pass


def gen_rank_from_minus_log_prob(minus_log_prob_pairs: List[Tuple[float, str]]) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    calculate the ranks according to Monte Carlo method
    :param minus_log_prob_pairs: List of (minus log prob, password) tuples
    :return: minus_log_probs and corresponding ranks
    """
    minus_log_probs = numpy.fromiter((lp for lp, _ in minus_log_prob_pairs), float)
    minus_log_probs.sort()
    logn = log2(len(minus_log_probs))
    positions = (2 ** (minus_log_probs - logn)).cumsum()
    return minus_log_probs, positions
    pass


def minus_log_prob2rank(minus_log_probs, positions, minus_log_prob):
    idx = bisect.bisect_right(minus_log_probs, minus_log_prob)
    return positions[idx - 1] if idx > 0 else 1
    pass


lds_re = re.compile(r"([a-zA-Z]+|[0-9]+|[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]+)")
luds_re = re.compile(r"([a-z]+|[A-Z]+|[0-9]+|[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]+)")
terminal_re = re.compile(r"([ADKOXY]\d+)")
tag_re = re.compile(r"(L+|D+|S+|U+)")


def extract_lds(pwd: str) -> str:
    segs = [s for s in lds_re.split(pwd) if len(s) > 0]
    ret = ""
    for seg in segs:
        if seg.isalpha():
            ret += ("L" * len(seg))
        elif seg.isdigit():
            ret += ("D" * len(seg))
        else:
            ret += ("S" * len(seg))
        pass
    return ret
    pass


def extract_luds(pwd: str) -> List[Tuple[str, int]]:
    segs = [s for s in luds_re.split(pwd) if len(s) > 0]
    ret = []
    for seg in segs:
        if seg.isalpha() and seg.islower():
            ret.append(("L", len(seg)))
        elif seg.isalpha() and seg.isupper():
            ret.append(("U", len(seg)))
        elif seg.isdigit():
            ret.append(("D", len(seg)))
        else:
            ret.append(("S", len(seg)))
        pass
    return ret
    pass


def aod2lds(raw_struct: str) -> (str, List[Tuple[int, int]], int):
    # if raw_struct.find("X") > -1 or raw_struct.find("K") > -1:
    #     return use_all
    parts = terminal_re.findall(raw_struct)
    res = ""
    plen = 0
    rm = []
    start_pos = 0
    for p in parts:
        tag, span = p[0], int(p[1:])
        add_len = span
        if tag == 'A':
            res += ("L" * span)
        elif tag == 'D':
            res += ("D" * span)
        elif tag == 'O':
            res += ("S" * span)
        elif tag == 'Y':
            res += "DDDD"
            add_len *= 4
        elif tag == 'K':
            rm.append((start_pos, span))
        elif tag == 'X':
            add_len *= 2
            rm.append((start_pos, add_len))
            pass
        start_pos += add_len
        plen += add_len
    # if len(res) == plen:
    return res, rm, plen
    # else:
    #     print(f"struct for {raw_struct} does't match in length: {res}", file=sys.stderr)
    #     sys.exit(-1)
    pass


def rm_substr(struct: str, rm: List[Tuple[int, int]]) -> str:
    res = ""
    for i, s in enumerate(struct):
        need_rm = False
        for start, span in rm:
            if start <= i < start + span:
                need_rm = True
                break
        if not need_rm:
            res += s
    return res


class MyScorer:
    def __init__(self, rule: str, limit=0):
        # Information for using this grammar
        #
        print(f"rule: {rule}", file=sys.stderr)
        self.encoding = None

        # The probability limit to cut-off being categorized as a password
        self.limit = limit
        self.rule = rule

        # The following counters hold the base grammar
        #
        self.count_keyboard = {}
        self.count_emails = Counter()
        self.count_email_providers = Counter()
        self.count_website_urls = Counter()
        self.count_website_hosts = Counter()
        self.count_website_prefixes = Counter()
        self.count_years = Counter()
        self.count_context_sensitive = Counter()
        self.count_alpha = {}
        self.count_alpha_masks = {}
        self.count_digits = {}
        self.count_other = {}
        self.count_base_structures = Counter()

        self.count_raw_base_structures = Counter()
        self.minimal_prob = sys.float_info.min
        print("Start loading grammars...", end="", file=sys.stderr)
        self.__load_grammars()
        self.__terminal_re = re.compile(r"([ADKOXY]\d+)")

        print("Done!", file=sys.stderr)
        luds2base_structures = {}
        filtered = defaultdict(lambda: [])
        for struct in tqdm(self.count_base_structures, desc="Pre-processing, stage 1: "):
            lds, rm, plen = aod2lds(struct)
            if len(rm) != 0:
                filtered[plen].append((lds, rm, struct))
                pass
            else:
                if lds not in luds2base_structures:
                    luds2base_structures[lds] = set()
                luds2base_structures[lds].add(struct)
            pass
        for s in tqdm(luds2base_structures, desc="Pre-processing, stage 2: "):
            ls = len(s)
            if ls not in filtered:
                continue
            add_ons = filtered.get(ls)
            for lds, rm, origin_struct in add_ons:
                rmd = rm_substr(s, rm)
                if rmd == lds:
                    luds2base_structures[s].add(origin_struct)
        del filtered
        self.lds2base_structures = luds2base_structures
        self.__extend_structure = extend_dict(self.count_base_structures)
        self.__extend_years = extend_dict(self.count_years)
        self.__extend_context = extend_dict(self.count_context_sensitive)
        self.__extend_alpha = {k: extend_dict(v) for k, v in self.count_alpha.items()}
        self.__extend_alpha_mask = {k: extend_dict(v) for k, v in self.count_alpha_masks.items()}
        self.__extend_keyboard = {k: extend_dict(v) for k, v in self.count_keyboard.items()}
        self.__extend_other = {k: extend_dict(v) for k, v in self.count_other.items()}
        self.__extend_digits = {k: extend_dict(v) for k, v in self.count_digits.items()}

    def __load_grammars(self):
        load_grammar4scorer(self, rule_directory=self.rule)
        pass

    def calc_prob(self, pwd: str) -> float:
        # lpwd = len(pwd)
        struct = extract_lds(pwd)
        try:
            structs = self.lds2base_structures[struct]
        except KeyError:
            return 0

        prob_list = []
        for s in structs:
            prob = 1.0
            prob *= self.count_base_structures.get(s, 0.0)
            terminals = self.__terminal_re.findall(s)
            start_pos = 0
            for t in terminals:
                tag, span = t[0], int(t[1:])
                addon = span
                if tag == 'Y':
                    addon *= 4
                elif tag == 'X':
                    addon *= 2
                pwd_part = pwd[start_pos:start_pos + addon]
                if tag == 'A':
                    prob *= self.count_alpha.get(len(pwd_part), {}).get(pwd_part.lower(), 0.0)
                    if prob <= self.minimal_prob:
                        break
                    alpha_mask = ''
                    for p in pwd_part:
                        if p.isupper():
                            alpha_mask += 'U'
                        else:
                            alpha_mask += "L"
                    prob *= self.count_alpha_masks.get(len(alpha_mask), {}).get(alpha_mask, 0.0)
                elif tag == 'O':
                    prob *= self.count_other.get(len(pwd_part), {}).get(pwd_part, 0.0)
                elif tag == 'D':
                    prob *= self.count_digits.get(len(pwd_part), {}).get(pwd_part, 0.0)
                elif tag == 'K':
                    prob *= self.count_keyboard.get(len(pwd_part), {}).get(pwd_part, 0.0)
                elif tag == 'X':
                    prob *= self.count_context_sensitive.get(pwd_part, 0.0)
                elif tag == 'Y':
                    prob *= self.count_years.get(pwd_part, 0.0)
                else:
                    print(f"unknown tag: {tag} in {s} for {pwd}")
                    sys.exit(-1)
                    pass
                start_pos += addon
                if prob == 0:
                    break
                pass
            if prob != 0:
                prob_list.append(prob)
            pass
        if len(prob_list) == 0:
            return 0
        else:
            return max(prob_list)

    def minus_log2_prob(self, pwd: str) -> float:
        prob = self.calc_prob(pwd)
        return -log2(max(prob, self.minimal_prob))

    def calc_minus_log2_prob_from_file(self, passwords: TextIO) -> Dict[Any, Tuple[int, float]]:
        """

        :param passwords: passwords, do not close it please!
        :return:
        """
        print(f"target: {passwords.name}", file=sys.stderr)
        raw_pwd_counter = defaultdict(int)
        passwords.seek(0)
        for pwd in passwords:
            pwd = pwd.strip("\r\n")
            raw_pwd_counter[pwd] += 1
        pwd_counter = defaultdict(lambda: (0, .0))
        for pwd, num in tqdm(iterable=raw_pwd_counter.items(), total=len(raw_pwd_counter),
                             desc="Calc prob: "):
            pwd_counter[pwd] = (num, self.minus_log2_prob(pwd))
        return pwd_counter
        pass

    def gen_n_rand_pwd(self, n: int = 10000) -> List[Tuple[float, str]]:
        pairs = []
        for _ in tqdm(iterable=range(n), desc="Sampling: "):
            pairs.append(self.gen_rand_pwd())
        return pairs
        pass

    def gen_rand_pwd(self) -> Tuple[float, str]:
        log_prob = 0
        pwd = ""
        ext_structs = self.__extend_structure
        lp_struct, struct = pick_extend(ext_structs)
        log_prob += lp_struct
        terminals = self.__terminal_re.findall(struct)
        for t in terminals:
            tag, span = t[0], int(t[1:])
            if tag == 'A':
                lp_alpha, alpha = pick_extend(self.__extend_alpha.get(span))
                lp_mask, mask = pick_extend(self.__extend_alpha_mask.get(span))
                final_alpha = ''
                for a, m in zip(alpha, mask):  # type: str, str
                    if m == 'U':
                        final_alpha += a.upper()
                    else:
                        final_alpha += a
                    pass
                log_prob += (lp_alpha + lp_mask)
                pwd += final_alpha
            elif tag == 'O':
                lp_other, other = pick_extend(self.__extend_other.get(span))
                log_prob += lp_other
                pwd += other
            elif tag == 'D':
                lp_digits, digits = pick_extend(self.__extend_digits.get(span))
                log_prob += lp_digits
                pwd += digits
            elif tag == 'K':
                lp_kbd, kbd = pick_extend(self.__extend_keyboard.get(span))
                log_prob += lp_kbd
                pwd += kbd
            elif tag == 'X':
                lp_context, context = pick_extend(self.__extend_context)
                log_prob += lp_context
                pwd += context
            elif tag == 'Y':
                lp_year, year = pick_extend(self.__extend_years)
                log_prob += lp_year
                pwd += year
            pass
        return log_prob, pwd
        pass


def transform_struct2(pwd: str, from_struct: str, to_struct: str) -> str:
    """
    transform pwd from {from_struct} to {to_struct},
    for example, a1b2c3d4, LDLDLDLD, LLDDLLDD -> ab12cd34
    :param pwd:
    :param from_struct:
    :param to_struct:
    :return:
    """
    if from_struct == to_struct:
        return pwd
    if len(pwd) == len(from_struct) == len(to_struct):

        to_pwd = [' ' for _ in range(len(pwd))]
        for idx, k in enumerate(from_struct):
            _i = to_struct.index(k)
            to_struct = to_struct[:_i] + ' ' + to_struct[_i + 1:]
            to_pwd[_i] = pwd[idx]
        return "".join(to_pwd)
        pass
    else:
        print('unequal len for pwd and their structures')
        raise Exception
    pass


def wc_l(file: TextIO):
    """
    a pure function, file will not be closed and move the pointer to the begin
    :param file: file to count lines
    :return: number of lines
    """
    file.seek(0)
    new_line = "\n"
    buf_size = 8 * 1024 * 1024
    count = 0
    while True:
        buffer = file.read(buf_size)
        if not buffer:
            count += 1
            break
        count += buffer.count(new_line)
    file.seek(0)
    return count


def seg_of_struct(struct: str):
    segs = [tag for tag in tag_re.split(struct) if len(tag) != 0]
    return len(segs)
    pass


def struct_transform4ideal_improvement(pwd_list: TextIO, pcfg_scorer: MyScorer):
    transform_groups = defaultdict(lambda: set())
    pwd_counter = defaultdict(int)
    num_lines = wc_l(pwd_list)
    pwd_list.seek(0)
    for line in tqdm(iterable=pwd_list, total=num_lines, desc="Counting unique: "):
        line = line.strip("\r\n")
        pwd_counter[line] += 1

    mid_res = []
    for pwd, appearance in tqdm(iterable=pwd_counter.items(), total=len(pwd_counter), desc="Pre-processing: "):
        segments = extract_luds(pwd)
        tag_dict = defaultdict(int)
        for tag, num in segments:
            tag_dict[tag] += num
        struct = "".join([f"{tag * num}" for tag, num in segments])
        group = "".join([f"{tag}{num}" for tag, num in sorted(tag_dict.items())])
        chr_cls = tag_dict.keys()
        if len(segments) > len(chr_cls):
            transform_groups[group].add(struct)
        mid_res.append((pwd, appearance, struct, group, chr_cls))
    del pwd_counter
    res = []
    struct_seg_counter = defaultdict(int)
    for pwd, appearance, struct, group, chr_cls in tqdm(iterable=mid_res, total=len(mid_res), desc="Calc Max Prob: "):
        to_structs = transform_groups[group]
        if struct not in struct_seg_counter:
            struct_seg_counter[struct] = seg_of_struct(struct)
        n_seg_of_origin = struct_seg_counter[struct]
        min_mlp = pcfg_scorer.minus_log2_prob(pwd)
        opt_pwd = pwd
        for to_struct in to_structs:
            if to_struct not in struct_seg_counter:
                struct_seg_counter[to_struct] = seg_of_struct(to_struct)
            # if struct_seg_counter[to_struct] > n_seg_of_origin:
            #     continue
            to_pwd = transform_struct2(pwd, from_struct=struct, to_struct=to_struct)
            mlp = pcfg_scorer.minus_log2_prob(to_pwd)
            if mlp < min_mlp:
                min_mlp = mlp
                opt_pwd = to_pwd
        res.append((pwd, opt_pwd, min_mlp, appearance))
    del transform_groups
    del mid_res
    return res


def monte_carlo_wrapper(rule: str, target: TextIO, save2: TextIO, n: int = 100000):
    pcfg_scorer = MyScorer(rule=rule)
    rand_pairs = pcfg_scorer.gen_n_rand_pwd(n=n)
    minus_log_prob_list, ranks = gen_rank_from_minus_log_prob(rand_pairs)
    del rand_pairs
    scored_pwd_list = pcfg_scorer.calc_minus_log2_prob_from_file(passwords=target)
    target.close()
    del pcfg_scorer
    cracked = 0
    prev_rank = 0
    total = sum([n for n, _ in scored_pwd_list.values()])
    for pwd, info in tqdm(iterable=sorted(scored_pwd_list.items(), key=lambda x: x[1][1], reverse=False),
                          total=len(scored_pwd_list), desc="Estimating: "):
        num, mlp = info
        rank = ceil(max(minus_log_prob2rank(minus_log_prob_list, ranks, mlp), prev_rank + 1))
        prev_rank = rank
        cracked += num
        save2.write(f"{pwd}\t{mlp:.8f}\t{num}\t{rank}\t{cracked}\t{cracked / total * 100:.2f}\n")
    save2.flush()
    save2.close()
    del minus_log_prob_list
    del ranks
    del scored_pwd_list


def actual_ideal_wrapper(rule: str, target: TextIO, save_ideal: TextIO, n: int = 100000):
    pcfg_scorer = MyScorer(rule=rule)
    rand_pairs = pcfg_scorer.gen_n_rand_pwd(n=n)
    minus_log_prob_list, ranks = gen_rank_from_minus_log_prob(rand_pairs)
    del rand_pairs
    # scored_pwd_list = pcfg_scorer.calc_minus_log2_prob_from_file(passwords=target)
    # total = sum([n for n, _ in scored_pwd_list.values()])
    # cracked = 0
    # prev_rank = 0
    #
    # for pwd, info in tqdm(iterable=sorted(scored_pwd_list.items(), key=lambda x: x[1][1], reverse=False),
    #                       total=len(scored_pwd_list), desc="Estimating: "):
    #     num, mlp = info
    #     rank = ceil(max(minus_log_prob2rank(minus_log_prob_list, ranks, mlp), prev_rank + 1))
    #     prev_rank = rank
    #     cracked += num
    #     save2.write(f"{pwd}\t{mlp:.8f}\t{num}\t{rank}\t{cracked}\t{cracked / total * 100:.2f}\n")
    #     pass
    # del scored_pwd_list
    # save2.flush()
    # save2.close()
    ideal_res = struct_transform4ideal_improvement(pwd_list=target, pcfg_scorer=pcfg_scorer)
    del pcfg_scorer
    target.close()
    cracked = 0
    prev_rank = 0
    total2 = sum([num for _, _, _, num in ideal_res])
    for pwd, to_pwd, mlp, num in tqdm(iterable=sorted(ideal_res, key=lambda x: x[2]),
                                      total=len(ideal_res), desc="Estimating Ideal: "):
        rank = ceil(max(minus_log_prob2rank(minus_log_prob_list, ranks, mlp), prev_rank + 1))
        prev_rank = rank
        cracked += num
        save_ideal.write(f"{pwd}\t{to_pwd}\t{mlp:.8f}\t{num}\t{rank}\t{cracked}\t{cracked / total2 * 100:.2f}\n")
        pass
    del ideal_res
    del minus_log_prob_list
    del ranks
    save_ideal.flush()
    save_ideal.close()
    pass


def main():
    cli = argparse.ArgumentParser("Monte Carlo Simulator for PCFGv4.1")
    cli.add_argument("-r", "--rule", required=True, dest="rule", type=str, help="rule set obtained by trainer")
    cli.add_argument("-t", "--target", required=True, dest="target", type=argparse.FileType("r"),
                     help="password list to be parsed")
    cli.add_argument("-n", "--n-sample", required=False, dest="n", type=int, default=100000,
                     help="samples generated to execute Monte Carlo Simulation, default=100000")
    cli.add_argument("-s", "--save", required=True, dest="save2", type=argparse.FileType("w"),
                     help="save the results to specified file, the format of the results is:\n"
                          "pwd  minus_log2_prob appearance  guess_number    cracked_num cracked_ratio")
    args = cli.parse_args()
    monte_carlo_wrapper(args.rule, target=args.target, save2=args.save2)


def test():
    pcfg_scorer = MyScorer(rule="./Rules/Origin/rockyou")
    usr_in = ""
    while usr_in != "exit":
        usr_in = input("Type in password: ")
        print(pcfg_scorer.minus_log2_prob(usr_in))
    # print(pcfg_scorer.minus_log2_prob("iluvyandel"))
    # print(pcfg_scorer.minus_log2_prob("0O9I8U7Y"))
    # print(pcfg_scorer.minus_log2_prob("custom"))
    # print(pcfg_scorer.minus_log2_prob("imsocool"))
    pass


def ideal():
    for corpus in ["csdn", "rockyou", "dodonew", "webhost", "xato"]:
        actual_ideal_wrapper(
            f"C:\\SegLab\\Rules\\PCFG41\\{corpus}",
            target=open(f"D:\\SegLab\\Corpora\\{corpus}-tar.txt"),
            # save2=open(f"/home/cw/Documents/Expirements/SegLab/4src/{corpus}-tar-actual.txt", "w"),
            save_ideal=open(f"D:\\SegLab\\SimulatedCracked\\{corpus}-ideal.txt", "w"),
            n=1000000)
        pass
    pass


if __name__ == '__main__':
    main()
    pass
