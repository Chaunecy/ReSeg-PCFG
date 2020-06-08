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
terminal_re = re.compile(r"([ADKOXY]\d+)")


def extract_luds(pwd: str) -> str:
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
        print("Start loading grammars...", end="", file=sys.stderr)
        self.__load_grammars()
        self.__terminal_re = re.compile(r"([ADKOXY]\d+)")

        print("Done!\n"
              "Pre-processing...", end="", file=sys.stderr)
        luds2base_structures = {}
        filtered = defaultdict(lambda: [])
        for struct in self.count_base_structures:
            lds, rm, plen = aod2lds(struct)
            if len(rm) != 0:
                filtered[plen].append((lds, rm, struct))
                pass
            else:
                if lds not in luds2base_structures:
                    luds2base_structures[lds] = set()
                luds2base_structures[lds].add(struct)
            pass
        for s in luds2base_structures:
            ls = len(s)
            if ls not in filtered:
                continue
            add_ons = filtered.get(ls)
            for lds, rm, origin_struct in add_ons:
                rmd = rm_substr(s, rm)
                if rmd == lds:
                    luds2base_structures[s].add(origin_struct)
        del filtered
        print("Done!\n"
              "Generating Cache...", end="", file=sys.stderr)
        self.lds2base_structures = luds2base_structures
        self.__extend_structure = extend_dict(self.count_base_structures)
        self.__extend_years = extend_dict(self.count_years)
        self.__extend_context = extend_dict(self.count_context_sensitive)
        self.__extend_alpha = {k: extend_dict(v) for k, v in self.count_alpha.items()}
        self.__extend_alpha_mask = {k: extend_dict(v) for k, v in self.count_alpha_masks.items()}
        self.__extend_keyboard = {k: extend_dict(v) for k, v in self.count_keyboard.items()}
        self.__extend_other = {k: extend_dict(v) for k, v in self.count_other.items()}
        self.__extend_digits = {k: extend_dict(v) for k, v in self.count_digits.items()}
        print("done!", file=sys.stderr)

    def __load_grammars(self):
        load_grammar4scorer(self, rule_directory=self.rule)
        pass

    def calc_prob(self, pwd: str) -> float:
        # lpwd = len(pwd)
        struct = extract_luds(pwd)
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
                    if prob < 1e-100:
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
        return -log2(max(prob, 1e-100))

    def calc_minus_log2_prob_from_file(self, passwords: TextIO) -> Dict[Any, Tuple[int, float]]:
        print("Calculating minus log2 prob for passwords...", file=sys.stderr)
        raw_pwd_counter = defaultdict(int)
        for pwd in passwords:
            pwd = pwd.strip("\r\n")
            raw_pwd_counter[pwd] += 1
        pwd_counter = defaultdict(lambda: (0, .0))
        for pwd, num in tqdm(iterable=raw_pwd_counter.items(), total=len(raw_pwd_counter)):
            pwd_counter[pwd] = (num, self.minus_log2_prob(pwd))
        passwords.close()
        print("done!", file=sys.stderr)
        return pwd_counter
        pass

    def gen_n_rand_pwd(self, n: int = 10000) -> List[Tuple[float, str]]:
        print(f"Generating {n} samples...", end="", file=sys.stderr)
        pairs = []
        for _ in tqdm(range(n)):
            pairs.append(self.gen_rand_pwd())
        print("done!", file=sys.stderr)
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


def monte_carlo_wrapper(rule: str, target: TextIO, save2: TextIO, n: int = 100000):
    pcfg_scorer = MyScorer(rule=rule)
    rand_pairs = pcfg_scorer.gen_n_rand_pwd(n=n)
    minus_log_prob_list, ranks = gen_rank_from_minus_log_prob(rand_pairs)
    scored_pwd_list = pcfg_scorer.calc_minus_log2_prob_from_file(passwords=target)
    cracked = 0
    total = sum([n for n, _ in scored_pwd_list.values()])
    prev_rank = 0
    for pwd, info in tqdm(iterable=sorted(scored_pwd_list.items(), key=lambda x: x[1][1], reverse=False),
                          total=len(scored_pwd_list)):
        num, mlp = info
        rank = ceil(max(minus_log_prob2rank(minus_log_prob_list, ranks, mlp), prev_rank + 1))
        prev_rank = rank
        cracked += num
        save2.write(f"{pwd}\t{mlp:.8f}\t{num}\t{rank}\t{cracked}\t{cracked / total * 100:.2f}\n")
        pass
    save2.flush()
    save2.close()
    pass


def main():
    monte_carlo_wrapper("./Rules/Origin/rockyou",
                        target=open("/home/cw/Codes/Python/PwdTools/corpora/tar/rockyou-tar.txt"),
                        save2=open("./test.pickle", "w"))
    pass


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


if __name__ == '__main__':
    main()
    pass
