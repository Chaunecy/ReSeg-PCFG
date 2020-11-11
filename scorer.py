import argparse
import bisect
import random
import re
import sys
from collections import Counter, defaultdict
from math import log2, ceil
from typing import List, Tuple, TextIO, Any, Dict

from lib_scorer.grammar_io import load_grammar as load_grammar4scorer

"""
Note that OMEN scorer is not considered here
if prob for OMEN is not 0, use lib_scorer please
WARNING: 
L33t has not been considered. so passwords with l33t will not be correctly evaluated.
"""

terminal_re = re.compile(r"([ADKOXY]\d+)")


def split_ado(string):
    """
    a replacement for re
    :param string: any string
    :return: alpha, digit, other parts in a list
    """
    prev_chr_type = None
    acc = ""
    parts = []
    for c in string:
        if c.isalpha():
            cur_chr_type = "alpha"
        elif c.isdigit():
            cur_chr_type = "digit"
        else:
            cur_chr_type = "other"
        if prev_chr_type is None:
            acc = c
        elif prev_chr_type == cur_chr_type:
            acc += c
        else:
            parts.append(acc)
            acc = c
        prev_chr_type = cur_chr_type
    parts.append(acc)
    return parts


def extend_dict(counter: Counter):
    items = list(counter.keys())
    cum_counts = my_cumsum(list(counter.values()))
    return counter, items, cum_counts
    pass


def pick_extend(extend: (Counter, List[str], [])):
    counter, items, cum_counts = extend  # type: (Counter, [], [])
    total = cum_counts[-1]
    idx = bisect.bisect_right(cum_counts, random.uniform(0, total))
    item = items[idx]
    return -log2(counter.get(item)), item
    pass


def my_cumsum(lst: List[float]):
    if len(lst) <= 0:
        return []
    acc = 0
    cumsum = []
    for v in lst:
        acc += v
        cumsum.append(acc)
    return cumsum


def gen_rank_from_minus_log_prob(minus_log_prob_pairs: List[Tuple[float, str]]) -> Tuple[List[float], List[float]]:
    """
    calculate the ranks according to Monte Carlo method
    :param minus_log_prob_pairs: List of (minus log prob, password) tuples
    :return: minus_log_probs and corresponding ranks
    """
    minus_log_probs = [lp for lp, _ in minus_log_prob_pairs]
    minus_log_probs.sort()
    logn = log2(len(minus_log_probs))
    positions = my_cumsum([2 ** (mlp - logn) for mlp in minus_log_probs])
    return minus_log_probs, positions
    pass


def minus_log_prob2rank(minus_log_probs, positions, minus_log_prob):
    idx = bisect.bisect_right(minus_log_probs, minus_log_prob)
    return positions[idx - 1] if idx > 0 else 1
    pass


def extract_lds(pwd: str) -> str:
    segs = split_ado(pwd)
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


def ado2lds(raw_struct: str) -> (str, List[Tuple[int, int]], int):
    """
    Note that the length of Tag X is unknown, here I assign it length 2.
    This is a trade off.

    X, K, these two tags is hard to parse. So passwords containing these
    will be parsed in another way.
    :param raw_struct: structure of password
    :return: what password's structures may be, (LLLDDD, removed, 6)
    """
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
    """
    remove substr "rm" from struct
    :param struct: struct to be parsed
    :param rm: List[(start_pos, length)]
    :return: removed struct
    """
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
        print(f"Start loading grammars...", end="", file=sys.stderr)
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
        self.__load_grammars()
        self.__terminal_re = terminal_re
        print("Done!", file=sys.stderr)
        base_struct_tree = {}
        for base_struct in self.count_base_structures:
            terminals = self.__terminal_re.findall(base_struct)
            tree = base_struct_tree
            for terminal in terminals:
                tag = terminal[0]
                num = int(terminal[1:])
                n_terminal = (tag, num)
                if n_terminal not in tree:
                    tree[n_terminal] = {}
                tree = tree[n_terminal]
            tree[("END", 0)] = self.count_base_structures.get(base_struct)
        self.base_struct_tree = base_struct_tree
        all_terminals = {}
        for counter, tag in [(self.count_alpha, "A"), (self.count_alpha_masks, "C"), (self.count_digits, "D"),
                             (self.count_other, "O"), (self.count_keyboard, "K")]:
            for _len, items in counter.items():
                all_terminals[(tag, _len)] = items

            pass
        for items, tag in [(self.count_context_sensitive, "X"), (self.count_years, "Y")]:
            all_terminals[(tag, 1)] = items
            pass
        self.all_terminals = all_terminals

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

    def my_calc_prob(self, pwd: str, cap: str, idx: int, root: Dict, acc_prob: float, max_prob: List[float]):
        if idx == len(pwd):
            if ("END", 0) in root:
                final_prob = root[("END", 0)] * acc_prob
                if final_prob > max_prob[0]:
                    max_prob[0] = final_prob
            return
        for terminal in root:
            tag, _len = terminal

            if tag == "A":
                end_idx = idx + _len
                if end_idx > len(pwd):
                    continue
                part = pwd[idx: end_idx]
                mask = cap[idx: end_idx]
                part_prob = self.count_alpha.get(_len, {}).get(part, .0)
                mask_prob = self.count_alpha_masks.get(_len, {}).get(mask, .0)
                mul_prob = part_prob * mask_prob
                # if mul_prob < sys.float_info.min:
                #     continue
            elif tag == 'Y':
                end_idx = idx + 4
                if end_idx > len(pwd):
                    continue
                part = pwd[idx: end_idx]
                mul_prob = self.count_years.get(part, .0)
            elif tag == 'X':
                for addon in [2, 3, 4]:
                    if idx + addon > len(pwd):
                        continue
                    part = pwd[idx:idx + addon]
                    mul_prob = self.count_context_sensitive.get(part, .0)

                    if mul_prob > sys.float_info.min:
                        self.my_calc_prob(pwd, cap, idx + addon, root[terminal], acc_prob * mul_prob, max_prob)
                continue
            elif tag != "END":
                # D, O, K
                end_idx = idx + _len
                if end_idx > len(pwd):
                    continue
                part = pwd[idx: end_idx]
                mul_prob = self.all_terminals[terminal].get(part, .0)
                pass
            else:
                continue
            if mul_prob > sys.float_info.min:
                tmp_acc_prob = acc_prob * mul_prob
                self.my_calc_prob(pwd, cap, end_idx, root[terminal], tmp_acc_prob, max_prob)
            pass
        pass

    def minus_log2_prob(self, pwd: str) -> float:
        # prob = self.calc_prob(pwd)
        cap = ""
        for c in pwd:
            if c.isupper():
                cap += "U"
            else:
                cap += "L"
        max_prob = [sys.float_info.min]
        self.my_calc_prob(pwd.lower(), cap, 0, self.base_struct_tree, 1.0, max_prob)
        return -log2(max(max_prob[0], self.minimal_prob))

    def calc_minus_log2_prob_from_file(self, passwords: TextIO) -> Dict[Any, Tuple[int, float]]:
        """
        return a dict, whose key is pwd, value is tuple of (appearance, minus_log_prob)
        :param passwords: passwords, do not close it please!
        :return:
        """
        raw_pwd_counter = defaultdict(int)
        passwords.seek(0)
        for pwd in passwords:
            pwd = pwd.strip("\r\n")
            raw_pwd_counter[pwd] += 1
        pwd_counter = defaultdict(lambda: (0, .0))
        total_items = len(raw_pwd_counter)
        print("Calc prob...", file=sys.stderr)
        i = 0
        back = "\b" * 21
        for pwd, num in raw_pwd_counter.items():
            pwd_counter[pwd] = (num, self.minus_log2_prob(pwd))
            i += 1
            print(f"{i:10d}/{total_items:10d}{back}", end="", file=sys.stderr)
        return pwd_counter
        pass

    def gen_n_rand_pwd(self, n: int) -> List[Tuple[float, str]]:
        pairs = []
        for _ in range(n):
            pairs.append(self.gen_rand_pwd())
        return pairs

    def gen_rand_pwd(self) -> Tuple[float, str]:
        """
        generate a random password and get its minus log probability
        :return: (minus_log_prob, pwd)
        """
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


def monte_carlo_wrapper(rule: str, target: TextIO, save2: TextIO, n: int = 100000):
    print(f"rule: {rule}", file=sys.stderr)
    print(f"target: {target.name}", file=sys.stderr)
    pcfg_scorer = MyScorer(rule=rule)
    # sampling n passwords
    rand_pairs = pcfg_scorer.gen_n_rand_pwd(n=n)
    # generate corresponding rank list
    minus_log_prob_list, ranks = gen_rank_from_minus_log_prob(rand_pairs)
    del rand_pairs
    # scoring passwords in test set
    scored_pwd_list = pcfg_scorer.calc_minus_log2_prob_from_file(passwords=target)
    target.close()
    del pcfg_scorer
    cracked = 0
    prev_rank = 0
    total = sum([n for n, _ in scored_pwd_list.values()])
    # estimating
    print("Estimating...", file=sys.stderr)
    for pwd, info in sorted(scored_pwd_list.items(), key=lambda x: x[1][1], reverse=False):
        num, mlp = info
        # rank should be an integer, and larger than previous one
        rank = ceil(max(minus_log_prob2rank(minus_log_prob_list, ranks, mlp), prev_rank + 1))
        prev_rank = rank
        cracked += num
        save2.write(f"{pwd}\t{mlp:.8f}\t{num}\t{rank}\t{cracked}\t{cracked / total * 100:.2f}\n")
    save2.flush()
    save2.close()
    del minus_log_prob_list
    del ranks
    del scored_pwd_list


def main():
    cli = argparse.ArgumentParser("Monte Carlo Simulator for PCFGv4.1")
    cli.add_argument("-r", "--rule", required=True, dest="rule", type=str, help="rule set obtained by trainer")
    cli.add_argument("-t", "--target", required=True, dest="target", type=argparse.FileType("r"),
                     help="password list to be parsed")
    cli.add_argument("-n", "--n-sample", required=False, dest="n", type=int, default=100000,
                     help="samples generated to execute Monte Carlo Simulation, default=100000")
    cli.add_argument("-s", "--save", required=True, dest="save2", type=argparse.FileType("w"),
                     help="save the results to specified file")
    args = cli.parse_args()
    try:
        monte_carlo_wrapper(args.rule, target=args.target, save2=args.save2, n=args.n)
    except KeyboardInterrupt:
        print("You canceled the progress.\n"
              "Exited", file=sys.stderr)
        sys.exit(-1)


def test():
    pcfg_scorer = MyScorer(rule="/home/cw/Documents/Experiments/SegLab/Rules/DRed/rockyou")
    usr_in = ""
    while usr_in != "exit":
        usr_in = input("Type in password: ")
        print(pcfg_scorer.minus_log2_prob(usr_in))


if __name__ == '__main__':
    main()
    pass
