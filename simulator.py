import argparse
import sys
from collections import defaultdict

from bisect import bisect_right
from math import log2
from typing import TextIO

import numpy as np

from lib_guesser.pcfg_grammar import PcfgGrammar
from lib_scorer.grammar_io import load_grammar as load_grammar4scorer
from lib_scorer.pcfg_grammar import PcfgGrammar as PcfgScorer


def rand_key(a_dict: dict):
    return np.random.choice(list(a_dict.keys()), p=np.array(list(a_dict.values())).ravel())


class WeirPCFGv41:
    def __init__(self, rule: str, test: str, sample_size, fout_gc: TextIO):
        """

        :param rule: rule set
        :param test: test set
        """
        pcfg_grammar = PcfgGrammar(rule_name=rule, base_directory=rule, version="4.1", save_file=None,
                                   skip_brute=True, skip_case=False, debug=True, base_structure_folder="Grammar",
                                   guess_number=0, save_guesses_to_file=None)
        self.__structures = {tuple(b['replacements']): b['prob'] for b in pcfg_grammar.base}
        self.__replacements = {rp_type: {v: b['prob'] for b in l for v in b['values']} for rp_type, l in
                               pcfg_grammar.grammar.items()}
        self.__test_set = test
        self.__sample_size = sample_size
        self.__log_prob_of_samples = []  # type: [(float, str)] # list of tuple of log prob & sample,
        self.__pcfgScorer = PcfgScorer()
        load_grammar4scorer(self.__pcfgScorer, rule_directory=rule)
        self.__pcfgScorer.create_multiword_detector()
        self.__pcfgScorer.create_omen_scorer(base_directory=rule, max_omen_level=9)
        self.__fout_gc = fout_gc
        pass

    def __generate_one(self):
        struct = rand_key(self.__structures)
        prob = self.__structures.get(struct)
        guess = ""
        log_prob = -log2(prob)
        for part in struct:  # type: str
            replacement = rand_key(self.__replacements.get(part))
            if part.startswith('C'):
                base_len = len(guess) - len(replacement)
                list_guess = list(guess)
                for i, c in enumerate(list(replacement)):
                    if c == 'U':
                        list_guess[base_len + i] = list_guess[base_len + i].upper()
                guess = "".join(list_guess)
            else:
                guess += replacement
            log_prob += (-log2(self.__replacements.get(part).get(replacement)))

        return guess, log_prob
        pass

    def sample(self):
        print(f"Sampling {self.__sample_size} guesses...", file=sys.stderr)
        for i in range(self.__sample_size):
            if i % 1024 == 0:
                print(f"Sampling progress: {i / self.__sample_size * 100:5.2f}%", file=sys.stderr)
            guess, prob = self.__generate_one()
            self.__log_prob_of_samples.append((prob, guess))
        print("Sampling done!", file=sys.stderr)
        pass

    def evaluate(self):
        if len(self.__log_prob_of_samples) != self.__sample_size:
            self.sample()
        log_probs = np.fromiter((lp for lp, _ in self.__log_prob_of_samples), float)
        log_probs.sort()
        logn = log2(len(log_probs))
        positions = (2 ** (log_probs - logn)).cumsum()
        estimations = []
        print("Parsing test set...", file=sys.stderr)
        pwd_set = defaultdict(int)
        with open(self.__test_set, 'r') as fin:
            for line in fin:
                line = line.strip('\r\n')
                pwd_set[line] += 1
        scored = {}
        total = sum(pwd_set.values())
        for pwd, cnt in pwd_set.items():
            _, _, prob, _ = self.__pcfgScorer.parse(pwd)
            prob = max(sys.float_info.min, prob)
            log_prob = -log2(prob)
            idx = bisect_right(log_probs, log_prob)
            pos = positions[idx - 1] if idx > 0 else 0
            scored[pwd] = (cnt, log_prob, pos)
        del pwd_set
        cracked, prev_rank = 0, 0
        fd_out = self.__fout_gc
        for pwd, (cnt, log_prob, pos) in sorted(scored.items(), key=lambda x: x[1][1]):
            cracked += cnt
            rank = max(int(pos + .5), prev_rank + 1)
            prev_rank = rank
            fd_out.write(f"{pwd}\t{log_prob:.8f}\t{cnt}\t{rank}\t{cracked}\t{cracked / total * 100:.2f}\n")
            pass
        print("Parsing test set done!", file=sys.stderr)
        self.__fout_gc.flush()

    def __del__(self):
        self.__fout_gc.close()


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser("Monte Carlo Simulation")
    cli_parser.add_argument("--rule", "-r", dest="abs_rule_set", required=True, help="Rule Set")
    cli_parser.add_argument("--test", "-t", dest="abs_test_set", required=True, help="Test set")
    cli_parser.add_argument("--sample-n", "-n", dest="sample_size", required=False, type=int, default=10000,
                            help="Sample n passwords")
    cli_parser.add_argument("--guess-crack-file", "-f", dest="fout_gc", required=True,
                            type=argparse.FileType("w"), default=sys.stdout,
                            help="save guess-crack info here, use \"-\" to print into stdout")
    args = cli_parser.parse_args()
    print("WARNING: This method is DEPRECATED. If you want to use Monte Carlo method, try scorer.py please!",
          file=sys.stderr)
    weirPCFGv41 = WeirPCFGv41(rule=args.abs_rule_set, test=args.abs_test_set, sample_size=args.sample_size,
                              fout_gc=args.fout_gc)
    weirPCFGv41.evaluate()
    pass
