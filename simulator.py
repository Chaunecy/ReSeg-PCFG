import argparse
import itertools
import numpy as np
# import matplotlib.pyplot as plt
from bisect import bisect_right
from lib_guesser.pcfg_grammar import PcfgGrammar
from random import choice as rand_choice
from math import log2, ceil
from lib_scorer.pcfg_grammar import PcfgGrammar as PcfgScorer
from lib_scorer.grammar_io import load_grammar as load_grammar4scorer


def rand_key(a_dict: dict):
    return np.random.choice(list(a_dict.keys()), p=np.array(list(a_dict.values())).ravel())


def gen_guess_crack(estimations: [int], upper_bound=10 ** 20):
    guesses = [0]
    cracked = [0]
    estimations.sort()
    print(estimations)
    gc_pairs = {}
    for idx, est in enumerate(estimations):
        gc_pairs[est] = idx
    # for m, n in itertools.groupby(estimations):
    #     if m <= upper_bound:
    #         guesses.append(m)
    #         cracked.append((cracked[-1]) + len(list(n)))
    # return guesses[1:], cracked[1:]
    return gc_pairs.keys(), gc_pairs.values()
    pass


# def draw_gc_curve(guesses: [int], cracked: [float], label, save2file: str):
#     plt.plot(guesses, cracked, label=label)
#     plt.xscale("log")
#     plt.grid(ls="--")
#     plt.xlabel('Guesses')
#     plt.ylabel('Cracked(%)')
#     plt.legend(loc=2)
#     plt.savefig(save2file)
#
#     pass


class WeirPCFGv41:
    def __init__(self, rule: str, test: str, sample_size, gc_filename):
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
        self.__gc_filename = gc_filename
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
        for i in range(self.__sample_size):
            guess, prob = self.__generate_one()
            self.__log_prob_of_samples.append((prob, guess))
        pass

    def evaluate(self):
        if len(self.__log_prob_of_samples) != self.__sample_size:
            self.sample()
        log_probs = np.fromiter((lp for lp, _ in self.__log_prob_of_samples), float)
        log_probs.sort()
        logn = log2(len(log_probs))
        positions = (2 ** (log_probs - logn)).cumsum()
        max_pos_idx = len(positions) - 1
        estimations = []
        with open(self.__test_set, "r") as fin:
            for line in fin:
                line = line.strip("\r\n")
                try:
                    _, _, prob, _ = self.__pcfgScorer.parse(line)
                    log_prob = -log2(prob)
                except ValueError:
                    log_prob = float("inf")
                idx = bisect_right(log_probs, log_prob)
                pos = positions[idx - 1] if idx > 0 else 0
                estimations.append(int(pos))
        guesses, cracked = gen_guess_crack(estimations)
        with open(self.__gc_filename, "w") as fout:
            for g, c in zip(guesses, cracked):
                fout.write(f"{g} : {c}\n")
            fout.flush()
            pass
        pass

    pass


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser("Monte Carlo Simulation")
    cli_parser.add_argument("--rule", "-r", dest="abs_rule_set", required=True, help="Rule Set")
    cli_parser.add_argument("--test", "-t", dest="abs_test_set", required=True, help="Test set")
    cli_parser.add_argument("--sample-n", "-n", dest="sample_size", required=False, type=int, default=10000,
                            help="Sample n passwords")
    cli_parser.add_argument("--guess-crack-file", "-f", dest="gc_filename", required=False, type=str,
                            help="save guess-crack info here")
    args = cli_parser.parse_args()
    weirPCFGv41 = WeirPCFGv41(rule=args.abs_rule_set, test=args.abs_test_set, sample_size=args.sample_size,
                              gc_filename=args.gc_filename)
    # weirPCFGv41.sample()
    weirPCFGv41.evaluate()
    pass
