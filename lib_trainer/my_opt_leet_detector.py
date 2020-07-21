import functools
import itertools
import re


class EngL33tDetector:

    # Initialize the Leet detector
    #
    # multi_word_detector: A previously trained multi word detector
    #
    def __init__(self, multi_word_detector):
        self.multi_word_detector = multi_word_detector

        # A mapping of possible l33t replacements to check
        #
        # Note: Currently there is no support for multi letter l33t replacements
        # For example '|-|' = "h".
        # I don't think this happens too often so that is a future improvement
        # to look into.
        #
        # Also note, while eventually I'd like to check all possible
        # leet replacements, and I included multiple leet replacements below,
        # the current PoC only checks the first item.
        #
        # Checking multiple replacements is a higher priority target simply
        # because '1' can map to both 'L', and 'i', fairly frequently.
        #
        self.replacements = {
            '/-\\': ['a'],
            "/\\": ['a'],
            "|3": ['b'],
            "|o": ['b'],
            "(": ['c', 'g'],
            "<": ['c'],
            "k": ['c', 'k'],
            "s": ['c', 's'],
            "|)": ['d'],
            "o|": ["d"],
            "|>": ['d'],
            "<|": ["d"],
            "|=": ['f'],
            "ph": ['f', 'ph'],
            "9": ['g'],
            "|-|": ['h'],
            "]-[": ['h'],
            '}-{': ['h'],
            "(-)": ['h'],
            ")-(": ['h'],
            "#": ['h'],
            "l": ['i', 'l'],
            "|": ['i', 'l'],
            "!": ['i'],
            "][": ['i'],
            "_|": ['j'],
            "|<": ['k'],
            "/<": ['k'],
            "\\<": ['k'],
            "|{": ['k'],
            "|_": ['l'],
            "|v|": ['m'],
            "/\\/\\": ['m'],
            "|'|'|": ['m'],
            "(v)": ['m'],
            "/\\\\": ['m'],
            "/|\\": ['m'],
            '/v\\': ['m'],
            '|\\|': ['n'],
            "/\\/": ['n'],
            "|\\\\|": ['n'],
            "/|/": ['n'],
            "()": ['o'],
            "[]": ['o'],
            "{}": ['o'],
            "|2": ['p', 'r'],
            "|D": ["p"],
            "(,)": ['q'],
            "kw": ['q', 'kw'],
            "|z": ['r'],
            "|?": ['r'],
            "+": ['t'],
            "']['": ['t'],
            "|_|": ['u'],
            "|/": ['v'],
            "\\|": ['v'],
            "\\/": ['v'],
            "/": ['v'],
            "\\/\\/": ['w'],
            "\\|\\|": ['w'],
            "|/|/": ['w'],
            "\\|/": ['w'],
            "\\^/": ['w'],
            "//": ['w'],
            "vv": ['w'],
            "><": ['x'],
            "}{": ['x'],
            "`/": ['y'],
            "'/": ['y'],
            "j": ['y', 'j'],
            "(\\)": ['z'],
            '@': ['a'],
            '8': ['b', 'ate'],
            '3': ['e'],
            '6': ['b', 'g'],
            '1': ['i', 'l'],
            '0': ['o'],
            # '9': ['q'],
            '5': ['s'],
            '7': ['t'],
            '2': ['z', 'too'],
            '4': ['a', 'for'],
            '$': ['s']
        }
        repl_dict_tree = {}
        for repl, convs in self.replacements.items():
            tmp_d = repl_dict_tree
            for c in repl:
                if c not in tmp_d:
                    tmp_d[c] = {}
                tmp_d = tmp_d[c]
            tmp_d["\x02"] = convs
        self.repl_dict_tree = repl_dict_tree
        self.max_len_repl = len(max(self.replacements, key=lambda x: len(x)))
        self.l33t_map = {}
        self.dict_l33ts = {}
        self.__min_l33ts = 4
        self.__max_l33ts = 8
        self.__re_lds = re.compile(r"^([0-9]+|[a-zA-Z]+|[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]+)$")
        # lower string
        self.__re_invalid = re.compile(
            r"^("
            r"[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e0-9]{1,3}[a-z]{1,3}"  # except (S or D) + L
            r"|[0-9]+[a-z]{1,2}|[a-z]{1,2}[0-9]+"  # remove m150, 
            r"|[a-z]{1,3}[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e0-9]{1,3}"  # except L + (S or D)
            r"|[02356789@]{1,2}[a-z]+"  # except 5scott
            r"|[a-z0-9]4(ever|life)"  # except a4ever, b4ever
            r"|1[a-z]{1,4}[^u]"  # except 1hateu, 1loveu
            r"|1il(ov|uv).+"  # except 1iloveyou
            r"|[a-z]{3,}[0-9$]+"
            r"|(000)?we?bh(o?st)?)$")
        self.__re_end_at = re.compile(r"^([a-z]+)@+$")

    def unleet(self, word: str) -> itertools.product:
        unleeted = []
        repl_dtree = self.repl_dict_tree
        i = 0
        while i < len(word):
            max_m = word[i]
            if max_m not in repl_dtree:
                unleeted.append([max_m])
                i += 1
                continue
            add_on = 1
            for t in range(2, self.max_len_repl + 1):

                n_key = word[i:i + t]
                if n_key not in self.replacements:
                    continue
                max_m = n_key
                add_on = t
            if max_m not in self.replacements:
                repl_list = [max_m]
            else:
                repl_list = self.replacements.get(max_m)
            i += add_on
            unleeted.append(repl_list)
        all_num = functools.reduce(lambda x, y: x * y, [len(p) for p in unleeted])
        if all_num >= 256:
            return []
        all_possibles = itertools.product(*unleeted)
        return all_possibles

    def find_l33t(self, unleeted: itertools.product):

        pass

    def detect_l33t(self, pwd: str):
        pass


if __name__ == '__main__':
    ml33t = EngL33tDetector(None)
    t = ml33t.unleet("llove|_|")
    for p in t:
        print(p)
