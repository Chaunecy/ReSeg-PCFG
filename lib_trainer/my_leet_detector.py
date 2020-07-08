#!/usr/bin/env python3


#########################################################################
# Tries to detect L33t mangling in the training dataset
#
# Aka P@ssw0rd = password
#
# This relies on the multi-word detector to identify base/multi words
#
# General Approach:
# 1) If the alpha string is a base or multi word, exit
# 2) If alpha string is not a base word (too short, or not in dictionary)
#    convert one common replacement if it exists. Start with replacements
#    between two alpha strings first. Aka p@ss1 -> "pass"1. If no replacement
#    can be found, exit
# 3) Attempt to parse the expanded alpha string and see if it is a base word
#    or multiword. If it is, then return it as a l33t replacement
# 4) If not, go back to step 2 and coninue. For example now try
#    pass1 -> passl
#
# The above will likely get a bit more complicated if/when I add support for
# multiple leet conversions. Aka "a" can be transformed as "4" or "@"
#
# This first PoC will just look at the most common replacement
#
# The reason to slowly try to unmangle l33t word and start in the middle is
# to deal with people adding digits/special to the end of their password.
# For example: 1passw0rd or passw0rd1
#
# In that case, we want to "unmangle" the 0 but leave the "1" alone
#
# Note, this is very much mapped to ASCII/UTF-8 l33t mapping. Will likely
# have problems with other encoding schemes.
#
#########################################################################


# Attempts to identify l33t mangling
#
# Creating this as a class in case I want to add more advanced features later
#
import collections
import pickle
import re
import sys

from lib_trainer.my_multiword_detector import MyMultiWordDetector


class MyL33tDetector:

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
            # '4': ['a', 'h'],
            '@': ['a'],
            '8': ['ate'],
            '3': ['e'],
            '6': ['b'],
            '1': ['i'],
            '0': ['o'],
            '9': ['q'],
            '5': ['s'],
            '7': ['t'],
            '2': ['z'],
            '$': ['s']
        }
        self.l33ts = set()
        self.dict_l33ts = {}
        self.__min_l33ts = 4
        self.__max_l33ts = 8
        self.__re_lds = re.compile(r"^([0-9]+|[a-zA-Z]+|[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e]+)$")
        self.__re_invalid = re.compile(
            r"^([\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e].{1,3}|.{1,3}[\x21-\x2f\x3a-\x40\x5b-\x60\x7b-\x7e])$")

    def detect_l33t(self, pwd):
        if self.__re_lds.search(pwd):
            return False
        lower = pwd.lower()
        counter = collections.Counter(lower)
        if len(counter) < 3 or self.__re_invalid.search(pwd) or max(counter.values()) >= len(pwd) / 2:
            return False
        is_l33t, l33t = self._find_leet(lower)
        if is_l33t:
            self.l33ts.add(lower)
            if len(lower) > self.__max_l33ts:
                self.__max_l33ts = len(lower)
            if len(lower) < self.__min_l33ts:
                self.__min_l33ts = len(lower)
        return is_l33t

    def gen_dict_l33t(self):
        l33ts = sorted(self.l33ts, key=lambda x: len(x), reverse=True)
        if len(l33ts) == 0:
            return
        self.__min_l33ts = len(l33ts[-1])
        self.__max_l33ts = len(l33ts[0])
        for l33t in l33ts:
            dict_l33t = self.dict_l33ts
            for c in l33t:
                if c not in dict_l33t:
                    dict_l33t[c] = {}
                dict_l33t = dict_l33t[c]
            dict_l33t["\x03"] = True
        pass

    def _unleet(self, password):
        npasswd = ""
        for x in password:
            if x in self.replacements:
                npasswd += self.replacements[x][0]
            else:
                npasswd += x
        return npasswd

    def _find_leet(self, password):
        # if password.isalpha() or password.isdigit():
        #     return False, ""
        working_pw = self._unleet(password.lower())
        if not working_pw or password == working_pw:
            return False, ""
        else:
            count = self.multi_word_detector._get_count(working_pw)
            if count >= 5:
                return True, password
            else:
                return False, ""

    # Detects if the input has l33t replacement
    #
    # password: Password to parse
    #
    # Returns:
    #     None: If no l33t replacements were found
    #     {'1npu7':{
    #       "word":'input',
    #       "replacements":{
    #           "1":'i',
    #           "7":'t'
    #       },
    #       "strategy": "unknown"
    #      }
    #     } 
    #
    def __get_mask(self, seg):
        mask = ""
        for e in seg:
            if e.isupper():
                mask += "U"
            elif e.islower():
                mask += "L"
            else:
                mask += "L"
        return mask

    def extract_l33t(self, pwd):
        """
        find the longest match of l33t
        :param pwd:  password to be identified
        :return: list of [segment, start_idx, is_l33t]
        """
        l33t_list = []
        # candidate for a l33t
        a_l33t = ""
        # this is not a l33t, but I want to keep it to simplify the following steps
        not_l33t = ""
        # dict tree for l33ts, to speedup
        dict_l33ts = self.dict_l33ts
        lower_pwd = pwd.lower()
        len_pwd = len(pwd)
        i = 0
        prev_i = 0
        cur_i = i
        while i < len_pwd and cur_i < len_pwd:
            c = lower_pwd[cur_i]
            if c in dict_l33ts:
                a_l33t += c
                dict_l33ts = dict_l33ts[c]
                if "\x03" in dict_l33ts:
                    add_a_l33t = ""
                    bak_add_a_l33t = ""
                    for addi in range(cur_i + 1, min(cur_i + self.__max_l33ts - len(a_l33t) + 1, len_pwd)):
                        addc = lower_pwd[addi]
                        if addc not in dict_l33ts:
                            break
                        dict_l33ts = dict_l33ts[addc]
                        add_a_l33t += addc
                        if "\x03" in dict_l33ts:
                            bak_add_a_l33t = add_a_l33t
                        pass
                    if bak_add_a_l33t != "":
                        a_l33t += bak_add_a_l33t
                        cur_i += len(bak_add_a_l33t)
                    # find a l33t
                    if len(not_l33t) != 0:
                        l33t_list.append((not_l33t, cur_i - len(a_l33t) - len(not_l33t) + 1, False))
                        not_l33t = ""
                    l33t_list.append((a_l33t, cur_i - len(a_l33t) + 1, True))
                    # successfully find a l33t, move forward i
                    i += len(a_l33t)
                    cur_i = i
                    # used to find not_l33t
                    prev_i = i
                    a_l33t = ""
                    dict_l33ts = self.dict_l33ts
                cur_i += 1
            else:
                i += 1
                cur_i = i
                not_l33t = lower_pwd[prev_i:i]
                a_l33t = ""
                dict_l33ts = self.dict_l33ts
        if cur_i != i:
            l33t_list.append((lower_pwd[i:cur_i], i, False))
        else:
            print("h")
        return l33t_list
        pass

    def parse(self, password):
        if password in self.l33ts:
            return [(password, f"A{len(password)}")], [password], [self.__get_mask(password)]
        if len(password) < self.__min_l33ts or self.__re_lds.search(password) is not None:
            return [(password, None)], [], []
        l33t_list = self.extract_l33t(password)
        if len(l33t_list) == 0:
            return [(password, None)], [], []
        l33t_list = sorted(l33t_list, key=lambda x: x[1])
        section_list = []
        leet_list = []
        mask_list = []
        for l33t, idx, is_l33t in l33t_list:
            leet = password[idx:idx + len(l33t)]
            lower_leet = leet.lower()
            if is_l33t:
                section_list.append((lower_leet, f"A{len(l33t)}"))
                leet_list.append(lower_leet)
                mask = self.__get_mask(leet)
                mask_list.append(mask)
            else:
                section_list.append((lower_leet, None))
        return section_list, leet_list, mask_list

    def parse_sections(self, sections):
        parsed_sections = []
        parsed_l33t = []
        parsed_mask = []
        for section, tag in sections:
            if tag is not None:
                parsed_sections.append((section, tag))
                continue
            if len(section) < self.__min_l33ts or self.__re_lds.search(section):
                parsed_sections.append((section, None))
                continue
            section_list, leet_list, mask_list = self.parse(section)
            parsed_sections.extend(section_list)
            parsed_l33t.extend(leet_list)
            parsed_mask.extend(mask_list)
        return parsed_sections, parsed_l33t, parsed_mask


def main():
    # m = MyMultiWordDetector()
    # m.train_file(open("/home/cw/Documents/Expirements/SegLab/Corpora/csdn-src.txt"))
    # pickle.dump(m, open("./tmpcsdnmulti.pickle", "wb"))
    m = pickle.load(open("./tmpcsdnmulti.pickle", "rb"))
    # nm = pickle.load(open("/home/cw/Codes/Python/SegLab/src/SegFinder/lib_seg/multi-csdn-tar.pickle", "rb"))
    l33t = MyL33tDetector(m)
    for line in open("/home/cw/Codes/Python/pcfg_cracker/Rules/csdnl33t/L33t/all.txt"):
        line = line.strip("\r\n")
        l33t.l33ts.add(line)
    l33t.gen_dict_l33t()
    # l33t.detect_l33t("p@ssw0rd")
    sections = [["p@ssw0rd", None]]
    print("hello")
    cc = l33t.parse_sections([("abP@ssw0rds", None)])
    print(cc)
    # sections = l33t.parse_sections(sections)
    # print(sections)
    # print(res)
    pass


if __name__ == '__main__':
    main()
    pass
