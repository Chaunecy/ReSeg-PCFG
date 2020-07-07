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
import pickle


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

    def _unleet(self, password):
        npasswd = ""
        for x in password:
            if x in self.replacements:
                npasswd += self.replacements[x][0]
            else:
                npasswd += x
        return npasswd

    def _find_leet(self, password):
        working_pw = self._unleet(password.lower())
        if not working_pw or password == working_pw:
            return None
        else:
            multi_num, result = self.multi_word_detector.parse(working_pw, threshold=5)
            return multi_num, result

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
    def parse(self, password):
        l33t_list = []
        leet = self._find_leet(password)
        if leet:
            multi_num, compositions = leet
            if multi_num > 1:
                l33t = compositions[0][0]
                prob = compositions[0][1]
                if prob < 1e-15:
                    return False, [(password, None)], []
                origin = []
                i = 0
                for part in l33t:
                    restore = password[i:i + len(part)]
                    if len(restore) >= 4 and restore != part and not restore.isdigit():
                        origin.append((restore, f"A{len(restore)}"))
                        l33t_list.append(restore)
                    else:
                        return False, [(password, None)], []
                    i += len(part)
                return True, origin, l33t_list
            elif multi_num == 1:
                return True, [(password, f"A{len(password)}")], []
            return False, [(password, None)], []

        return False, [(password, None)], []

    def parse_sections(self, sections):
        parsed_sections = []
        parsed_l33t = []
        for section, tag in sections:
            if tag is not None:
                parsed_sections.append((section, tag))
                continue
            if len(section) < 4 or section.isdigit() or section.isalpha():
                parsed_sections.append((section, tag))
                continue
            is_leet, parsed, l33t_list = self.parse(section)
            parsed_sections.extend(parsed)
            parsed_l33t.extend(l33t_list)
        return parsed_sections, parsed_l33t


def main():
    # m = MultiWordDetector()
    # for pwd in ["input123", "input123", "input123", "input123", "input123", "input123",
    #             "hello123", "hello123", "hello123", "hello123", "hello123", "hello123",
    #             "inputhello", "inputhello", "inputhello"]:
    #     m.train(pwd)
    nm = pickle.load(open("./multi-csdn-tar.pickle", "rb"))
    l33t = MyL33tDetector(nm)
    sections = [["h3llo", None]]
    sections = l33t.parse_sections(sections)
    print(sections)
    # print(res)
    pass


if __name__ == '__main__':
    main()
    pass
