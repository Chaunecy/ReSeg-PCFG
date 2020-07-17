"""
This is a function to detect mixing
What is mixing?
a password may be composed of several segments, however, we can reduce the number
of segments and convert it to form of easy to remember.
For example, a1b2c3d4 can be converted to abcd1234.

How to:
resort Alpha, Digit, Other, conteXt, Year
find all possible structure and calculate probabilities
pick one with largest probability.
"""


class MixingDetector:
    def __init__(self, pcfg_parser):
        structs = pcfg_parser.count_base_structures

        pass

    pass


def calc_largest(pwd, pcfg_parser):
    pass
