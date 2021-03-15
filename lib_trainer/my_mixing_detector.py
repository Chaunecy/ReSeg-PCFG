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
import functools
import os
import re
from collections import defaultdict
from itertools import product
from typing import Tuple, List, Dict, Set, Counter

import sys

re_tag = re.compile(r"([ADOKYX]\d+)")


def split_ado_struct(string: str) -> List[Tuple[str, str, int]]:
    """
    a replacement for re

    give me an arbitrary password and I'll give you a
    list of (section, tag)

    For example, give me hello123world,
    I'll give you [(hello, A5), (123, D3), (world, A5)]

    Note that uppercase letters will be treated as lowercase
    letters.
    :param string: any string
    :return: List[(section, tag)]
    """
    prev_chr_type = None
    acc = ""
    parts = []
    for c in string:
        if c.isalpha():
            cur_chr_type = "A"
        elif c.isdigit():
            cur_chr_type = "D"
        else:
            cur_chr_type = "O"
        if prev_chr_type is None:
            acc = c
        elif prev_chr_type == cur_chr_type:
            acc += c
        else:
            parts.append((acc, prev_chr_type, len(acc)))
            acc = c
        prev_chr_type = cur_chr_type
    parts.append((acc, prev_chr_type, len(acc)))
    return parts


def group4struct(struct: str) -> (Tuple[Tuple[str, int], ...], Tuple[Tuple[str, int], ...]):
    """
    passwords composed of same number of alphas, digits and others
    are in the same group

    count number of char of different tags.
    convert struct to an easy-to-understand form.
    :param struct: A8D3-like string
    :return: group and struct
    """
    tags = re_tag.findall(struct)
    tmp_dict = defaultdict(int)
    n_struct = []
    for tag in tags:
        k = tag[0]
        n = int(tag[1:])
        tmp_dict[k] += n
        n_struct.append((k, n))
    group = []
    for k, n in sorted(tmp_dict.items()):
        group.append((k, n))
    return tuple(group), tuple(n_struct)


def conv_pwd(pwd: str, origin_struct: Tuple[Tuple[str, int], ...], to_struct: Tuple[Tuple[str, int], ...]):
    if origin_struct == to_struct:
        return pwd
    origin_s = "".join([f"{t}" * n for t, n in origin_struct])
    to_s = "".join([f"{t}" * n for t, n in to_struct])
    n_to_s = list(to_s)
    n_pwd = ["" for _ in range(len(pwd))]
    for i, c in enumerate(origin_s):
        t_i = n_to_s.index(c)
        n_to_s[t_i] = ' '
        n_pwd[t_i] = pwd[i]
    return "".join(n_pwd)


class MixingDetector:
    def __init__(self, pcfg_parser):
        structs = pcfg_parser.count_base_structures
        extra = []
        struct_group = defaultdict(set)
        for struct in structs:
            if 'X' in set(struct):
                extra.append(struct)
            else:
                group, n_struct = group4struct(struct)
                struct_group[tuple(group)].add(n_struct)
        self.struct_group = struct_group  # type: Dict[Tuple[Tuple[str, int],...], Set[Tuple[Tuple[str, int],...]]]
        self.pwds_may_restore = pcfg_parser.pwds_may_restore

        def i2f(counter: Counter):
            s = sum(counter.values())
            n_counter = {}
            for k in counter:
                n_counter[k] = counter[k] / s
            return n_counter

        def li2f(len_counter: Dict[int, Counter]):
            n_len_counter = {}
            for _l, counter in len_counter.items():
                n_counter = {}
                s = sum(counter.values())
                for k, v in counter.items():
                    n_counter[k] = v / s
                n_len_counter[_l] = n_counter
            return n_len_counter

        self.count_base_structures = i2f(pcfg_parser.count_base_structures)
        self.count_alpha = li2f(pcfg_parser.count_alpha)
        self.count_alpha_masks = li2f(pcfg_parser.count_alpha_masks)
        self.count_other = li2f(pcfg_parser.count_other)
        self.count_digits = li2f(pcfg_parser.count_digits)
        self.count_years = i2f(pcfg_parser.count_years)
        self.count_context_sensitive = i2f(pcfg_parser.count_context_sensitive)
        self.found_mixing = defaultdict(set)
        pass

    def calc_prob(self, pwd: str):
        """
        convert a password from one structure to another,
        and find one with largest probability
        :param pwd:
        :return: prob, converted_struct, origin_struct, converted_segments, origin_segments
        """
        segmented = split_ado_struct(pwd)
        tag_set = set([t for _, t, n, in segmented])
        if len(segmented) <= len(tag_set):
            return .0, None, None, None, None
        origin_struct = tuple([(t, n) for _, t, n, in segmented])
        str_struct = "".join([f"{t}{n}" for _, t, n, in segmented])
        group, n_struct = group4struct(str_struct)
        possible_structs = self.struct_group.get(group, set())
        possible_structs.add(origin_struct)
        res_list = []
        for possible_s in possible_structs:
            # reduce the number of segments, instead of adding
            if len(possible_s) >= len(origin_struct):
                continue
            prob = 1.0
            s = "".join([f"{t}{n}" for t, n in possible_s])
            n_pwd = conv_pwd(pwd, origin_struct=origin_struct, to_struct=possible_s)
            n_pwd_parts = []
            prob *= self.count_base_structures.get(s, 0.0)
            try:
                i = 0
                for tag, span in possible_s:
                    pwd_part = n_pwd[i: i + span]
                    n_pwd_parts.append(pwd_part)
                    i += span
                    if prob <= sys.float_info.min:
                        break
                    if tag == 'A':
                        prob *= self.count_alpha.get(len(pwd_part), {}).get(pwd_part.lower(), 0.0)
                        if prob <= sys.float_info.min:
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
                    # elif tag == 'K':
                    #     prob *= self.pcfg_parser.count_keyboard.get(len(pwd_part), {}).get(pwd_part, 0.0)
                    elif tag == 'X':
                        prob *= self.count_context_sensitive.get(pwd_part, 0.0)
                    elif tag == 'Y':
                        prob *= self.count_years.get(pwd_part, 0.0)
                    else:
                        print(f"unknown tag: {tag} in {span} for {pwd_part}")
                        sys.exit(-1)
                        pass
                    pass
            except ValueError as e:
                print(e)
                sys.exit(-1)
            if prob >= sys.float_info.min:
                res_list.append((prob, possible_s, n_pwd_parts))
            pass
        if len(res_list) == 0:
            return .0, None, None, None, None
        pwd_parts = [p for p, _, _ in segmented]
        prob, to_struct, n_pwd_parts = max(res_list, key=lambda x: x[0])
        return prob, to_struct, origin_struct, n_pwd_parts, pwd_parts

    def parse(self, save_dir: str):
        """
        parse mixing, and save them
        :param save_dir:
        :return:
        """
        pwds = self.pwds_may_restore
        struct_map = defaultdict(lambda: defaultdict(list))
        for pwd, num in pwds.items():
            prob, to_struct, origin_struct, converted_parts, origin_pwd_parts = self.calc_prob(pwd)
            if origin_struct is None:
                continue
            struct_map[origin_struct][to_struct].append((origin_pwd_parts, converted_parts, num))

        folder = os.path.join(save_dir, "Mixing")
        try:
            if not os.path.exists(folder):
                os.mkdir(folder)
            for root, dirs, files in os.walk(folder):
                for f in files:
                    os.unlink(os.path.join(root, f))
                pass
        except Exception as msg:
            print(msg)
            sys.exit(-1)

        save2 = open(os.path.join(folder, 'all.txt'), "w")

        def get_set():
            return set()

        to_save = defaultdict(get_set)
        for struct, to_structs in struct_map.items():
            for reduce_to_struct, transforms in to_structs.items():
                data = defaultdict(set)
                for origin_pwd_parts, converted_parts, appearance in transforms:

                    to_save["".join(origin_pwd_parts)].add(
                        "".join(converted_parts))
                    idx = 0
                    for part in converted_parts:
                        data[idx].add(part)
                        idx += 1
                targets = [list(p) for _, p in sorted(data.items())]
                all_patterns = functools.reduce(lambda x, y: x * y, [len(p) for p in targets])
                # a hack, avoid exp growth
                if all_patterns > 256:
                    continue
                generalized = set(product(*targets))
                for g in generalized:
                    reduced = "".join(g)
                    origin = conv_pwd(reduced, to_struct=reduce_to_struct, origin_struct=struct)

                    to_save[origin].add(reduced)
                    # save2.write(f"{reduced}\t{origin}\n")
                pass
        # save2 = sys.stdout
        for origin, reduces_pwd_list in to_save.items():
            for reduce in reduces_pwd_list:
                save2.write(f"{reduce}\t{origin}\n")
        save2.flush()
        save2.close()
        #     reduces_pwd_list = list(reduces_pwd_list)
        #     if len(reduces_pwd_list) == 1 and origin not in reduces_pwd_list:
        #         save2.write(f"{reduces_pwd_list[0]}\t{origin}\n")
        #     else:
        #         tmp_list = defaultdict(float)
        #
        #         for reduced in reduces_pwd_list:
        #             tmp_list[reduced] = pwd_map.get(reduced, 10000)
        #         reduced = min(tmp_list, key=lambda k: tmp_list[k])
        #         if reduced == origin:
        #             if len(tmp_list) > 1:
        #                 tmp_list[reduced] = 100000
        #                 reduced = min(tmp_list, key=lambda k: tmp_list[k])
        #             else:
        #                 continue
        #         save2.write(f"{reduced}\t{origin}\n")
        # save2.close()
