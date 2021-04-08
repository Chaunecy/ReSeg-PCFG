"""
What's Keyboard Pattern?
"""
import abc
from typing import List, Tuple, Dict, Union, TextIO

import sys


def single(string: str) -> bool:
    return string.isdigit() or string.isalpha() or all([not c.isdigit() and not c.isalpha() for c in string])


def rm_same_seq(string: str) -> str:
    s = []
    for c in string:
        if len(s) == 0:
            s.append(c)
        elif s[-1] != c:
            s.append(c)
    return "".join(s)
    pass


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
            parts.append((prev_chr_type, acc))
            acc = c
        prev_chr_type = cur_chr_type
    parts.append((prev_chr_type, acc))
    return parts


class Keyboard:
    def __init__(self):
        # first, second, third, forth = layout
        self._unshift = {}
        self._kbd = {}
        self._max_x = 5
        self._max_y = 15
        self._min_kbd_len = 4
        pass

    def get_bound(self):
        return self._max_x, self._max_y

    @abc.abstractmethod
    def get_layout(self) -> Dict[str, Tuple[int, int]]:
        return {}

    @abc.abstractmethod
    def get_pos(self, c: str) -> Tuple[int, int]:
        return -1, -1

    @abc.abstractmethod
    def get_chr(self, pos: Tuple[int, int]) -> str:
        return ""

    def get_zero_kbd(self) -> List[List[List[int]]]:
        """

        :return: 5 lists of 15 elements
        """
        return [[[] for _ in range(self._max_y)] for _ in range(self._max_x)]
        pass

    def rm_shift(self, string: str) -> str:
        n_string = ""
        for c in string:
            if c in self._unshift:
                n_string += self._unshift[c]
            else:
                n_string += c
        return n_string

    def get_track(self, string: str):
        """

        :param string: a string
        :return: the track on keyboard
        """
        zero_kbd = self.get_zero_kbd()
        for i, c in enumerate(string):
            x, y = self.get_pos(c)
            zero_kbd[x][y].append(i)
        return zero_kbd

    def get_tight_track(self, string: str):
        """

        :param string: a string
        :return: the track on keyboard
        """
        zero_kbd = self.get_zero_kbd()
        appear_x = set()
        appear_y = set()
        for i, c in enumerate(string):
            x, y = self.get_pos(c)
            zero_kbd[x][y].append(i)
            appear_x.add(x)
            appear_y.add(y)
        tight = []
        for x, line in enumerate(zero_kbd):
            if x not in appear_x:
                continue
            tight_line = []
            for y, k in enumerate(line):
                if y in appear_y:
                    if len(k) > 0:
                        tight_line.append(self.get_chr((x, y)))
                    else:
                        # to simplify
                        tight_line.append('\x00')
            if len(tight_line) > 0:
                tight.append(tight_line)
        return tight, (len(appear_x), len(appear_y))


class AmericanKeyboard(Keyboard):

    def get_chr(self, pos: Tuple[int, int]) -> str:
        """
        Note that this function return only unshift keys
        :param pos:
        :return:
        """
        return self.__pos_dict.get(pos)

    def __init__(self):
        super().__init__()
        _kbd = [
            (
                "`1234567890-=",
                "~!@#$%^&*()_+",
            ),
            (
                "\tqwertyuiop[]\\",
                "\tQWERTYUIOP{}|",
            ),
            (
                "\x00asdfghjkl;'\n",
                "\x00ASDFGHJKL:\"\n",
            ),
            (
                "\x00zxcvbnm,./\x00",
                "\x00ZXCVBNM<>?\x00",
            )
        ]

        kbd = {}
        unshift = {}
        pos_dict = {}
        for idx_x, (normal_line, shift_line) in enumerate(_kbd):
            normal_chr_list = [c for c in list(normal_line) if len(c) > 0]
            shift_chr_list = [c for c in list(shift_line) if len(c) > 0]
            for idx_y, (normal_chr, shift_chr) in enumerate(zip(normal_chr_list, shift_chr_list)):
                unshift[shift_chr] = normal_chr
                kbd[normal_chr] = (idx_x, idx_y)
                kbd[shift_chr] = (idx_x, idx_y)
                pos_dict[(idx_x, idx_y)] = normal_chr
        self.__layout = kbd
        self.__pos_dict = pos_dict
        self._unshift = unshift

    def get_layout(self):
        return self.__layout

    def get_pos(self, c: str) -> Tuple[int, int]:
        return self.__layout.get(c)

    def rm_shift(self, string: str) -> str:
        n_string = ""
        for c in string:
            if c in self._unshift:
                n_string += self._unshift[c]
            else:
                n_string += c
        return n_string


class KeyboardDetection:
    def __init__(self, kbd: Keyboard):
        self.__kbd = kbd
        self.__track: List[List[List[int]]] = []
        pass

    def is_isolated(self, point: Tuple[int, int]):
        x, y = point
        edge = 0
        join_edge = 0
        min_x, min_y = 0, 0
        max_x, max_y = self.__kbd.get_bound()
        for addon_x in [-1, 0, 1]:
            for addon_y in [-1, 0, 1]:
                if addon_x == 0 and addon_y == 0:
                    continue
                n_x, n_y = x + addon_x, y + addon_y
                if n_x < min_x or n_x >= max_x or n_y < min_y or n_y >= max_y:
                    continue
                if len(self.__track[n_x][n_y]) > 0:
                    join_edge += 1
                else:
                    edge += 1
                    pass
        return join_edge, edge
        # if join_edge == 0:
        #     return True
        # else:
        #     return False

    def __reject_isolated(self):
        track = self.__track
        new_track: List[List[List[int]]] = []
        appear_x, appear_y = set(), set()
        total_join_edge = 0
        total_single_edge = 0
        chr_cnt = 0
        uniq_cnt = 0
        rejected = []
        for x, line in enumerate(track):
            tight_line: List[List[int]] = []
            for y, k in enumerate(line):
                if len(self.__track[x][y]) == 0:
                    tight_line.append([])
                else:
                    join_edge, single_edge = self.is_isolated((x, y))
                    if join_edge == 0:
                        rejected.extend(self.__track[x][y])
                        tight_line.append([])
                    else:
                        appear_x.add(x)
                        appear_y.add(y)
                        tight_line.append(k)
                        chr_cnt += len(k)
                        uniq_cnt += 1
                    total_join_edge += join_edge
                    total_single_edge += single_edge
                pass

            new_track.append(tight_line)

        return new_track, rejected, (appear_x, appear_y), (chr_cnt, uniq_cnt), int(total_join_edge / 2)

    def sequence(self, string: str, min_kbd_len: int = 4):
        if len(string) < min_kbd_len:
            return [], []
        prev_x, prev_y = self.__kbd.get_pos(string[0])
        kbd_list = []
        sec_list = []
        seq = ""
        idx = 0
        len_string = len(string)
        prev_kbd_idx = 0
        idx_list = []
        while idx < len_string:
            c = string[idx]
            cur_x, cur_y = self.__kbd.get_pos(c)
            adjacent = abs(cur_x - prev_x) <= 1 and abs(cur_y - prev_y) <= 1
            prev_x, prev_y = cur_x, cur_y
            if adjacent:
                seq += c
            else:
                if len(seq) >= min_kbd_len and not single(seq):
                    kbd_list.append(seq)
                    idx_list.append(idx - len(seq))
                    if len(seq) < idx - prev_kbd_idx:
                        sec_list.append((string[prev_kbd_idx:idx - len(seq)], None))
                    sec_list.append((seq, f"K{len(seq)}"))
                    prev_kbd_idx = idx
                seq = c
            idx += 1
        if len(seq) >= min_kbd_len and not single(seq):
            kbd_list.append(seq)
            idx_list.append(idx - len(seq))
            if len(seq) < idx - prev_kbd_idx:
                sec_list.append((string[prev_kbd_idx:idx - len(seq)], None))
            sec_list.append((seq, f"K{len(seq)}"))
        else:
            sec_list.append((string[prev_kbd_idx:idx], None))
        return kbd_list, idx_list

    def parallel2(self, string: str, min_kbd_len: int = 4):
        if len(string) < min_kbd_len or single(string):
            return [], []
        all_kbd = []
        all_idx = []
        next_i = 0
        for i in range(0, len(string) - min_kbd_len + 1):
            if i < next_i:
                continue
            for j in range(len(string), i + min_kbd_len - 1, -1):
                tmp_s = string[i:j]
                if single(tmp_s):
                    break
                track, (row_cnt, col_cnt) = self.__kbd.get_tight_track(tmp_s)
                if len(set(tmp_s)) == row_cnt * col_cnt and min(row_cnt, col_cnt) > 1:
                    all_kbd.append(tmp_s)
                    all_idx.append(i)
                    next_i = j
                    break
                else:
                    next_i = i + 1
                pass
            pass
        return all_kbd, all_idx
        pass

    def vertical(self, string: str, min_kbd_len: int = 4):
        if len(string) < min_kbd_len or single(string):
            return [], []
        all_kbd = []
        all_idx = []
        next_i = 0
        for i in range(0, len(string) - min_kbd_len):
            if i < next_i:
                continue
            for j in range(len(string), i + min_kbd_len - 1, -1):
                tmp_s = string[i:j]
                if len(tmp_s) < min_kbd_len:
                    break
                if single(tmp_s):
                    break
                alphas = set()
                others = set()
                for c in tmp_s:
                    if c.isalpha():
                        alphas.add(c)
                    else:
                        others.add(c)
                k = split_ado(tmp_s)
                if abs(len(alphas) - len(others)) > 1:
                    continue
                if len(k) < len(tmp_s):
                    continue
                track, _ = self.__kbd.get_tight_track(tmp_s)
                col_no_x00 = True
                row_no_x00 = True
                for idx_r, row in enumerate(track):
                    if row[0] != '\x00':
                        continue
                    for c in row:
                        if c == '\x00':
                            row_no_x00 = False
                            break
                    break
                if len(track) > 1:
                    c_idx = -1
                    for row in track:
                        if len(row) <= 1:
                            break
                        for idx_c, itm in enumerate(row):
                            if itm != '\x00':
                                if c_idx < 0:
                                    c_idx = idx_c
                                elif idx_c != c_idx:
                                    col_no_x00 = False
                                break
                        pass
                else:
                    col_no_x00 = True
                    row_no_x00 = True

                if col_no_x00 and row_no_x00:
                    all_kbd.append(tmp_s)
                    all_idx.append(i)
                    next_i = j
                    break
                else:
                    next_i = i + 1
                pass
            pass
        return all_kbd, all_idx

    def extract_kbd(self, string: str):
        """
        detect keyboard patterns in string, and return these keyboard patterns
        :param string:
        :return:
        """
        # the pwd has two segments, or each segment has length 3+
        kbd_list = []
        idx_list = []
        seq, idx4seq = self.sequence(string)
        # total += 1
        if len(seq) > 0:
            kbd_list = seq
            idx_list = idx4seq
            # sequence += 1
            print(1)
            pass
        else:
            par, idx4par = self.parallel2(string)
            if len(par) > 0:
                kbd_list = par
                idx_list = idx4par
                # parallel += 1
                print(2)
                pass
            else:
                ver, idx4ver = self.vertical(string)
                if len(ver) > 0:
                    kbd_list = ver
                    idx_list = idx4ver
                    pass
        renewed = []
        prev = 0
        for i in range(len(idx_list)):
            idx_of_i = idx_list[i]
            if idx_of_i > prev:
                renewed.append((0, idx_of_i, False))
            renewed.append((idx_of_i, len(kbd_list[i]), True))
            prev = idx_of_i + len(kbd_list[i])
        if prev < len(string):
            renewed.append((prev, len(string), False))
        return kbd_list, renewed

    def parse_sections(self, string: str, tag4kbd: str = "K") -> Tuple[List[str], List[Tuple[str, Union[str, None]]]]:
        """
        find keyboard patterns and structures of the given string
        :param string: string to be parsed
        :param tag4kbd: you may define how to label kbd patterns
        :return: keyboard patterns, sections of the string
        """
        _, idx_list = self.extract_kbd(string)
        section_list: List[Tuple[str, Union[str, None]]] = []
        kbd_list: List[str] = []
        for _start, _len, _is_kbd in idx_list:
            section = string[_start:_start + _len]
            tag = f"{tag4kbd}{_len}"
            if _is_kbd:
                section_list.append((section, tag))
                kbd_list.append(section)
            else:
                section_list.append((section, None))
        return kbd_list, section_list

    pass


def test():
    am = AmericanKeyboard()
    kd = KeyboardDetection(am)
    for pwd in []:
        print(kd.parse_sections(pwd))
        pass
    pass


def wrapper(corpora: List[TextIO]):
    am = AmericanKeyboard()
    kd = KeyboardDetection(am)
    for corpus in corpora:
        sequence = 0
        parallel = 0
        vertical = 0
        total = 0
        fd = corpus
        for line in fd:
            if total % 10000 == 0:
                print(total, file=sys.stderr)
            line = line.strip("\r\n")
            seq, _ = kd.sequence(line)
            total += 1
            if len(seq) > 0:
                sequence += 1
                # print('seq', seq)
            else:
                par, _ = kd.parallel2(line)
                if len(par) > 0:
                    parallel += 1
                    # print('par', par)
                else:
                    ver, _ = kd.vertical(line)
                    if len(ver) > 0:
                        # print('ver', ver)
                        vertical += 1
            pass
        total_kbd = sequence + parallel + vertical
        print("")
        print(f"corpus: {corpus.name}")
        print(f"sequence: {sequence}, %: {sequence / total_kbd * 100},\n"
              f"parallel: {parallel}, %: {parallel / total_kbd * 100},\n"
              f"vertical: {vertical}, %: {vertical / total_kbd * 100}")


if __name__ == '__main__':
    test()
    pass
