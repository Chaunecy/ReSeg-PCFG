"""
What's Keyboard Pattern?
"""
import abc
from collections import defaultdict
from typing import List, Tuple, Dict, Union


def single(string: str) -> bool:
    return string.isdigit() or string.isalpha() or all([not c.isdigit() and not c.isalpha() for c in string])


class Keyboard:
    def __init__(self):
        # first, second, third, forth = layout
        self._unshift = {}
        self._kbd = {}
        self._max_x = 5
        self._max_y = 15
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
                    tight_line.append(y)
                pass
        pass


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

    def adjacent(self):
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

    def sequence(self, string: str):

        pass

    def extract_kbd(self, string: str):
        """
        detect keyboard patterns in string, and return these keyboard patterns
        :param string:
        :return:
        """
        fast_fail = [], [(0, len(string), False)]
        if single(string):
            return fast_fail
        self.__track = self.__kbd.get_track(string)
        new_track, rejected, (appear_x, appear_y), (chr_cnt, uniq_cnt), total_join_edge = \
            self.__reject_isolated()
        is_kbd = False
        if chr_cnt < 4:
            return fast_fail
        elif uniq_cnt == len(appear_x) * len(appear_y):
            is_kbd = True
        # elif total_join_edge > uniq_cnt:
        #     is_kbd = True
        if not is_kbd:
            return fast_fail
        kbd_str = ["\x03" for _ in string]
        for x, line in enumerate(new_track):
            for y, indices in enumerate(line):
                c = self.__kbd.get_chr((x, y))
                for idx in indices:
                    kbd_str[idx] = c
        # keyboards = "".join(kbd_str).split("\x03")
        keyboard = ""
        kbd_list = []
        idx_list = []
        for i, c in enumerate(kbd_str):
            if c != "\x03":
                keyboard += c
            else:
                if len(keyboard) > 0:
                    idx_list.append((i - len(keyboard), len(keyboard), not single(keyboard)))
                    kbd_list.append(keyboard)
                keyboard = ""
                if len(idx_list) == 0:
                    idx_list.append((i, 1, False))
                else:
                    _start, _len, _is_kbd = idx_list[-1]
                    if not _is_kbd:
                        idx_list[-1] = (_start, _len + 1, _is_kbd)
                    else:
                        idx_list.append((i, 1, False))

        if len(keyboard) > 0:
            kbd_list.append(keyboard)
            idx_list.append((len(kbd_str) - len(keyboard), len(keyboard), not single(keyboard)))
        return kbd_list, idx_list

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


def main():
    am = AmericanKeyboard()
    kd = KeyboardDetection(am)
    kbd_dict = defaultdict(lambda: defaultdict(int))
    with open("/home/cw/Documents/Experiments/SegLab/Corpora/csdn-src.txt") as fd:
        for line in fd:
            line = line.strip("\r\n")
            kbd_list, section_list = kd.parse_sections(line)
            for kbd in kbd_list:
                kbd_dict[len(kbd)][kbd] += 1
    save = open("save.o", "w")
    for _len, len_kbd_dict in kbd_dict.items():
        for kbd, cnt in len_kbd_dict.items():
            save.write(f"{kbd}\t{cnt}\n")

    save.close()
    # print(am.get_layout())
    # t = am.get_track("hello")
    # print(t)


if __name__ == '__main__':
    main()
