"""
What's Keyboard Pattern?
"""
import abc
from typing import List, Tuple, Dict


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
        for idx_x, (normal_line, shift_line) in enumerate(_kbd):
            normal_chr_list = [c for c in list(normal_line) if len(c) > 0]
            shift_chr_list = [c for c in list(shift_line) if len(c) > 0]
            for idx_y, (normal_chr, shift_chr) in enumerate(zip(normal_chr_list, shift_chr_list)):
                unshift[shift_chr] = normal_chr
                kbd[normal_chr] = (idx_x, idx_y)
                kbd[shift_chr] = (idx_x, idx_y)
        self.__layout = kbd
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
        new_track = []
        appear_x, appear_y = set(), set()
        total_join_edge = 0
        total_single_edge = 0
        chr_cnt = 0
        uniq_cnt = 0
        for x, line in enumerate(track):
            tight_line = []
            for y, k in enumerate(line):
                if len(self.__track[x][y]) == 0:
                    tight_line.append([])
                else:
                    join_edge, single_edge = self.is_isolated((x, y))
                    if join_edge == 0:
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
        return new_track, (appear_x, appear_y), (chr_cnt, uniq_cnt), int(total_join_edge / 2)

    def detect(self, string: str):
        if string.isdigit() or string.isalpha() \
                or all([not c.isdigit() and not c.isalpha() for c in string]):
            return
        self.__track = self.__kbd.get_track(string)
        new_track, (appear_x, appear_y), (chr_cnt, uniq_cnt), total_join_edge = self.__reject_isolated()
        if total_join_edge > uniq_cnt - 2:
            pass
        for t in new_track:
            print(t)
        print(chr_cnt, uniq_cnt, total_join_edge)
        pass

    pass


if __name__ == '__main__':
    am = AmericanKeyboard()
    kd = KeyboardDetection(am)
    kd.detect("1a2s3d4fh")
    kd.detect("1qaz3edc")
    kd.detect("11q2a3z4")

    # print(am.get_layout())
    # t = am.get_track("hello")
    # print(t)
