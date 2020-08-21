"""
Keyboard Patterns Detection implemented is insufficient to parse q1w2e3r4 or so.
Here I try to implement it in another way:
First, find all possible keyboard patterns,
Then, detect all possible keyboard patterns in passwords.
"""
import os
from typing import Dict, List, Tuple, Any

from .my_dfs_lib import gen_dtree, extract, post_parse4case_free

path_kbd_allow = os.path.join(os.path.dirname(__file__), "kbd.allow")
path_kbd_ignore = os.path.join(os.path.dirname(__file__), "kbd.ignore")


def kbd_filter(candidate: str):
    """
    whether candidate follows keyboard pattern
    :param candidate:
    :return: is_kbd_pattern
    """
    if candidate.isdigit() or candidate.isalpha() or not any([(c.isdigit() or c.isalpha()) for c in candidate]):
        return False
    return True


def load_kbd_allow(file: str) -> Dict[str, int]:
    """
    words in this set will be treated as l33t and will not be parsed again
    :return: set of l33ts
    """
    if not os.path.exists(file):
        return {}
    fd = open(file, "r")
    allow = {}
    for line in fd:
        allow[line.strip("\r\n")] = 1
    fd.close()
    return allow


def load_kbd_ign(file: str) -> Dict[str, int]:
    """
    l33t.ignore, one instance per line
    :return: set of ignored l33ts
    """
    if not os.path.exists(file):
        return {}
    fd = open(file, "r")
    ign = {}
    for line in fd:
        ign[line.strip("\r\n")] = 1
    return ign


def init(allow: str, ign: str, end: str = "\x03"):
    kbd_allow = load_kbd_allow(allow)
    kbd_ign = load_kbd_ign(ign)
    kbd_dtree, max_kbd_len = gen_dtree(kbd_allow, end=end)
    return kbd_dtree, max_kbd_len, kbd_allow, kbd_ign


def kbd_detection(pwd: str, kbd_dtree: Dict, max_kbd_len: int, end: str = "\x03"):
    raw_kbd_lst = extract(dtree=kbd_dtree, pwd=pwd, max_kbd_len=max_kbd_len, end=end)
    sec_lst, kbd_lst = post_parse4case_free(raw_kbd_lst, pwd=pwd, tag="K")
    return sec_lst, kbd_lst


def kbd_detection4seclist(sections: List[Tuple[str, Any]],
                          kbd_dtree: Dict, max_kbd_len: int, end: str = "\x03"):
    sec_list = []
    kbd_list = []
    for sec, tag in sections:
        if tag is not None:
            sec_list.append((sec, tag))
        else:
            sub_sec_list, sub_kbd_list = kbd_detection(sec, kbd_dtree, max_kbd_len, end)
            sec_list.extend(sub_sec_list)
            kbd_list.extend(sub_kbd_list)
    return sec_list, kbd_list


def test():
    kbd_dtree, max_kbd_len, kbd_allow, kbd_ign = init(path_kbd_allow, path_kbd_ignore)
    print(kbd_detection("1Q2w3e4rhh,", kbd_dtree, max_kbd_len))
    pass


if __name__ == '__main__':
    # print(kbd_detection("1Q2w3e4rhh"))
    test()
    pass
