"""
Keyboard Patterns Detection implemented is insufficient to parse q1w2e3r4 or so.
Here I try to implement it in another way:
First, find all possible keyboard patterns,
Then, detect all possible keyboard patterns in passwords.
"""
import os
from typing import Dict

from .my_dfs_lib import gen_dtree, extract, post_parse4case_free

path_kbd_allow = os.path.join(os.path.dirname(__file__), "kbd.allow")
path_kbd_ignore = os.path.join(os.path.dirname(__file__), "kbd.ignore")


def load_kbd_allow() -> Dict[str, int]:
    """
    words in this set will be treated as l33t and will not be parsed again
    :return: set of l33ts
    """
    if not os.path.exists(path_kbd_allow):
        return {}
    fd = open(path_kbd_allow, "r")
    allow = {}
    for line in fd:
        allow[line.strip("\r\n")] = 1
    fd.close()
    return allow


def load_kbd_ign() -> Dict[str, int]:
    """
    l33t.ignore, one instance per line
    :return: set of ignored l33ts
    """
    if not os.path.exists(path_kbd_ignore):
        return {}
    fd = open(path_kbd_ignore, "r")
    ign = {}
    for line in fd:
        ign[line.strip("\r\n")] = 1
    return ign


kbd_allow = load_kbd_allow()
print(kbd_allow)
kbd_ign = load_kbd_ign()
end = "\x03"
kbd_dtree = gen_dtree(kbd_allow, end=end)


def kbd_detection(pwd: str):
    raw_kbd_lst = extract(dtree=kbd_dtree, pwd=pwd, min_kbd_len=4, end=end)
    sec_lst, kbd_lst = post_parse4case_free(raw_kbd_lst, pwd=pwd, tag="K")
    return sec_lst, kbd_lst


if __name__ == '__main__':
    print(kbd_detection("1q2w3e4rhh"))
    pass
