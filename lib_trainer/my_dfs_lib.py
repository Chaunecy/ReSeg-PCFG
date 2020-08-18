"""
This is a library for Depth First Search and Dict Tree Compare.
"""
from typing import Dict, Any, List, Tuple


def extract(dtree: Dict[str, Any], pwd: str, min_kbd_len: int = 4, end: str = "\x03"):
    """
    find overlaps between dtree and pwd.
    Find the longest match first
    :param min_kbd_len:
    :param end: the end symbol of search a dict tree
    :param dtree:
    :param pwd:
    :return: List[(start idx, length of segment, whether this segment is tagged or not)]
    """
    kbd_list = []
    a_kbd = ""
    lower_pwd = pwd.lower()
    len_pwd = len(pwd)
    i = 0
    cur_i = i
    len_kbd = 0
    bk_dtree = dtree
    while i < len_pwd and cur_i < len_pwd:
        c = lower_pwd[cur_i]
        if c in bk_dtree:
            a_kbd += c
            bk_dtree = bk_dtree[c]
            if end in bk_dtree:
                add_a_kbd = ""
                bak_add_a_kbd = ""
                for addi in range(cur_i + i, min(cur_i + min_kbd_len - len(a_kbd) + 1, len_pwd)):
                    addc = lower_pwd[addi]
                    if addc not in bk_dtree:
                        break
                    bk_dtree = bk_dtree[addc]
                    add_a_kbd += addc
                    if end in bk_dtree:
                        bak_add_a_kbd = add_a_kbd
                    pass
                if bak_add_a_kbd != "":
                    a_kbd += bak_add_a_kbd
                    cur_i += len(bak_add_a_kbd)
                len_a_kbd = len(a_kbd)
                # start index of this kbd, length of this kbd, this is kbd
                kbd_list.append((cur_i - len_a_kbd + 1, len_a_kbd, True))
                len_kbd += len_a_kbd
                i += len_a_kbd
                cur_i = i
                a_kbd = ""
                bk_dtree = dtree
            cur_i += 1
        else:
            i += 1
            cur_i = i
            a_kbd = ""
            bk_dtree = dtree
    if len_kbd == len_pwd:
        return kbd_list
    elif len(kbd_list) == 0:
        return [(0, len_pwd, False)]
    else:
        # n_list keeps all nodes that a kbd starts or ends
        # is_kbd_set keeps that whether the node is the start of kbd
        n_list = set()
        is_kbd_set = set()
        n_list.add(0)
        # i: start idx of kbd, kl: length of kbd
        for i, kl, is_kbd in kbd_list:
            n_list.add(i)
            n_list.add(i + kl)
            is_kbd_set.add(i)
        n_list.add(len_pwd)
        n_list = sorted(n_list)
        n_kbd_list = []
        for n_i, pwd_i in enumerate(n_list[:-1]):
            n_kbd_list.append((pwd_i, n_list[n_i + 1] - pwd_i, pwd_i in is_kbd_set))
        n_kbd_list = sorted(n_kbd_list, key=lambda x: x[0])
        return n_kbd_list
    pass


def post_parse4case_free(res: List[Tuple[int, int, bool]], pwd: str, tag: str):
    section_list = []
    tag_list = []
    for idx, len_seg, is_tagged in res:
        seg = pwd[idx:idx + len_seg]
        if is_tagged:
            section_list.append((seg, f"{tag}{len_seg}"))
            tag_list.append(seg)
        else:
            section_list.append((seg, None))
        pass
    return section_list, tag_list
    pass


def gen_dtree(entries: Dict[str, int], end: str = "\x03"):
    """
    get dict tree of keyboard from a dict
    :param entries:
    :param end: End of a keyboard pattern
    :return:
    """
    lst = sorted(entries.keys(), key=lambda x: len(x), reverse=True)
    if len(lst) == 0:
        return {}
    # min_len = len(lst[-1])
    # max_len = len(lst[0])
    dtree = {}
    for entry in entries:
        tmp_dtree = dtree
        for c in entry:
            if c not in tmp_dtree:
                tmp_dtree[c] = {}
            tmp_dtree = tmp_dtree[c]
        tmp_dtree[end] = True
    return dtree
    pass


class KbdPtnDetection:
    def __init__(self):
        self.__kbd_dict = {}
        pass

    pass


def main():
    d = {
        "1q2w3e4r": 1,
        "q1w2e3r4": 1,
        "1qazxsw2": 1,
    }
    dtree = gen_dtree(d)
    pwd = "1q2w3e4rhh"
    kbd_list = extract(dtree, pwd)
    sec_list, kbds = post_parse4case_free(kbd_list, pwd, "K")
    print(sec_list)
    pass


if __name__ == '__main__':
    main()
