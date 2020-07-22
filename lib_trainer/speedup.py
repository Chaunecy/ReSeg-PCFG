import os
import pickle
from typing import Dict, List

path_found_l33t = os.path.join(os.path.dirname(__file__), "l33t.found")
path_ignore_l33t = os.path.join(os.path.dirname(__file__), "l33t.ignore")
path_found_mixing = os.path.join(os.path.dirname(__file__), "mixing.found")


def load_l33t_found():
    pass


def load_l33t_ign():
    pass


def save_l33t_found(l33ts: Dict[str, int]) -> None:
    fd = open(path_found_l33t, "wb")
    l33ts = set(l33ts.keys())
    pickle.dump(l33ts, fd)
    pass


def load_mixing() -> Dict[str, set]:
    if not os.path.exists(path_found_mixing):
        return {}
    fd = open(path_found_mixing, "rb")
    mixing_dict = pickle.load(fd)  # type: Dict[str, set]
    fd.close()
    return mixing_dict


def save_mixing(mixing_dict: Dict) -> None:
    fd = open(path_found_mixing, "wb")
    pickle.dump(mixing_dict, fd)
    fd.close()
