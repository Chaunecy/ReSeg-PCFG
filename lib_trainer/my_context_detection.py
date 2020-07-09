ld = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
context_dict = [
    ";p",
    ":p",
    "*0*",
    "#1",
    "No.1",
    "no.1",
    "No.",
    "i<3",
    "I<3",
    "<3",
    "Mr.",
    "mr.",
    "MR.",
    "MS.",
    "Ms.",
    "ms.",
    "Mz.",
    "mz.",
    "MZ.",
    "St.",
    "st.",
    "Dr.",
    "dr.",

]


def detect_context(section: str):
    ret = []
    len_sec = len(section)

    for context in context_dict:
        len_x = len(context)
        idx = section.find(context)
        if idx < 0:
            continue
        if context[-1] in ld and idx + len_x < len_sec and section[idx + len_x] in ld:
            continue
        if idx > 0 and (context[0] in ld) and (section[idx - 1] in ld):
            continue
        if idx > 0:
            ret.append((section[:idx], None))
        ret.append((section[idx: idx + len_x], "X1"))
        if idx + len_x < len_sec:
            ret.append((section[idx + len_x:], None))
        return ret, [context]
    return [(section, None)], []


def detect_context_sections(sections_list):
    parsed_sections = []
    contexts = []
    for section, tag in sections_list:
        if len(section) < 2:
            parsed_sections.append((section, None))
        else:
            parsed_s, cont_s = detect_context(section)
            parsed_sections.extend(parsed_s)
            contexts.extend(cont_s)
    return parsed_sections, contexts


pass
