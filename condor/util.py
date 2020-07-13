# -*- coding:utf-8 -*-

def generate_x_y(sent):
    x, y = [], []
    x_ = list(sent + '#')
    for idx, s in enumerate(x_[:-1]):
        if s != ' ':
            x.append(s)
            if x_[idx+1] == ' ':
              y.append(1)
            else:
                y.append(0)
    return (x, y)


def decode(char_list, preds):
    result = ''
    for char, l in zip(char_list, preds):
        if l == 0:
            result += char
        elif l == 1:
            result += char + ' '
    return result
