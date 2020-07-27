import re


class RuleCorrector:
    def __init__(self, rule_path: str, josa_path: str = None):
        with open(rule_path, 'r', encoding='utf-8') as file:
            rules = file.read().split('\n')

        self.rule_dict, self.rule_pattern = self.construct_rule_dict(rules)

        with open(josa_path, 'r', encoding='utf-8') as file:
            josa = file.read().split('\n')
        self.josa_pattern = self.construct_josa_pattern(josa)

    def correct(self, text: str):
        if not text or text.__class__ != str:
            return text
        else:
            text = self.correct_rule(text)
            text = self.correct_josa(text)
            return text

    def correct_josa(self, text: str):
        match = re.findall(pattern=self.josa_pattern, string=text)
        if match:
            for m in match:
                text = text.replace(m, m[1:])
        return text

    def correct_rule(self, text: str):
        match = re.findall(pattern=self.rule_pattern, string=text)
        if match:
            for m in match:
                text = text.replace(m, self.rule_dict[m.replace(' ', '')])
        return text

    @staticmethod
    def construct_josa_pattern(josa_list: list):
        pattern = []
        for d in josa_list:
            if d.endswith(('?', '!', '.', ',')):
                pattern.append(f'\\s{d[:-1]}\\{d[-1]}')
            else:
                pattern.append(f'\\s{d}\\s')
        pattern = '|'.join(pattern)
        return pattern

    @staticmethod
    def construct_rule_dict(rule_list: list):
        rule_dict = {}
        pattern = []
        for r in rule_list:
            word, repl = r.split('|')
            pattern.append('\\s?'.join(list(word.replace(' ', ''))))
            rule_dict[word.replace(' ', '')] = repl
        pattern = '|'.join(sorted(pattern, key=lambda x: len(x), reverse=True))
        return rule_dict, pattern
