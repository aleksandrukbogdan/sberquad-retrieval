import re

_TOKEN = re.compile(r"[а-яa-z0-9]+")


def normalize(text):
    return text.lower().replace("ё", "е")


def tokenize(text):
    return _TOKEN.findall(normalize(text))
