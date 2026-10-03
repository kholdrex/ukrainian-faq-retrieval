import re

APOSTROPHES = re.compile(r"[’ʼ`´‘]")
WORD = re.compile(r"[\w']+", re.UNICODE)


def normalise(text: str) -> str:
    return APOSTROPHES.sub("'", text).lower()


def words(text: str) -> list[str]:
    return WORD.findall(normalise(text))
