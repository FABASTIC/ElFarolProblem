import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KNOWN = {"ref", "right", "rho", "rangle", "rbrace", "raggedright", "renewcommand", "rule", "resizebox", "rm", "relax"}


def main():
    with open(os.path.join(HERE, "main.tex"), encoding="utf-8") as f:
        tex = f.read()
    with open(os.path.join(HERE, "results.tex"), encoding="utf-8") as f:
        defined = set(re.findall(r"\\newcommand\{\\(r[A-Za-z]+)\}", f.read()))
    used = set(re.findall(r"\\(r[a-z][A-Za-z]*)", tex))
    missing = sorted(u for u in used if u not in defined and u not in KNOWN)
    for name in missing:
        line = next(i for i, text in enumerate(tex.splitlines(), 1) if "\\" + name in text)
        print(f"undefined \\{name} first used on line {line}")
    print(f"{len(used & defined)} result macros used, {len(missing)} undefined")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
