"""把论文 PDF 抽取成纯文本，便于检索/引用 (需要 ``pypdf``)。

用法::

    python scripts/pdf_to_text.py "paper.pdf" -o paper.txt
    # 不带参数时自动寻找工作目录下的第一个 .pdf
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="论文 PDF -> 文本")
    ap.add_argument("pdf", nargs="?", default=None)
    ap.add_argument("-o", "--out", default="paper.txt")
    args = ap.parse_args()

    pdf = args.pdf
    if pdf is None:
        found = sorted(glob.glob("*.pdf"))
        if not found:
            print("未找到 PDF，请显式传入路径。", file=sys.stderr)
            return 1
        pdf = found[0]

    try:
        from pypdf import PdfReader
    except ImportError:
        print("需要 pypdf: pip install pypdf", file=sys.stderr)
        return 1

    reader = PdfReader(pdf)
    chunks = []
    for i, page in enumerate(reader.pages):
        chunks.append(f"===== PAGE {i + 1} =====\n{page.extract_text() or ''}")
    text = "\n".join(chunks)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"{os.path.basename(pdf)}: {len(reader.pages)} 页, "
          f"{len(text)} 字符 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
