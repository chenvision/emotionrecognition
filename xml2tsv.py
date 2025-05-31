# === xml2tsv.py  ==========================================
"""
Convert NLPCC-2014 Emotion Classification XMLs→TSV
Usage:
    python xml2tsv.py --dir evtestdata1 --out_dir data --ratio 0.8 0.1 0.1
If a *Testing*-xml 有标签，它将整体写入 test.tsv；如无标签，则从训练集随机切分 dev/test。
"""
from __future__ import annotations
import argparse, random, csv, pathlib, xml.etree.ElementTree as ET

POS_EMOS = {"happiness", "like", "surprise"}
NEG_EMOS = {"disgust", "sadness", "anger", "fear"}

# ---------- 解析标签 -------------------------------------------------- #
def _map_label(emotion: str | None) -> int | None:
    if not emotion or emotion.lower() in {"none", ""}:
        return None
    e = emotion.lower()
    if e in POS_EMOS:
        return 1
    if e in NEG_EMOS:
        return 0
    return None

def _decode_weibo(weibo) -> int | None:
    # 1) 先看 weibo 层
    for k in ("emotion-type1", "emotion-type2"):
        lbl = _map_label(weibo.attrib.get(k))
        if lbl is not None:
            return lbl
    # 2) 再看 sentence 层
    for sent in weibo.iter("sentence"):
        for k in ("emotion-1-type", "emotion-2-type"):
            lbl = _map_label(sent.attrib.get(k))
            if lbl is not None:
                return lbl
    return None

def load_xml(path):
    recs = []
    for weibo in ET.parse(path).iter("weibo"):
        label = _decode_weibo(weibo)
        if label is None:
            continue                     # 丢弃无标签样本
        text = " ".join(
            (seg.text or "").strip()
            for seg in weibo.iter("sentence")
            if seg.text and seg.text.strip()
        ).replace("\t", " ")
        if text:
            recs.append((text, label))
    return recs

def dump_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t").writerows(rows)

# ---------- 主流程 ---------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default='./evtestdata1', help="包含 4 个 XML 的文件夹")
    ap.add_argument("--out_dir", default="data", help="输出 TSV 目录")
    ap.add_argument("--ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN","DEV","TEST"),
                    help="若测试集 XML 无标签时，按比例切分")
    args = ap.parse_args()

    d = pathlib.Path(args.dir)
    assert d.exists(), f"目录不存在: {d}"

    # 按文件名识别：Classification vs Expression
    train_xml = None
    test_xml  = None
    for f in d.glob("*.xml"):
        name = f.name.lower()
        if "emotion classification" in name and "training" in name:
            train_xml = f
        elif "emotion classification" in name and "testing" in name:
            test_xml = f

    if not train_xml:
        raise RuntimeError("未找到 *Training data for Emotion Classification.xml*")

    print(f"[+] 读取训练集  {train_xml}")
    train_records = load_xml(train_xml)
    print(f"    样本数: {len(train_records)}")

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(exist_ok=True, parents=True)

    # 解析测试集（如含标签）
    test_records = []
    if test_xml:
        print(f"[+] 读取测试集  {test_xml}")
        test_records = load_xml(test_xml)
        print(f"    样本数(含标签): {len(test_records)}")

    # 若测试集有标签，则不再随机切分 test；否则按 ratio 切 dev/test
    if test_records:
        random.shuffle(train_records)
        dev_size = int(len(train_records)*args.ratio[1])
        dev_records = train_records[:dev_size]
        train_records = train_records[dev_size:]
    else:
        random.shuffle(train_records)
        n = len(train_records)
        i1 = int(n*args.ratio[0])
        i2 = i1 + int(n*args.ratio[1])
        train_records, dev_records, test_records = (
            train_records[:i1], train_records[i1:i2], train_records[i2:]
        )

    dump_tsv(out_dir / "train.tsv", train_records)
    dump_tsv(out_dir / "dev.tsv",   dev_records)
    dump_tsv(out_dir / "test.tsv",  test_records)
    print(f"[✓] 已写入 TSV → {out_dir}")
    print(f"    train / dev / test = {len(train_records)} / {len(dev_records)} / {len(test_records)}")

if __name__ == "__main__":
    main()
