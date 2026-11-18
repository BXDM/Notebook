"""从 Wikimedia Commons 抓人像素材（开放授权），用于标定（瘦脸阈值等）。每张记录出处、作者、授权。

    python tools/fetch_commons.py 输出目录 --category "Portrait photographs of men" --search "portrait man" --max 200

只取 jpg、宽高都 ≥ 1000 的照片，下载 1600 像素宽的缩略图；黑白照片（彩度很低）丢弃。
出处清单写在 输出目录/manifest.tsv（文件名、页面、作者、授权）。已下载的跳过，可以反复运行补充。
"""
import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

API = "https://commons.wikimedia.org/w/api.php"
UA = "photo-retouching-research/0.1 (personal non-commercial calibration)"


def get(url, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def query(params):
    base = {"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime",
            "iiurlwidth": "1600"}
    cont = {}
    while True:
        p = {**base, **params, **cont}
        data = json.loads(get(API + "?" + urllib.parse.urlencode(p)))
        for page in data.get("query", {}).get("pages", {}).values():
            yield page
        if "continue" not in data:
            return
        cont = data["continue"]


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip().replace("\t", " ").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--category", action="append", default=[])
    ap.add_argument("--search", action="append", default=[])
    ap.add_argument("--max", type=int, default=200, help="每个分类/搜索最多下载几张")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = os.path.join(a.out, "manifest.tsv")
    have = set()
    if os.path.exists(man):
        have = {l.split("\t")[0] for l in open(man, encoding="utf-8")}
    fh = open(man, "a", encoding="utf-8")
    sources = [("category", c) for c in a.category] + [("search", s) for s in a.search]
    for kind, q in sources:
        if kind == "category":
            params = {"generator": "categorymembers", "gcmtitle": "Category:" + q, "gcmtype": "file", "gcmlimit": "50"}
        else:
            params = {"generator": "search", "gsrsearch": q + " filetype:bitmap", "gsrnamespace": "6", "gsrlimit": "50"}
        n = 0
        for page in query(params):
            if n >= a.max:
                break
            ii = (page.get("imageinfo") or [{}])[0]
            if ii.get("mime") != "image/jpeg" or min(ii.get("width", 0), ii.get("height", 0)) < 1000:
                continue
            name = re.sub(r"[^\w.-]+", "_", page["title"][5:])[:120]
            if name in have:
                continue
            try:
                buf = np.frombuffer(get(ii["thumburl"]), np.uint8)
            except Exception as e:
                print("下载失败", name, e)
                continue
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is None:
                continue
            lab = cv2.cvtColor(cv2.resize(im, (200, 200)), cv2.COLOR_BGR2Lab).astype(np.float32)
            if np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).mean() < 4:      # 黑白照片
                continue
            cv2.imwrite(os.path.join(a.out, name if name.lower().endswith(".jpg") else name + ".jpg"), im,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            md = ii.get("extmetadata", {})
            fh.write("\t".join([name, ii.get("descriptionurl", ""), strip_html(md.get("Artist", {}).get("value")),
                                md.get("LicenseShortName", {}).get("value", "")]) + "\n")
            fh.flush()
            have.add(name)
            n += 1
            time.sleep(0.5)
        print(f"{kind} {q}: {n} 张", flush=True)


if __name__ == "__main__":
    main()
