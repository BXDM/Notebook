"""在 Modal 云端 GPU 上并行跑整个修图流程（开发阶段默认用这个，不占本机时间）。

    python cloud.py main  ~/PycharmProjects/sample/人像 -o out/          # 人像精修，每张一个容器并行
    python cloud.py auto  ~/PycharmProjects/sample/夜景 -o out/ --grade none
    python cloud.py main  照片目录 -o out/ --project <另一份代码目录>      # 用别的版本（比如 git archive 出的旧版）对比
    python cloud.py photo 照片目录 -o out/                                  # 2.0 统一入口 photo.py

容器里装与本机相同的依赖，带上 models/（不含 1.3 GB 的 sapiens），照片上传、结果回传，不在云端保存。
除 -o、--project 外的参数原样传给 main.py / auto.py / photo.py。
"""
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def build_app(project):
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("libgl1", "libglib2.0-0", "fonts-noto-cjk")
        .pip_install("torch==2.8.0", "numpy==2.2.6", "opencv-contrib-python-headless==4.12.0.88",
                     "mediapipe==1.2.3", "onnxruntime==1.23.2", "einops==0.8.2", "Pillow==11.3.0")
        .apt_install("libegl1", "libgles2")
        .add_local_dir(project, "/proj", ignore=["models/sapiens", "**/__pycache__", "output", "debug"])
    )
    app = modal.App("photo-retouch-run")

    @app.function(image=image, gpu=["L4", "A10G", "T4"], cpu=8.0, memory=16384, timeout=1800, serialized=True)
    def run_one(tool: str, name: str, data: bytes, args: list) -> tuple:
        import glob
        import tempfile
        d = tempfile.mkdtemp()
        os.makedirs(f"{d}/in")
        open(f"{d}/in/{name}", "wb").write(data)
        t = time.time()
        p = subprocess.run([sys.executable, f"{tool}.py", f"{d}/in", "-o", f"{d}/out", *args],
                           cwd="/proj", capture_output=True, text=True)
        log = "\n".join(l for l in (p.stdout + p.stderr).splitlines()
                        if l.startswith(os.path.splitext(name)[0]) or "Error" in l or "失败" in l)
        outs = {os.path.basename(f): open(f, "rb").read() for f in glob.glob(f"{d}/out/*")}
        # --debug 输出（蒙版等）也带回来，放在输出目录的 debug/<图名>/ 下
        for f in glob.glob(f"/proj/debug/*/*"):
            outs["debug/" + os.path.relpath(f, "/proj/debug")] = open(f, "rb").read()
        return name, outs, log or (p.stdout + p.stderr)[-2000:], time.time() - t

    return app, run_one


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("main", "auto", "photo"):
        sys.exit(__doc__)
    tool, src, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    out, project = "output", HERE
    args = []
    i = 0
    while i < len(rest):
        if rest[i] in ("-o", "--output"):
            out = rest[i + 1]; i += 2
        elif rest[i] == "--project":
            project = os.path.abspath(rest[i + 1]); i += 2
        else:
            args.append(rest[i]); i += 1
    files = [os.path.join(src, f) for f in sorted(os.listdir(src)) if f.lower().endswith(EXTS)] \
        if os.path.isdir(src) else [src]
    os.makedirs(out, exist_ok=True)
    app, run_one = build_app(project)
    t0 = time.time()
    with modal.enable_output(), app.run():
        jobs = [(tool, os.path.basename(f), open(f, "rb").read(), args) for f in files]
        for name, outs, log, dt in run_one.starmap(jobs, order_outputs=False):
            for fn, b in outs.items():
                os.makedirs(os.path.dirname(os.path.join(out, fn)), exist_ok=True)
                open(os.path.join(out, fn), "wb").write(b)
            print(f"[{dt:.0f}s] {log.strip()}", flush=True)
    print(f"共 {len(files)} 张，总耗时 {time.time() - t0:.0f}s（并行）")


if __name__ == "__main__":
    main()
