# run_pipeline.py
import subprocess
import sys

scripts = [
    "preprocessing.py",
    "training.py --epochs=50",
    "evaluation.py --dataset=test"
]

for script in scripts:
    print(f"⏳ 开始执行: {script}")
    result = subprocess.run(
        [sys.executable] + script.split(),
        check=True,  # 前一个失败则停止后续执行
        text=True
    )
    if result.returncode != 0:
        print(f"❌ 执行失败: {script}", file=sys.stderr)
        sys.exit(1)
    print(f"✅ 完成: {script}\n")

print("🎉 所有任务完成!")