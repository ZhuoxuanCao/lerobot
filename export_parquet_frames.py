# import pyarrow.parquet as pq
# import io, os
# from PIL import Image
# import imageio.v2 as imageio

# # 你的 parquet 文件路径
# parquet_path = "/home/cao/.cache/huggingface/lerobot/root/so101_stack_blue_on_green/data/chunk-000/file-000.parquet"

# # 输出文件夹
# output_dir = "./preview_output"
# os.makedirs(output_dir, exist_ok=True)

# # 读取 parquet
# table = pq.read_table(parquet_path)
# df = table.to_pandas()

# print("列名:", df.columns)
# print("数据行数:", len(df))

# # 只导出前10帧
# num_frames = min(10, len(df))

# # 分别存 top 和 hand 相机
# for cam in ["top", "hand"]:
#     cam_dir = os.path.join(output_dir, cam)
#     os.makedirs(cam_dir, exist_ok=True)

#     images = []
#     for i in range(num_frames):
#         img_bytes = df.iloc[i]["images"][cam]  # 注意: 每行是一个 dict-like
#         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#         img_path = os.path.join(cam_dir, f"{i:03d}.jpg")
#         img.save(img_path)
#         images.append(imageio.imread(img_path))

#     # 拼接成 mp4
#     mp4_path = os.path.join(output_dir, f"{cam}_preview.mp4")
#     imageio.mimsave(mp4_path, images, fps=5)  # fps=5 方便预览
#     print(f"{cam} 前{num_frames}帧已导出到 {cam_dir} 并生成视频 {mp4_path}")
import os, io, sys
import pyarrow.parquet as pq
from PIL import Image
import imageio.v2 as imageio

# ==== 配置你的 parquet 路径 ====
PARQUET_PATH = "/home/cao/.cache/huggingface/lerobot/root/so101_stack_blue_on_green/data/chunk-000/file-000.parquet"
OUT_DIR = "./preview_output"
OS_MKDIR_MODE = 0o755

# --------- 工具函数 ---------
def try_get(d, path_list):
    """
    在 dict/嵌套结构中依次尝试多条路径，返回第一个命中的值。
    path 语法: ["cameras.top.image"] -> d["cameras"]["top"]["image"]
               ["images.top", "top.image"] -> 依次尝试
    """
    for path in path_list:
        cur = d
        ok = True
        for key in path.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return None

def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

# --------- 读取 parquet + 探测结构 ---------
table = pq.read_table(PARQUET_PATH)
cols = [c for c in table.column_names]
print("列名:", cols)

# 先只把 observation.state 一列转成 Python（避免 pandas 丢失嵌套结构）
if "observation.state" not in cols:
    print("没有找到 'observation.state' 列，请把 schema 打印出来排查：")
    print(table.schema)
    sys.exit(1)

states = table.column("observation.state").to_pylist()
n = len(states)
print(f"共有 {n} 行")

if n == 0:
    print("文件里没有数据行。")
    sys.exit(0)

# 打印第 1 行的 key 布局，帮助你确认结构
sample_state = states[0]
print("\n样例 observation.state[0] 类型:", type(sample_state))
print("长度:", len(sample_state))
for i, elem in enumerate(sample_state):
    print(f"  [{i}] 类型={type(elem)}, 内容摘要={str(elem)[:200]}")

if "cameras" in sample_state and isinstance(sample_state["cameras"], dict):
    print("cameras 子 keys：", list(sample_state["cameras"].keys()))
    for cam_k, cam_v in sample_state["cameras"].items():
        if isinstance(cam_v, dict):
            print(f"  cameras['{cam_k}'] keys:", list(cam_v.keys()))
else:
    # 可能直接就是 top/hand 两个键
    for k, v in sample_state.items():
        if isinstance(v, dict):
            print(f"state['{k}'] keys:", list(v.keys()))

# --------- 尝试抽取并导出前 10 帧 ---------
ensure_dir(OUT_DIR)

cams = ["top", "hand"]   # 你当前就两路
export_frames = 10
fps_preview = 5          # 预览视频帧率

# 常见的几种可能路径（按顺序尝试）
PATH_CANDIDATES = {
    "top": [
        "cameras.top.image",
        "images.top",
        "top.image",
        "top",            # 有些版本直接就是 bytes
    ],
    "hand": [
        "cameras.hand.image",
        "images.hand",
        "hand.image",
        "hand",
    ],
}

any_exported = False

for cam in cams:
    cam_dir = os.path.join(OUT_DIR, cam)
    ensure_dir(cam_dir)
    frames = []
    saved = 0

    for i in range(min(export_frames, n)):
        state = states[i]

        # 先拿到 bytes（JPEG/PNG 编码），不同版本字段位置可能不同
        img_bytes = try_get(state, PATH_CANDIDATES[cam])

        # 某些版本把图像封成 dict { 'data': bytes, 'format': 'jpeg' }
        if isinstance(img_bytes, dict) and "data" in img_bytes:
            img_bytes = img_bytes["data"]

        if img_bytes is None:
            # 再兜底查找：把所有 value 里较大的 bytes 当作候选（启发式）
            candidate = None
            for k, v in state.items():
                if isinstance(v, dict) and cam in k.lower():
                    # 在与相机名相关的子 dict 里找 'image' 或 'data'
                    candidate = v.get("image") or v.get("data")
                    if candidate is not None:
                        img_bytes = candidate
                        break

        if img_bytes is None or not isinstance(img_bytes, (bytes, bytearray)):
            # 这帧没找到，跳过
            continue

        try:
            im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"[{cam}] 第 {i} 帧解码失败：{e}")
            continue

        jpg_path = os.path.join(cam_dir, f"{i:03d}.jpg")
        im.save(jpg_path)
        frames.append(imageio.imread(jpg_path))
        saved += 1

    if saved > 0:
        mp4_path = os.path.join(OUT_DIR, f"{cam}_preview.mp4")
        imageio.mimsave(mp4_path, frames, fps=fps_preview)
        print(f"[{cam}] 导出 {saved} 帧到 {cam_dir}，并生成视频 {mp4_path}")
        any_exported = True
    else:
        print(f"[{cam}] 没能在前 {export_frames} 行里找到可解码的图像字节。")

if not any_exported:
    print("\n没有成功解出任何图像！请把上面打印的 sample_state 结构发我，我帮你对症修改路径。")
