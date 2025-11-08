import json
import ast
import base64
import openai
from openai import OpenAI
import re
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
import multiprocessing as mp
import os

# ===== 从配置文件加载 OpenAI 参数 =====
def load_api_config(config_path="../api_config.json"):
    """从 JSON 文件加载 api_key 和 api_base"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "api_key" not in cfg or "api_base" not in cfg:
        raise ValueError("配置文件必须包含 'api_key' 和 'api_base' 字段")
    return cfg["api_base"], cfg["api_key"]

# ===== 加载配置 =====
api_base, api_key = load_api_config("../api_config.json")

openai.api_base = api_base
openai.api_key = api_key


client = OpenAI(api_key=openai.api_key, base_url=openai.api_base)

# ==============================
# --- Paths relative to this file, so CWD changes don't break things ---
ROOT = Path(__file__).resolve().parent
INPUT_JSONL = ROOT / "ground_truths" / "data.jsonl"
OUTPUT_DIR = ROOT / "single_step_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # <-- ensure directory exists
NUM_WORKERS = 8

# 多模型批处理
MODEL_NAMES = [
    "o4-mini-2025-04-16", 
    "claude-sonnet-4-20250514-thinking",
    "gpt-5-2025-08-07",
    "gemini-2.5-pro",
    "claude-3-7-sonnet-20250219-thinking",
    "gpt-4o-2024-11-20",
]
# ==============================

def extract_boxed_actions(text):
    lst = re.findall(r'\\+boxed\{([^}]*)\}', text)
    filtered = []
    for act in lst:
        if not act:
            continue
        filtered.append(act)
    return filtered

def smart_split(s, sep=','):
    res = []
    buf = ''
    level = 0
    for c in s:
        if c == '[':
            level += 1
        elif c == ']':
            level -= 1
        if c == sep and level == 0:
            res.append(buf.strip())
            buf = ''
        else:
            buf += c
    if buf:
        res.append(buf.strip())
    return res

def parse_action_seq(raw_action_seq):
    def strip_quotes(s):
        return re.sub(r'^[\'"“”‘’]+|[\'"“”‘’]+$', '', s)
    actions = []
    for act in smart_split(raw_action_seq):
        act = act.strip()
        if not act:
            continue
        m = re.match(r'click\[(.*?)\]', act, re.I)
        if m:
            id_ = strip_quotes(m.group(1).strip())
            actions.append({"action": "click", "id": id_})
            continue
        m = re.match(r'enter\[(.*?)\]\[(.*?)\]', act, re.I)
        if m:
            id_ = strip_quotes(m.group(1).strip())
            val = strip_quotes(m.group(2).strip())
            actions.append({"action": "input", "id": id_, "input_val": val})
            continue
        m = re.match(r'select\[(.*?)\]\[(.*?)\]', act, re.I)
        if m:
            id_ = strip_quotes(m.group(1).strip())
            val = strip_quotes(m.group(2).strip())
            actions.append({"action": "select", "id": id_, "value": val})
            continue
        return None
    return actions

def find_node_by_id_for_prompt(domtree, node_id):
    if not domtree:
        return None
    if isinstance(domtree, str):
        try:
            domtree = ast.literal_eval(domtree)
        except Exception:
            return None
    if str(domtree.get('id')) == str(node_id):
        return {
            "id": domtree.get("id"),
            "tag": domtree.get("tag"),
            "attrs": domtree.get("attrs"),
            "visible_text": domtree.get("visible_text"),
        }
    for child in domtree.get('children', []):
        res = find_node_by_id_for_prompt(child, node_id)
        if res:
            return res
    return None

def get_response(message, model_name):
    response = client.chat.completions.create(
             model=model_name,
             messages=message,
             max_completion_tokens=10000,
             timeout=600
         )
    return response

def get_op_actions_by_model_no_history(img_path, domtree, model_name):
    img_path = (ROOT / "ground_truths" / img_path).resolve()
    with open(img_path, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    prompt_text = f"""你是一位互动网页助手，我现在希望检测这个网页中所有可互动按钮是否都可以正常工作，例如，如果页面中有搜索框，请搜索一个合理的内容，并点击确认，期望页面会发生变化。请注意，如果多个互动组件几乎一模一样，请只从中选一个。比如页面中有多个相似的条目，每个条目都有“编辑”按钮，请只选择一次。
        现在的页面状态处于检测过程中的一环，我会发送给你现在已经点击了哪些组件。如果你发现之前点击过别的图片，请专注于这次发给你的图片与之前的有何不同，例如弹出了某个新窗口，请一定只选择新部分中的互动组件。如果你发现这次给你的图片与历史中的某次图片几乎一致，例如按钮全部一致，只有微小的文字差异；那请直接回复“本页面所有操作已完成”
        注意，如果两个互动按钮间并非连续关系，比如在同页面下的两个点击按钮，请每个boxed中只包含一个，将它们分开。如果是连续关系，比如输入多个内容并点击确认，请把它们放在同一个boxed内。将你的答案分别用latex的\\boxed{{}}包裹。动作格式：click[id]表示点击，enter[id][text]表示输入内容，select[id][text]表示选择，每个操作中间用逗号分隔。
        【注意】：请只返回答案！不要有多余的东西！
        页面信息:\n{domtree}\n,"""
    image_content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}}
    ]
    message = [{"role": "user", "content": image_content}]
    response = get_response(message, model_name)
    output = str(response.choices[0].message)
    boxed_raws = extract_boxed_actions(output)
    actions_per_seq = [r for r in (parse_action_seq(bx) for bx in boxed_raws) if r is not None]
    actions_parrel = []
    for actlist in actions_per_seq:
        for a in actlist:
            action_dom_info = find_node_by_id_for_prompt(domtree, a.get("id"))
            a["dom_info"] = action_dom_info
            actions_parrel.append(a)
    return actions_parrel, output

def worker(task_queue, result_queue, model_name):
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, data = item
        try:
            img_path = data.get("img_name")
            domtree_str = data.get("domtree")
            domtree = ast.literal_eval(domtree_str)
            actions, model_output_text = get_op_actions_by_model_no_history(img_path, domtree, model_name)
            # 只保留 img_name, 结果 (动作+文本+错误)
            out_row = {
                "img_name": img_path,
                "model_actions": actions,
                "model_output_text": model_output_text
            }
        except Exception as e:
            out_row = {
                "img_name": data.get("img_name"),
                "model_actions": [],
                "model_output_text": "",
                "error": str(e)
            }
        result_queue.put((idx, out_row))

def run_for_model(model_name, rows, output_path):
    task_queue = Queue(maxsize=NUM_WORKERS * 10)
    result_queue = Queue()
    
    workers = []
    for _ in range(NUM_WORKERS):
        p = Process(target=worker, args=(task_queue, result_queue, model_name))
        p.start()
        workers.append(p)
    for idx, row in enumerate(rows):
        task_queue.put((idx, row))
    for _ in range(NUM_WORKERS):
        task_queue.put(None)

    results = [None] * len(rows)
    for _ in tqdm(range(len(rows)), desc=f"{model_name}"):
        idx, out_row = result_queue.get()
        results[idx] = out_row
    
    with open(output_path, "w", encoding="utf8") as fout:
        for row in results:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    for p in workers:
        p.join()

def main():
    with open(INPUT_JSONL, "r", encoding="utf8") as fin:
        rows = [json.loads(line) for line in fin if line.strip()]
    for model_name in MODEL_NAMES:
        print(f"\n============ Running for Model: {model_name} ============")
        output_path = OUTPUT_DIR / f"{model_name}_results.jsonl"
        run_for_model(model_name, rows, output_path)
        print(f"Model {model_name} done, output to: {output_path}")

if __name__ == "__main__":
    main()