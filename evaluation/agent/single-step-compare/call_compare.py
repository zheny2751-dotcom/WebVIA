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
ROOT = Path(__file__).resolve().parent
INPUT_JSONL = ROOT / "ground_truths" / "data.jsonl"
OUTPUT_DIR = ROOT / "single_step_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # <-- ensure directory exists
NUM_WORKERS = 8


# 多模型批处理s
MODEL_NAMES = [
    "o4-mini-2025-04-16", 
    "claude-sonnet-4-20250514-thinking",
    "gpt-5-2025-08-07",
    "gemini-2.5-pro",
    "claude-3-7-sonnet-20250219-thinking",
    "gpt-4o-2024-11-20",
]
# ==============================

def extract_boxed_text(text):
    match = re.search(r'\\+boxed\{([^}]*)\}', text)
    if match:
        return match.group(1)
    return ""

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

def check_yes_no(boxed_text):
    if boxed_text == "有":
        return True
    elif boxed_text == "无":
        return False
    else:
        return None


def get_response(message, model_name):
    response = client.chat.completions.create(
             model=model_name,
             messages=message,
             max_completion_tokens=10000,
             timeout=600
         )
    return response


def actions_seq_to_str(actions_seqs):
    result = ""
    for act in actions_seqs:
        if act["action"] == "select":
            result += f"_select_{act['id']}_{act['value']}"
        elif act["action"] == "input":
            result += f"_input_{act['id']}_{act['input_val']}"
        elif act["action"] == "click":
            visible_text = act.get("dom_info", {}).get("visible_text", "")
            # 如果 visible_text 为空，则用 id
            if visible_text:
                result += f"_{visible_text}"
            else:
                result += f"_click_{act['id']}"
        # 其它操作扩展
        # else: pass
    return result


def check_terminate(text):
        terminate_matches = re.findall(r'terminate\{(.*?)\}', text)
        if terminate_matches:
            termination_text = terminate_matches[-1].strip()
        else:
            termination_text = ""
        if termination_text == "继续":
            return False
        elif termination_text == "完成":
            return True
        else:
            return None


def get_compare_by_model_no_history(img_path_list, actions_seqs, model_name):
    img_b64_list = []
    for img_path in img_path_list:
        img_path = (ROOT / "ground_truths" / img_path).resolve()
        with open(img_path, "rb") as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            img_b64_list.append(img_b64)
    interact_name = actions_seq_to_str(actions_seqs)
    prompt_base = (
        "你会收到多张网页图片作为互动历史的验证过程，"
        "多张网页图片呈正序排列，最后一张图片代表互动完成，第一张图片是起始图，每进行一次操作会截一次图，直到最后一张完成操作。比如如果只有两张图，就是只进行了一个操作，操作前截图是第一张，操作后截图是第二张。如果有四张图，就是进行了三次操作，操作前截图是第一张，第一次操作后是第二张，第二次操作是第三章，以此类推。"
        "请完成以下两个内容：\n"
        "1. 判断本次互动序列后网页是否产生了符合互动组件的变化，比如点击编辑会弹出编辑窗口，在编辑窗口输入一系列内容后点击“保存”会讲刚才输入的内容完整地保存到页面上，点击“取消”或者“关闭”后不会保存。我会给你这次点击的互动组件序列名字叫什么，有哪些内容。请认真观察网页，理解网页进行这一轮操作之后应该会出现哪些变化，重点判断在最后一次操作之后图片是否发生了这一系列动作应该有的变化。没有发生的变化比如在点击一个按钮后只有按钮本身高亮了，网页本身无变化。或者点击保存后退出了弹窗，但网页中没有真的保存。给出一个判断，回复我“无”表示两张图片大体上无变化，“有”表示发生了变化。请注意。将你给出的答案提取出来放在latex的\\boxed{}中，并且给出你判断的理由"
        "2. 请你仔细比较本次序列生成的最后一张图片，判断网页与本次的起始图相比是否出现了新的重要部分。如果新部分出现，需要继续检测（即页面与所有历史图片都不极度相似），请用\\terminate{}包裹“继续”；"
        "请注意！如果没有新变化（如页面整体无明显新部分/出现内容与之前某张历史图极度相似/页面中出现的新部分极其微小或互动内容过少不值得继续/页面中出现的新内容与原图中内容高度重复），例如在页面中删除了一个栏目，没有任何新的互动组件出现。请用\\terminate{}包裹“完成”。请你务必只用terminate{...}包裹继续或完成。\n"
        "请详细说明你的理由（boxed和terminate都要输出，理由附在后面）。\n"
        "互动动作/组件名: " + str(interact_name)
    )
    image_content = [{"type": "text", "text": prompt_base}]
    for idx, img in enumerate(img_b64_list):
        image_content.append({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + img},
        })
    try:
        response = get_response([{"role": "user", "content": image_content}], model_name)
        output = str(response.choices[0].message)
        model_pass = check_yes_no(extract_boxed_text(output))
        model_terminate = check_terminate(output)
        # print(output)
        # print(f"完成检测")
        return output, model_pass, model_terminate
    except Exception as ex:
        print(f"Exception: {ex}")
        return ""

def worker(task_queue, result_queue, model_name):
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, data = item
        try:
            id = data.get("id")
            img_path_list = [data.get("img_name_before")] + data.get("img_name_list_after")
            actions_seq = data.get("actions_seq")
            model_output_text, model_pass, model_terminate = get_compare_by_model_no_history(img_path_list, actions_seq, model_name)
            # 只保留 img_name, 结果 (动作+文本+错误)
            out_row = {
                "id" : id,
                "model_pass": model_pass,
                "model_terminate": model_terminate,
                "model_output_text": model_output_text
            }
        except Exception as e:
            out_row = {
                "id": id,
                "model_pass": None,
                "model_terminate": None,
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