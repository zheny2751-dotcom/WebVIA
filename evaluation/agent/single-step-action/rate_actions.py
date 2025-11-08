import json
import os

def to_action_key(action):
    return str(action.get("dom_info"))

def calc_f1(pred_actions, gold_actions):
    pred_set = set([to_action_key(a) for a in pred_actions])
    gold_set = set([to_action_key(a) for a in gold_actions])
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1

def load_jsonl(path):
    rows = []
    with open(path, "r") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    # <<< 配置 >>>
    gt_file = "./ground_truths/data.jsonl"  # 标准答案
    result_dir = "single_step_results"    # 里面是 *_results.jsonl 文件
    output_jsonl = "MODEL_eval_results.jsonl" # 输出结果

    # <<< 加载标准答案 >>>
    gt_rows = load_jsonl(gt_file)
    gt_dict = {row["img_name"]: row for row in gt_rows}  # 用 img_name 匹配

    output_results = []

    for fname in os.listdir(result_dir):
        if not fname.endswith("_results.jsonl"):
            continue
        model_name = fname.replace("_results.jsonl", "")
        pred_rows = load_jsonl(os.path.join(result_dir, fname))
        scores = []
        matched = 0
        for row in pred_rows:
            img_name = row["img_name"]
            model_actions = row.get("model_actions", [])
            gt_row = gt_dict.get(img_name, None)
            if gt_row is None:
                continue
            gt_actions = gt_row.get("action_seqs", []) or gt_row.get("action_seq", [])
            precision, recall, f1 = calc_f1(model_actions, gt_actions)
            scores.append((precision, recall, f1))
            matched += 1
        avg_prec = sum(x[0] for x in scores) / max(len(scores), 1)
        avg_rec  = sum(x[1] for x in scores) / max(len(scores), 1)
        avg_f1   = sum(x[2] for x in scores) / max(len(scores), 1)
        output_results.append({
            "model_name": model_name,
            "avg_precision": round(avg_prec, 4),
            "avg_recall": round(avg_rec, 4),
            "avg_f1": round(avg_f1, 4),
            "matched_samples": matched
        })
        print(f"{model_name}: F1={avg_f1:.4f}")

    with open(output_jsonl, "w") as fout:
        for res in output_results:
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()