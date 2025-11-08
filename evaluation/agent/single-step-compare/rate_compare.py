import json
import os
from sklearn.metrics import f1_score, accuracy_score

def load_jsonl(path):
    rows = []
    with open(path, "r") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def binary_f1(y_true, y_pred):
    """
    计算 balanced/宏 F1（先分别对正类/负类算，再平均）
    """
    return f1_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)

def main():
    gt_file = "./ground_truths/data.jsonl"
    result_dir = "./single_step_results"
    output_jsonl = "COMPARE_EVAL_RESULTS.jsonl"
    id_key = "id"
    setup = [
        ("pass", "model_pass"),
        ("terminate", "model_terminate")
    ]
    gt_rows = load_jsonl(gt_file)
    gt_dict = {str(row[id_key]): row for row in gt_rows}
    output_results = []
    for fname in os.listdir(result_dir):
        if not fname.endswith("_results.jsonl"):
            continue
        model_name = fname.replace("_results.jsonl", "")
        pred_rows = load_jsonl(os.path.join(result_dir, fname))
        # print(pred_rows)
        pred_dict = {str(row[id_key]): row for row in pred_rows}
        out_row = {"model_name": model_name}
        f1_list = []
        acc_list = []
        for gt_key, pred_key in setup:
            y_true, y_pred = [], []
            matched = 0
            for sid, gt_row in gt_dict.items():
                pred_row = pred_dict.get(sid, {})
                gt_label = bool(gt_row.get(gt_key, False))
                pred_label = bool(pred_row.get(pred_key, False))
                y_true.append(gt_label)
                y_pred.append(pred_label)
                matched += 1
            try:
                f1 = f1_score(y_true, y_pred, average='macro')
                acc = accuracy_score(y_true, y_pred)
            except Exception:
                f1 = acc = 0.0
            # out_row[f"{gt_key}_f1"] = round(f1, 4)
            out_row[f"{gt_key}_accuracy"] = round(acc, 4)
            out_row[f"{gt_key}_matched_samples"] = matched
            print(f"{model_name}: {gt_key} accuracy={acc:.4f}, F1={f1:.4f}")
            f1_list.append(f1)
            acc_list.append(acc)
        # out_row["overall_f1"] = round(sum(f1_list) / len(f1_list), 4)
        out_row["overall_accuracy"] = round(sum(acc_list) / len(acc_list), 4)
        output_results.append(out_row)
    with open(output_jsonl, "w") as fout:
        for res in output_results:
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
if __name__ == "__main__":
    main()