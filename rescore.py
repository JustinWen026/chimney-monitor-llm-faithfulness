"""
rescore.py
----------
對既有的消融實驗輸出（不重跑模型）重新計分，並同時輸出「原始規則」與「修正後規則」的結果。

修正後規則（在查看新結果前就寫死）：
  1. 攝影機：真實編號後面不能緊接英數字，避免 CAM-01B 被算成 CAM-01。
  2. 風險等級：數字必須緊鄰「風險等級／风险等级／risk level」等字樣（或「5級風險」形式），
     不再接受報告中任意位置出現的數字（事件編號 54、年份 2026、0.15 ppm 都會誤中）。
  3. 事件類型：補齊常見中文同義詞（含簡體），並在檢查 fire 前先移除 "black_smoky+fire"
     字串，避免複合事件的英文原始值讓 fire 事件自動得分；複合事件需同時提到黑煙與火，
     或明確寫出複合類型。
  4. 雜訊提及：依來源分組計分（單事故使用 v17 的分組；多事故相同的誘餌相機編號只算一組）。

限制：多事故仍是全文比對，無法判斷事實寫在哪一筆事件底下（無法偵測跨事件張冠李戴）。

用法：
    python rescore.py --single ablation_results_v12.json --multi ablation_results_multi_v2_XXXX.json
    （兩個參數擇一或都給；--out 指定輸出 JSON 檔名，預設 rescore_output.json）
"""

import argparse
import json
import re
from itertools import combinations

import numpy as np
from scipy import stats

ARMS = ["baseline", "tool_only", "proposed"]

# ── 原始規則（照抄 v14 / multi_v2）─────────────────────────────────────────
OLD_EVENT_KW = {
    "fire": ["火災", "火灾", "明火", "火焰", "起火", "fire"],
    "black_smoky": ["黑煙", "黑烟", "black_smoky", "black smoke"],
    "white_smoky": ["白煙", "白烟", "白色煙霧", "白色烟雾", "white_smoky", "white smoke"],
    "black_smoky+fire": ["黑煙", "黑烟", "black_smoky", "複合", "复合", "black_smoky+fire"],
}


def old_checks(text, gt):
    return {
        "camera": gt["camera_id"] in text,
        "risk": str(gt["risk_level"]) in text,
        "event": any(kw in text for kw in OLD_EVENT_KW.get(gt["event_type"], [])),
    }


# ── 修正後規則 ─────────────────────────────────────────────────────────────
FIRE_TERMS = ["火災", "火灾", "明火", "火焰", "起火", "火警", "火情", "著火", "着火",
              "失火", "燃燒", "燃烧", "火勢", "火势", "火光", "火源", "火苗", "fire"]
# 「黑煙和火」「黑烟与火」這類並列寫法也算提到火
BLACK_AND_FIRE = re.compile(r"黑[煙烟](?:[和與与及、跟]|以及|並伴有|并伴有|伴隨|伴随|伴有)火")
BLACK_TERMS = ["黑煙", "黑烟", "黑色煙", "黑色烟", "black_smoky", "black smoke"]
WHITE_TERMS = ["白煙", "白烟", "白色煙", "白色烟", "white_smoky", "white smoke"]
COMPOSITE_TERMS = ["black_smoky+fire", "複合", "复合"]

RISK_PATTERNS = [
    # 允許簡繁混用（例如「风险等級」）
    re.compile(r"(?:[風风][險险](?:等[級级]|[級级][別别]|[層层][級级]|程度)|"
               r"risk[\s_-]*level)[^\d\n]{0,10}?([1-5])(?![\d.])", re.IGNORECASE),
    re.compile(r"(?<![\d.])([1-5])\s*[級级]\s*[風风][險险]"),
]


def _has(text, terms):
    return any(t.lower() in text.lower() for t in terms)


def new_event_hit(text, event_type):
    if event_type == "fire":
        stripped = re.sub(r"black_smoky\+fire", "", text, flags=re.IGNORECASE)
        return _has(stripped, FIRE_TERMS)
    if event_type == "black_smoky":
        return _has(text, BLACK_TERMS)
    if event_type == "white_smoky":
        return _has(text, WHITE_TERMS)
    if event_type == "black_smoky+fire":
        stripped = re.sub(r"black_smoky\+fire", "", text, flags=re.IGNORECASE)
        return (_has(text, COMPOSITE_TERMS) or BLACK_AND_FIRE.search(text) is not None
                or (_has(text, BLACK_TERMS) and _has(stripped, FIRE_TERMS)))
    return False


def risk_digits(text):
    found = set()
    for p in RISK_PATTERNS:
        found.update(int(m.group(1)) for m in p.finditer(text))
    return found


def new_checks(text, gt):
    cam = re.search(re.escape(gt["camera_id"]) + r"(?![0-9A-Za-z])", text) is not None
    return {
        "camera": cam,
        "risk": int(gt["risk_level"]) in risk_digits(text),
        "event": new_event_hit(text, gt["event_type"]),
    }


# ── 雜訊分組 ───────────────────────────────────────────────────────────────
SINGLE_GROUPS = {  # v17 的分組（v14 原本逐字計分）
    "camera_CAM-09": ["CAM-09"],
    "wind": ["12.4 m/s"],
    "air_quality": ["SO2", "NOx", "0.08 ppm", "0.15 ppm"],
    "email": ["safety_backup@petrochem.com"],
    "timestamps": ["13:25:00", "13:47:02"],
    "image_params": ["29.8 FPS", "contrast"],
}
SINGLE_FLAT = ["CAM-09", "12.4 m/s", "SO2", "NOx", "0.08 ppm", "0.15 ppm",
               "safety_backup@petrochem.com", "13:25:00", "13:47:02", "29.8 FPS", "contrast"]


def multi_groups(gts):
    g = {
        "wind": ["12.4 m/s"],
        "air_quality": ["SO2", "NOx", "0.08 ppm", "0.15 ppm"],
        "email": ["safety_backup@petrochem.com"],
        "timestamps": ["13:25:00", "13:47:02"],
        "image_params": ["29.8 FPS", "contrast"],
    }
    for cam in sorted({gt["camera_id"] for gt in gts}):  # 相同誘餌編號只算一組
        g[f"decoy_{cam}B"] = [f"{cam}B"]
    return g


def groups_hit(text, groups):
    t = text.lower()
    return [name for name, kws in groups.items() if any(k.lower() in t for k in kws)]


# ── 統計 ───────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def boot_ci(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = [0.0] * m, 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, pvals[i] * (m - k)))
        adj[i] = running
    return adj


def compare(values_by_arm):
    """一個比較家族（一個指標 × 一個規模）內的所有兩兩比較，Holm 校正。"""
    rows = []
    for a, b in combinations(ARMS, 2):
        x, y = values_by_arm[a], values_by_arm[b]
        if np.var(x) == 0 and np.var(y) == 0:
            rows.append({"pair": f"{a} vs {b}", "note": "both constant, no test"})
            continue
        t = stats.ttest_ind(x, y, equal_var=False)
        u = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "pair": f"{a} vs {b}",
            "mean_diff": float(np.mean(x) - np.mean(y)),
            "welch_t": float(t.statistic), "welch_p": float(t.pvalue),
            "mwu_p": float(u.pvalue),
            "cohens_d": cohens_d(x, y),
            "boot95ci_diff": boot_ci(x, y),
        })
    tested = [r for r in rows if "welch_p" in r]
    for r, pa, pm in zip(tested, holm([r["welch_p"] for r in tested]), holm([r["mwu_p"] for r in tested])):
        r["welch_p_holm"], r["mwu_p_holm"] = pa, pm
    return rows


def describe(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


# ── 主流程 ─────────────────────────────────────────────────────────────────
def score_file(path, scale):
    d = json.load(open(path, encoding="utf-8"))
    gts = [d["ground_truth_incident"]] if scale == "single" else d["ground_truth_incidents"]
    groups = SINGLE_GROUPS if scale == "single" else multi_groups(gts)
    out = {"scale": scale, "file": path, "arms": {}}

    for arm in ARMS:
        runs = d[f"raw_{arm}"]
        rec = {k: [] for k in ["old_rate", "new_rate", "old_all", "new_all",
                               "noise_flat", "noise_grouped"]}
        item = {f"{rule}_{inc}_{it}": [] for rule in ["old", "new"]
                for inc in [g["incident_id"] for g in gts] for it in ["camera", "risk", "event"]}
        group_counts = {g: 0 for g in groups}
        for r in runs:
            text = r["output_text"]  # Proposed 的 output_text 已移除 JSON 區塊
            oc, nc = [], []
            for gt in gts:
                o, n = old_checks(text, gt), new_checks(text, gt)
                for it in ["camera", "risk", "event"]:
                    item[f"old_{gt['incident_id']}_{it}"].append(o[it])
                    item[f"new_{gt['incident_id']}_{it}"].append(n[it])
                oc += list(o.values())
                nc += list(n.values())
            rec["old_rate"].append(sum(oc) / len(oc))
            rec["new_rate"].append(sum(nc) / len(nc))
            rec["old_all"].append(all(oc))
            rec["new_all"].append(all(nc))
            hits = groups_hit(text, groups)
            rec["noise_grouped"].append(len(hits))
            for h in hits:
                group_counts[h] += 1
            if scale == "single":
                rec["noise_flat"].append(sum(1 for k in SINGLE_FLAT if k.lower() in text.lower()))
        out["arms"][arm] = {
            "raw": rec,
            "narrative_old": describe(rec["old_rate"]),
            "narrative_new": describe(rec["new_rate"]),
            "all_correct_old": float(np.mean(rec["old_all"])),
            "all_correct_new": float(np.mean(rec["new_all"])),
            "noise_grouped": describe(rec["noise_grouped"]),
            "noise_flat": describe(rec["noise_flat"]) if rec["noise_flat"] else None,
            "item_hit_rates": {k: float(np.mean(v)) for k, v in item.items()},
            "noise_group_run_share": {g: c / len(runs) for g, c in group_counts.items()},
        }

    out["stats"] = {
        metric: compare({a: out["arms"][a]["raw"][key] for a in ARMS})
        for metric, key in [("narrative_old", "old_rate"), ("narrative_new", "new_rate"),
                            ("noise_grouped", "noise_grouped")]
    }
    return out


def print_report(res):
    print(f"\n{'=' * 78}\n  {res['scale'].upper()}  ({res['file']})\n{'=' * 78}")
    for arm in ARMS:
        a = res["arms"][arm]
        print(f"[{arm}] n={a['narrative_old']['n']}")
        print(f"   敘事提及率  原規則 {a['narrative_old']['mean']:.3f}±{a['narrative_old']['sd']:.3f}"
              f"   新規則 {a['narrative_new']['mean']:.3f}±{a['narrative_new']['sd']:.3f}")
        print(f"   全部項目正確  原規則 {a['all_correct_old']:.3f}   新規則 {a['all_correct_new']:.3f}")
        print(f"   雜訊提及(分組) {a['noise_grouped']['mean']:.2f}±{a['noise_grouped']['sd']:.2f}"
              + (f"   (逐字 {a['noise_flat']['mean']:.2f})" if a["noise_flat"] else ""))
        print("   各項命中率:", {k: round(v, 3) for k, v in a["item_hit_rates"].items()})
        print("   各雜訊組出現比例:", {k: round(v, 3) for k, v in a["noise_group_run_share"].items()})
    for metric, rows in res["stats"].items():
        print(f"\n  -- {metric}（家族內 Holm 校正）")
        for r in rows:
            if "note" in r:
                print(f"     {r['pair']:<24} {r['note']}")
            else:
                print(f"     {r['pair']:<24} diff={r['mean_diff']:+.3f}  d={r['cohens_d']:+.2f}  "
                      f"Welch p={r['welch_p']:.4f} (Holm {r['welch_p_holm']:.4f})  "
                      f"MWU p={r['mwu_p']:.4f} (Holm {r['mwu_p_holm']:.4f})  "
                      f"CI[{r['boot95ci_diff'][0]:+.3f},{r['boot95ci_diff'][1]:+.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single")
    ap.add_argument("--multi")
    ap.add_argument("--out", default="rescore_output.json")
    args = ap.parse_args()
    results = []
    if args.single:
        results.append(score_file(args.single, "single"))
    if args.multi:
        results.append(score_file(args.multi, "multi"))
    for r in results:
        print_report(r)
    for r in results:
        for a in r["arms"].values():
            a.pop("raw")
    json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n已存檔：{args.out}")


if __name__ == "__main__":
    main()
