"""第1回 宿題(発展): 応答時間とトークン数を計測し、複数のモデルや質問で比較するスクリプト。

使い方:
    uv run python -m app.bench                    # 既定のモデル × 質問 × 3回 を計測して表にする
    uv run python -m app.bench --models gemini-3.5-flash-lite gemini-3.5-flash --repeat 5
    uv run python -m app.bench --burst 20         # 間隔を空けずに連続で呼び、レート制限（429）に当たるか見る

ストリーミングで呼び出し、次の2つの時間を測る。
    最初のトークンまでの時間 … 送信してから回答の最初の文字が届くまで（思考トークンを使うモデルでは、考えている時間も含む）
    全体の時間             … 送信してから回答が最後まで届くまで
結果は CSV にも保存する（--out）。

無料枠のレート制限は Google の公式ドキュメントに数値が載っていないため、
429 が返ってきたら、エラーの中身（どの制限に・上限いくつで当たったか・何秒待てばよいか）を記録する。
"""

import argparse
import csv
import re
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.ask import DEFAULT_QUESTION
from app.config import TEMPERATURE, api_error_message, get_api_key

# gemini-2.5-flash-lite はモデル一覧には出るが、新規ユーザーには 404 を返す（2026年10月時点）
# gemini-3.5-flash は非推奨で gemini-3.6-flash に自動転送されるため、最初から 3.6-flash を指定する
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-flash-lite"]

# 出力の長さが違う3種類の質問
QUESTIONS = {
    "short": "LoRAとは何ですか？一文で答えてください。",
    "normal": "LoRAとは何ですか？",
    "course": DEFAULT_QUESTION,
}

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "lectures" / "01" / "bench_results.csv"


@dataclass
class Record:
    model: str
    question: str
    run: int
    status: str  # "ok" か、エラーのHTTPステータスコード
    first_token_s: float | None = None
    total_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thoughts_tokens: int | None = None
    total_tokens: int | None = None
    output_chars: int | None = None
    quota: str = ""  # 429 のとき: 当たった制限とその上限値
    retry_delay_s: float | None = None  # 429 のとき: サーバーが示した待ち時間
    served_model: str = ""  # 実際に応答したモデル（指定したモデルから転送されることがある）


def measure(client: genai.Client, model: str, question: str) -> Record:
    """1回呼び出して、時間とトークン数を測る。APIエラーはそのまま送出する。"""
    record = Record(model=model, question=question, run=0, status="ok")
    text = ""
    usage = None
    start = time.perf_counter()
    stream = client.models.generate_content_stream(
        model=model,
        contents=QUESTIONS[question],
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )
    for chunk in stream:
        if chunk.text:
            if record.first_token_s is None:
                record.first_token_s = time.perf_counter() - start
            text += chunk.text
        if chunk.usage_metadata is not None:
            usage = chunk.usage_metadata  # 最後のチャンクに全体の集計が入る
        if chunk.model_version:
            record.served_model = chunk.model_version
    record.total_s = time.perf_counter() - start

    record.output_chars = len(text)
    if usage is not None:
        record.prompt_tokens = usage.prompt_token_count
        record.output_tokens = usage.candidates_token_count
        record.thoughts_tokens = usage.thoughts_token_count or 0
        record.total_tokens = usage.total_token_count
    return record


def parse_rate_limit(e: errors.APIError) -> tuple[str, float | None]:
    """429 のエラー本文から、当たった制限（quotaId=上限値）と待ち時間（秒）を取り出す。"""
    body = e.details.get("error", {}) if isinstance(e.details, dict) else {}
    quotas, delay = [], None
    for detail in body.get("details", []):
        kind = detail.get("@type", "")
        if kind.endswith("QuotaFailure"):
            for v in detail.get("violations", []):
                quotas.append(f"{v.get('quotaId')}={v.get('quotaValue')}")
        elif kind.endswith("RetryInfo"):
            m = re.fullmatch(r"([\d.]+)s", detail.get("retryDelay", ""))
            delay = float(m.group(1)) if m else None
    return "; ".join(quotas), delay


def error_record(e: errors.APIError, model: str, question: str) -> Record:
    record = Record(model=model, question=question, run=0, status=str(e.code))
    if e.code == 429:
        record.quota, record.retry_delay_s = parse_rate_limit(e)
    return record


def run_bench(client: genai.Client, models: list[str], questions: list[str], repeat: int, interval: float) -> list[Record]:
    """モデル × 質問 × repeat 回を計測する。429 のときはサーバーが示す時間だけ待って1回だけやり直す。"""
    records = []
    unavailable = set()  # 404 を返したモデル。残りの試行は飛ばす
    for model in models:
        for question in questions:
            for run in range(1, repeat + 1):
                if model in unavailable:
                    break
                for attempt in range(2):
                    try:
                        record = measure(client, model, question)
                    except errors.APIError as e:
                        record = error_record(e, model, question)
                        print(f"  {model} {question} #{run}: {api_error_message(e.code, e.message)} {record.quota}")
                    record.run = run
                    records.append(record)
                    if record.status == "404":
                        unavailable.add(model)
                    if record.status != "429" or attempt == 1:
                        break
                    wait = (record.retry_delay_s or 60) + 1
                    print(f"  {wait:.0f} 秒待ってやり直します")
                    time.sleep(wait)
                if record.status == "ok":
                    print(
                        f"  {model} {question} #{run}: 全体 {record.total_s:.2f}s"
                        f"（最初のトークン {record.first_token_s:.2f}s）"
                        f" 出力 {record.output_tokens} / 思考 {record.thoughts_tokens} トークン"
                    )
                time.sleep(interval)
    return records


def run_burst(client: genai.Client, model: str, n: int) -> list[Record]:
    """間隔を空けずに n 回呼ぶ。最初の 429 で止める。"""
    records = []
    start = time.perf_counter()
    for i in range(1, n + 1):
        try:
            record = measure(client, model, "short")
        except errors.APIError as e:
            record = error_record(e, model, "short")
        record.run = i
        records.append(record)
        elapsed = time.perf_counter() - start
        print(f"  #{i} {elapsed:6.1f}s 経過: {record.status} {record.quota}")
        if record.status == "429":
            break
    return records


def min_max(values: list[float]) -> str:
    """「最小値〜最大値」の文字列にする（試行回数が少ないと標準偏差はあてにならないため）。"""
    if not values:
        return "-"
    lo, hi = min(values), max(values)
    fmt = "{:d}" if all(isinstance(v, int) for v in values) else "{:.2f}"
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}〜{fmt.format(hi)}"


def summarize(records: list[Record]) -> str:
    """モデル × 質問ごとに、最小値〜最大値 の Markdown の表を作る。"""
    lines = [
        "| モデル | 質問 | 成功/試行 | 全体 [s] | 最初のトークン [s] | 入力 | 出力 | 思考"
        " | 実効速度 [トークン/s] | 生成部分の速度 [トークン/s] |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    keys = dict.fromkeys((r.model, r.question) for r in records)
    for model, question in keys:
        group = [r for r in records if r.model == model and r.question == question]
        ok = [r for r in group if r.status == "ok"]
        # 実効速度 = (出力 + 思考) トークン数 / 全体の時間（通信や待ち時間も含めた、1秒あたりのトークン数）
        speed = [(r.output_tokens + r.thoughts_tokens) / r.total_s for r in ok if r.output_tokens]
        # 生成部分の速度 = 出力トークン数 / (全体 − 最初のトークン)（待ち時間・思考時間を除いた、回答を書き出す速さ）
        gen_speed = [
            r.output_tokens / (r.total_s - r.first_token_s)
            for r in ok
            if r.output_tokens and r.first_token_s is not None and r.total_s > r.first_token_s
        ]
        lines.append(
            f"| {model} | {question} | {len(ok)}/{len(group)}"
            f" | {min_max([r.total_s for r in ok])}"
            f" | {min_max([r.first_token_s for r in ok if r.first_token_s is not None])}"
            f" | {min_max([r.prompt_tokens for r in ok if r.prompt_tokens is not None])}"
            f" | {min_max([r.output_tokens for r in ok if r.output_tokens is not None])}"
            f" | {min_max([r.thoughts_tokens for r in ok if r.thoughts_tokens is not None])}"
            f" | {min_max(speed)} | {min_max(gen_speed)} |"
        )
    return "\n".join(lines)


def load_csv(path: Path) -> list[Record]:
    """save_csv で保存した結果を読み戻す（空欄は None）。"""
    types_ = {f.name: f.type for f in fields(Record)}
    records = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values = {}
            for name, raw in row.items():
                if raw == "" and types_[name] != str:
                    values[name] = None
                elif "int" in str(types_[name]):
                    values[name] = int(raw)
                elif "float" in str(types_[name]):
                    values[name] = float(raw)
                else:
                    values[name] = raw
            records.append(Record(**values))
    return records


def save_csv(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Record)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)


def main() -> None:
    parser = argparse.ArgumentParser(description="応答時間とトークン数を計測する")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--questions", nargs="+", choices=list(QUESTIONS), default=list(QUESTIONS))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--interval", type=float, default=5.0, help="呼び出しの間に空ける秒数（毎分の上限に当たらないように）")
    parser.add_argument("--burst", type=int, default=0, help="間隔なしで連続 N 回呼ぶ（--models の先頭のモデルを使う）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summarize", type=Path, help="APIを呼ばず、保存済みの CSV から表だけ作る")
    args = parser.parse_args()

    if args.summarize:
        print(summarize(load_csv(args.summarize)))
        return

    client = genai.Client(api_key=get_api_key())
    print(f"[temperature] {TEMPERATURE if TEMPERATURE is not None else "既定"}")
    if args.burst:
        records = run_burst(client, args.models[0], args.burst)
    else:
        records = run_bench(client, args.models, args.questions, args.repeat, args.interval)
        print()
        print(summarize(records))
    save_csv(records, args.out)
    print(f"\n[保存] {args.out}")


if __name__ == "__main__":
    main()
