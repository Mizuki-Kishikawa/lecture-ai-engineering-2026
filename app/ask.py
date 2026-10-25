"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 宿題(実装): 会話履歴を保ったまま続けて質問する
    uv run python -m app.ask --chat "LoRAとは何ですか？"   # 最初の質問を指定して対話を始める

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）

LLMのAPIの応答は、そのリクエストに含めた内容だけを材料に作られる（会話を指定する ID などはない）。
--chat では、これまでの質問と回答（履歴）を要約せずに毎回まとめて送り直すことで会話を続けている。
そのため、ターンを重ねるごとに入力トークン数が増えていく。
（サーバー側でも、前のリクエストと先頭が同じ部分はキャッシュとして使い回されることがある。
その分も入力トークン数には含まれ、内訳として [うちキャッシュ] に表示される）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, TEMPERATURE, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

# 対話モードを終了する入力
EXIT_COMMANDS = {"exit", "quit", "q"}


def generate(client: genai.Client, contents: str | list[types.Content]) -> str | None:
    """モデルを1回呼び出し、トークン数を表示して回答の文字列を返す。"""
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )

    # 第1回 演習: response.usage_metadata から入力・出力トークン数を表示する
    usage = response.usage_metadata
    if usage is not None:
        print(f"[入力トークン] {usage.prompt_token_count}")
        if usage.cached_content_token_count:
            print(f"[うちキャッシュ] {usage.cached_content_token_count}")
        print(f"[出力トークン] {usage.candidates_token_count}\n")

    return response.text


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    return generate(client, question)


def chat_turn(client: genai.Client, history: list[types.Content], question: str) -> str | None:
    """履歴に質問を足してモデルを呼び、回答も履歴に足して返す。

    回答が得られなかったとき（APIエラー・空の応答）は、質問を履歴から取り除く。
    user と model が交互に並んだ状態を保つため。
    """
    history.append(types.Content(role="user", parts=[types.Part(text=question)]))
    try:
        answer = generate(client, history)  # 質問1つではなく、履歴をまるごと送る
    except errors.APIError:
        history.pop()
        raise
    if not answer:
        history.pop()
        return None
    history.append(types.Content(role="model", parts=[types.Part(text=answer)]))
    return answer


def chat(first_question: str | None = None) -> None:
    """会話履歴を保ったまま、続けて質問する対話モード。"""
    client = genai.Client(api_key=get_api_key())
    history: list[types.Content] = []
    print("対話モードです。終了するには exit / quit / q か Ctrl+D を入力してください。\n")

    question = first_question
    while True:
        if question is None:
            try:
                question = input("あなた> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if question in EXIT_COMMANDS:
            break
        if not question:
            question = None
            continue

        print(f"[送信するメッセージ数] {len(history) + 1}（これまでの履歴 {len(history)} 件 + 今回の質問）")
        try:
            answer = chat_turn(client, history, question)
        except errors.APIError as e:
            # 1回のエラーで会話全体を失わないよう、終了せずに次の入力を待つ
            print(api_error_message(e.code, e.message) + "\n")
        else:
            print(f"モデル> {answer}\n" if answer else "（回答が得られませんでした）\n")
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保ったまま続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[temperature] {TEMPERATURE if TEMPERATURE is not None else "既定"}")

    if args.chat:
        chat(args.question)
        return

    question = args.question or DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
