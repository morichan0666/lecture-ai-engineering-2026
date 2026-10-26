"""第1回: 素のLLMに質問し、会話を続けるスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 講座に固有の質問から開始する

ツールは持たない「素の」呼び出し。続けて質問すると、同じチャットの会話履歴を使う。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import chats, errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(
    chat: chats.Chat,
    question: str,
) -> types.GenerateContentResponse:
    """LLMに質問してレスポンスを返す。"""
    return chat.send_message(question)


def print_usage(response: types.GenerateContentResponse) -> None:
    """入力・出力トークン数を表示する。"""
    usage = response.usage_metadata
    if usage is None:
        return

    print(
        f"[トークン数] 入力: {usage.prompt_token_count}, "
        f"出力: {usage.candidates_token_count}"
    )


def run_chat(chat: chats.Chat, question: str) -> None:
    """会話を継続する。空欄入力で終了。"""
    while question:
        print(f"\n[質問] {question}\n")

        response = ask(chat, question)
        print(response.text or "テキストの回答がありません。")
        print_usage(response)

        question = input(
            "\n続けて質問してください（空欄で終了）: "
        ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="素のLLMに質問する"
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")

    try:
        with genai.Client(api_key=get_api_key()) as client:
            chat = client.chats.create(model=MODEL)
            run_chat(chat, args.question)

    except (EOFError, KeyboardInterrupt):
        print("\n終了します。")

    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)





if __name__ == "__main__":
    main()
