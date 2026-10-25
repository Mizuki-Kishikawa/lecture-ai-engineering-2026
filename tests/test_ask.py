"""第1回 宿題(実装): 会話履歴のテスト（LLMもAPIキーも使わない）。"""

from types import SimpleNamespace

from app.ask import chat_turn


class FakeModels:
    """generate_content に渡された contents を記録し、決まった回答を返す偽物。"""

    def __init__(self, answers):
        self.answers = list(answers)
        self.sent = []

    def generate_content(self, model, contents, config):
        self.sent.append([(c.role, c.parts[0].text) for c in contents])
        return SimpleNamespace(text=self.answers.pop(0), usage_metadata=None)


def test_chat_turn_sends_whole_history():
    client = SimpleNamespace(models=FakeModels(["Aliceさんですね", "Aliceです"]))
    history = []

    chat_turn(client, history, "私の名前はAliceです")
    answer = chat_turn(client, history, "私の名前は？")

    assert answer == "Aliceです"
    # 2回目の呼び出しでは、1回目の質問と回答も一緒に送っている
    assert client.models.sent[1] == [
        ("user", "私の名前はAliceです"),
        ("model", "Aliceさんですね"),
        ("user", "私の名前は？"),
    ]
    assert [c.role for c in history] == ["user", "model", "user", "model"]


def test_chat_turn_drops_question_when_no_answer():
    client = SimpleNamespace(models=FakeModels([None]))
    history = []

    assert chat_turn(client, history, "こんにちは") is None
    assert history == []
