from eval.prompt import build_messages, parse_response, parse_used_memory
from eval.tasks import load_split

TASK = load_split("dev")[0].tasks[0]


def test_arm_a_prompt_has_no_memory_section_or_used_memory_instruction():
    user = build_messages(TASK, None)[1]["content"]
    assert "USED_MEMORY" not in user and "Notes from earlier sessions" not in user and "[M1]" not in user
    assert TASK.spec in user and TASK.visible_tests.strip() in user


def test_memory_prompt_numbers_notes_and_asks_for_declaration():
    user = build_messages(TASK, ["first note", "second note"])[1]["content"]
    assert "[M1] first note" in user and "[M2] second note" in user and "USED_MEMORY" in user


def test_empty_retrieval_is_explicit():
    assert "no notes were retrieved" in build_messages(TASK, [])[1]["content"]


def test_hidden_tests_never_reach_the_prompt():
    for t in load_split("dev")[0].tasks:
        text = " ".join(m["content"] for m in build_messages(t, ["x"]))
        assert t.hidden_tests.strip() not in text and "class Hidden" not in text


def test_code_selection_prefers_block_defining_the_entrypoint():
    fn = TASK.entrypoints[0]
    reply = f"```python\nprint('example usage')\n```\n```python\ndef {fn}(overrides, base=None):\n    return 1\n```\n```python\n# trailing demo\nx = 1\n```\nUSED_MEMORY: []"
    parsed = parse_response(reply, TASK, 0, True)
    assert parsed.code_blocks == 3 and f"def {fn}" in parsed.code


def test_no_fenced_block_means_no_code():
    assert parse_response("def f(): pass", TASK, 0, False).code is None
    assert parse_response("```js\nlet x\n```", TASK, 0, False).code is None


def test_used_memory_variants():
    assert parse_used_memory("USED_MEMORY: [1, 3]", 3) == ([1, 3], [], True)
    assert parse_used_memory("used_memory: [M2]", 3) == ([2], [], True)
    assert parse_used_memory("USED_MEMORY: []", 3) == ([], [], True)
    assert parse_used_memory("USED_MEMORY: [none]", 3) == ([], [], True)
    assert parse_used_memory("USED_MEMORY: [2, 2, 1]", 3) == ([2, 1], [], True)
    assert parse_used_memory("USED_MEMORY: [0, 4, x, 2]", 3) == ([2], ["0", "4", "x"], True)
    assert parse_used_memory("no declaration", 3) == ([], [], False)
    assert parse_used_memory("USED_MEMORY: [1]\nlater\nUSED_MEMORY: [2]", 3)[0] == [2]  # last one wins


def test_arm_a_never_parses_used_memory():
    parsed = parse_response("```python\ndef x(): pass\n```\nUSED_MEMORY: [1]", TASK, 0, expect_used_memory=False)
    assert parsed.used_indices == [] and parsed.used_parse_ok is False
