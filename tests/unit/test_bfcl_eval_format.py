from __future__ import annotations

import json

from evals.bfcl.bfcl_format import case_messages, case_tools, calls_to_bfcl_result, prediction_record
from agent_core.providers.base import ModelRole
from agent_core.types.tools import ToolCall


def test_bfcl_case_tools_preserve_raw_schema_keywords() -> None:
    case = {
        "id": "raw_schema",
        "question": "pick one",
        "function": [
            {
                "name": "choose",
                "description": "choose value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "integer"},
                            ]
                        }
                    },
                },
            }
        ],
    }

    tools = case_tools(case)

    assert tools[0].name == "choose"
    assert "oneOf" in tools[0].input_schema["properties"]["value"]


def test_bfcl_case_messages_accept_nested_question_messages() -> None:
    case = {
        "question": [[{"role": "user", "content": "hello"}]],
        "function": [],
    }

    messages = case_messages(case)

    assert messages[0].role == ModelRole.USER
    assert messages[0].content[0].text == "hello"  # type: ignore[union-attr]


def test_bfcl_prediction_record_matches_jsonl_shape() -> None:
    case = {"id": "simple_python_0", "question": "area?", "function": []}
    calls = [ToolCall(tool_call_id="c1", name="calculate_triangle_area", arguments={"base": 10, "height": 5})]

    record = prediction_record(case=case, index=0, calls=calls, assistant_text="")

    assert record["id"] == "simple_python_0"
    assert record["result"] == "calculate_triangle_area(base=10, height=5)"
    assistant = record["inference_log"][1]["content"]
    assert json.loads(assistant) == [{"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}}]
    assert calls_to_bfcl_result([]) == ""
