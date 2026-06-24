"""
Qwen3-Coder-Next 테스트 (OpenAI 호환 API / 코딩·툴콜 모델)
  8000 -> Qwen3-Coder-Next  (profile: coder)

설치:  pip install openai
실행:
  python test_coder.py                     # 기본 코드 생성
  python test_coder.py "원하는 코딩 요청"     # 프롬프트 교체
  python test_coder.py --stream            # 실시간 출력
  python test_coder.py --tool              # 함수 호출(tool calling) 데모
주의: 8000에서 Coder를 띄우려면 coder 프로파일로 기동
  docker compose --profile coder up -d
"""
import argparse
import json
from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"
MODEL = "Qwen3-Coder-Next"

DEFAULT_PROMPT = "파이썬으로 LRU 캐시를 데코레이터 없이 직접 구현하고, 간단한 사용 예시도 보여줘."

# Coder 권장 샘플링 (top_k/min_p는 OpenAI 표준 밖이라 extra_body로 전달)
SAMPLING = dict(temperature=1.0, top_p=0.95)
EXTRA = {"top_k": 40, "min_p": 0.01}


def run_blocking(client, prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048, **SAMPLING, extra_body=EXTRA,
    )
    print(resp.choices[0].message.content)
    u = resp.usage
    if u:
        print(f"\n[tokens] prompt={u.prompt_tokens} "
              f"completion={u.completion_tokens} total={u.total_tokens}")


def run_stream(client, prompt):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048, **SAMPLING, extra_body=EXTRA, stream=True,
    )
    for chunk in stream:
        c = chunk.choices[0].delta.content or ""
        print(c, end="", flush=True)
    print()


# ---- 툴 콜링 데모 -------------------------------------------------
def square_the_number(num: float) -> float:
    return num * num


TOOLS = [{
    "type": "function",
    "function": {
        "name": "square_the_number",
        "description": "주어진 숫자의 제곱을 반환한다.",
        "parameters": {
            "type": "object",
            "required": ["num"],
            "properties": {"num": {"type": "number", "description": "제곱할 숫자"}},
        },
    },
}]


def run_tool(client):
    messages = [{"role": "user", "content": "1024를 제곱해줘."}]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS,
        max_tokens=1024, **SAMPLING, extra_body=EXTRA,
    )
    msg = resp.choices[0].message

    if not msg.tool_calls:
        print("(모델이 툴을 호출하지 않음)\n", msg.content)
        return

    # 1) 모델의 tool_call 메시지 추가
    messages.append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
    })

    # 2) 각 tool_call 로컬 실행 후 결과 회신
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        print(f"[tool call] {tc.function.name}({args})")
        result = square_the_number(**args)
        print(f"[tool result] {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result),
        })

    # 3) 결과를 받아 최종 답변 생성
    final = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS,
        max_tokens=1024, **SAMPLING, extra_body=EXTRA,
    )
    print("\n----- 최종 답변 -----")
    print(final.choices[0].message.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--stream", action="store_true", help="실시간 스트리밍 출력")
    ap.add_argument("--tool", action="store_true", help="함수 호출(tool calling) 데모")
    args = ap.parse_args()

    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    print(f"===== {MODEL} =====")
    try:
        if args.tool:
            run_tool(client)
        else:
            print(f"[PROMPT] {args.prompt}\n")
            (run_stream if args.stream else run_blocking)(client, args.prompt)
    except Exception as e:
        print("ERROR:", repr(e))


if __name__ == "__main__":
    main()
