"""
QwQ-32B 테스트 (OpenAI 호환 API / 추론 모델)
  8001 -> QwQ-32B  (profile: qwq)

설치:  pip install openai
실행:
  python test_qwq.py "질문"              # 사고과정 + 최종답 (기본)
  python test_qwq.py "질문" --hide-think  # 최종답만
  python test_qwq.py "질문" --stream      # 생성되는 대로 실시간 출력
주의: 8001에서 QwQ를 띄우려면 qwq 프로파일로 기동
  docker compose --profile qwq up -d
"""
import argparse
import re
from openai import OpenAI

BASE_URL = "http://localhost:8001/v1"
MODEL = "QwQ-32B"
DEFAULT_PROMPT = (
    "기계 5대가 위젯 5개를 만드는 데 5분이 걸린다. "
    "기계 100대가 위젯 100개를 만드는 데는 몇 분이 걸리는가? 풀이 과정을 보여줘."
)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(reasoning: str, content: str):
    """reasoning_content가 비어있으면 content 안의 <think>...</think>를 직접 분리."""
    if reasoning:
        return reasoning.strip(), content.strip()
    m = THINK_RE.search(content or "")
    if m:
        thinking = m.group(1).strip()
        answer = THINK_RE.sub("", content).strip()
        return thinking, answer
    return "", (content or "").strip()


def get_extra(obj, key):
    val = getattr(obj, key, None)
    if val:
        return val
    extra = getattr(obj, "model_extra", None) or {}
    return extra.get(key, "") or ""


def run_blocking(client, prompt, show_think):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096, temperature=0.6, top_p=0.95,
    )
    msg = resp.choices[0].message
    reasoning = get_extra(msg, "reasoning_content")
    thinking, answer = split_thinking(reasoning, msg.content or "")

    if show_think and thinking:
        print("----- 추론 과정 -----")
        print(thinking, "\n")
    print("----- 최종 답변 -----")
    print(answer)
    u = resp.usage
    if u:
        print(f"\n[tokens] prompt={u.prompt_tokens} "
              f"completion={u.completion_tokens} total={u.total_tokens}")


def run_stream(client, prompt, show_think):
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096, temperature=0.6, top_p=0.95,
        stream=True,
    )
    in_think = False
    answer_started = False
    for chunk in stream:
        delta = chunk.choices[0].delta
        r = get_extra(delta, "reasoning_content")
        c = delta.content or ""
        if r and show_think:
            if not in_think:
                print("----- 추론 과정 -----")
                in_think = True
            print(r, end="", flush=True)
        if c:
            if not answer_started:
                print("\n\n----- 최종 답변 -----" if in_think else "----- 최종 답변 -----")
                answer_started = True
            # 파서가 분리 못 한 경우 <think> 태그 노이즈 최소 처리
            print(c, end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    ap.add_argument("--hide-think", action="store_true", help="사고과정 숨김")
    ap.add_argument("--stream", action="store_true", help="실시간 스트리밍 출력")
    args = ap.parse_args()

    show_think = not args.hide_think
    client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

    print(f"===== {MODEL} =====")
    print(f"[PROMPT] {args.prompt}\n")
    try:
        if args.stream:
            run_stream(client, args.prompt, show_think)
        else:
            run_blocking(client, args.prompt, show_think)
    except Exception as e:
        print("ERROR:", repr(e))


if __name__ == "__main__":
    main()
