"""
Qwen2.5-VL 이미지 입력 테스트 (OpenAI 호환 API)
  8000 -> Qwen2.5-VL-72B-Instruct-AWQ
  8001 -> Qwen2.5-VL-32B-Instruct-AWQ

설치:  pip install openai
실행:  python test_vl.py /path/to/image.jpg
       python test_vl.py https://example.com/image.jpg   # URL도 가능(서버가 외부망 있을 때)
"""
import base64
import mimetypes
import sys
from openai import OpenAI

# (태그, base_url, served-model-name)
ENDPOINTS = {
    "72B": ("http://localhost:8000/v1", "Qwen2.5-VL-72B-Instruct-AWQ"),
    "32B": ("http://localhost:8001/v1", "Qwen2.5-VL-32B-Instruct-AWQ"),
}

PROMPT = "이 이미지를 자세히 설명해줘."


def to_image_ref(src: str) -> str:
    """로컬 경로면 base64 data URI로, http(s)면 그대로 반환."""
    if src.startswith("http://") or src.startswith("https://"):
        return src
    mime = mimetypes.guess_type(src)[0] or "image/jpeg"
    with open(src, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def ask(base_url: str, model: str, image_ref: str) -> str:
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": image_ref}},
                ],
            }
        ],
        max_tokens=512,
        temperature=0.2,
    )
    return resp.choices[0].message.content


def main():
    if len(sys.argv) < 2:
        print("사용법: python test_vl.py <이미지경로 또는 URL>")
        sys.exit(1)

    image_ref = to_image_ref(sys.argv[1])

    for tag, (base_url, model) in ENDPOINTS.items():
        print(f"\n===== {tag}: {model} =====")
        try:
            print(ask(base_url, model, image_ref))
        except Exception as e:
            print("ERROR:", repr(e))


if __name__ == "__main__":
    main()
