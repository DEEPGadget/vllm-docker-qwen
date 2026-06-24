# vLLM 2-GPU 독립 서비스 (RTX Pro 6000 Blackwell SE x2)

단일 `docker-compose.yml` + `Makefile` 래퍼. 포트별 GPU 1장 전용(`TP=1`), 포트당 모델 1개를 선택.

## 전제 (이 구성이 가정하는 환경)

- **하드웨어**: RTX Pro 6000 Blackwell Server Edition(96GB) × 2장 (GPU 0 / GPU 1)
- **Docker 이미지**: `vllm/vllm-openai:v0.23.0` 가 호스트에 이미 로드되어 있음
  ```bash
  docker images | grep vllm-openai     # v0.23.0 존재 확인
  ```
- **가중치 경로**: `/mnt/data/models/` 아래 4개 모델 폴더가 존재 (compose에서 `:ro` 로 마운트)
  ```
  /mnt/data/models/
  ├── Qwen2.5-VL-72B-Instruct-AWQ      # 8000: vl72b
  ├── Qwen2.5-VL-32B-Instruct-AWQ      # 8001: vl32b
  ├── Qwen3-Coder-Next-FP8             # 8000: coder
  └── QwQ-32B                          # 8001: qwq
  ```
  각 폴더에는 `config.json`, `tokenizer.json`, `*.safetensors`, `model.safetensors.index.json` 등이 모두 포함되어 있어야 함.
- **네트워크**: 외부 인터넷 없는 온프레미스(에어갭). 모델·이미지·파이썬 패키지는 사전 반입 (아래 "에어갭 운영" 참고)
- **소프트웨어**: NVIDIA Container Toolkit + `nvidia` 런타임 동작, `docker compose` v2, `make`
- **경로/이미지 변경 시**: `docker-compose.yml` 의 `image:` 와 `volumes:` (`/mnt/data/models`) 값을 환경에 맞게 수정

| 포트 | GPU | 선택 가능한 모델 |
|------|-----|------------------|
| 8000 | 0 | `vl72b` (Qwen2.5-VL-72B-AWQ) · `coder` (Qwen3-Coder-Next-FP8) |
| 8001 | 1 | `vl32b` (Qwen2.5-VL-32B-AWQ) · `qwq` (QwQ-32B) |

## 사용 (Makefile)

```bash
make up            # 기본 조합 (8000=vl72b, 8001=vl32b)
make 8000-coder    # 8000을 Coder로 교체 (반대편 8001 무중단)
make 8001-qwq      # 8001을 QwQ로 교체
make ps            # 상태
make logs SVC=qwq  # 특정 서비스 로그
make down          # 전체 종료 + 정리
make help          # 명령 목록
```

각 교체 명령은 같은 포트의 다른 모델을 자동 stop 후 선택 모델을 기동하므로, 서비스명·profile을 직접 칠 필요가 없습니다.

## 테스트 스크립트

`pip install openai` 후 실행. (`test_img.jpg` 는 VL용 샘플 이미지)

```bash
python test_vl.py test_img.jpg                 # VL (8000=72B, 8001=32B)
python test_qwq.py "질문"                       # QwQ, 사고과정 분리 출력
python test_qwq.py "질문" --hide-think --stream # 옵션
```

## 호출 예 (curl, VL base64)

```bash
B64=$(base64 -w0 test_img.jpg)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-VL-72B-Instruct-AWQ","max_tokens":256,
       "messages":[{"role":"user","content":[
         {"type":"text","text":"이 이미지를 설명해줘"},
         {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,'"$B64"'"}}]}]}'
```

모델명은 `--served-model-name` 값: `Qwen2.5-VL-72B-Instruct-AWQ` / `Qwen2.5-VL-32B-Instruct-AWQ` / `QwQ-32B` / `Qwen3-Coder-Next`.

## 주요 설정 (compose에서 조정)

- `--max-model-len 32768` (config 한도이자 최대치, 초과하려면 YaRN)
- `--gpu-memory-utilization 0.90` (coder 0.92)
- AWQ `--quantization awq_marlin` · QwQ `--reasoning-parser deepseek_r1` · coder `--max-num-seqs 512`

## 에어갭(오프라인) 운영

- compose `environment` 에 오프라인·텔레메트리 차단 플래그 설정됨: `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `HF_HUB_DISABLE_TELEMETRY` / `VLLM_NO_USAGE_STATS` / `VLLM_DO_NOT_TRACK`
- VL 입력은 **base64만** 사용(외부 `image_url` URL은 컨테이너가 직접 fetch하므로 오프라인 불가)
- 반입 전 자급 패키징:
  ```bash
  docker save vllm/vllm-openai:v0.23.0 -o vllm-0.23.0.tar   # 인터넷 측에서
  docker load -i vllm-0.23.0.tar                            # 에어갭 측에서
  pip download openai -d wheels                             # 클라이언트용
  pip install --no-index --find-links wheels openai
  ```
- 검증: 컨테이너 외부차단을 원하면 호스트 `DOCKER-USER` 체인에 아웃바운드 DROP(로컬 대역 RETURN 예외) 적용

## 트러블슈팅

- **AWQ 커널(sm120) 실패**: `awq_marlin` → `awq`
- **ViT/flashinfer 오류**: compose `environment` 의 `VLLM_ATTENTION_BACKEND=FLASH_ATTN` 주석 해제
- **OOM**: `--max-model-len` 축소 또는 `--gpu-memory-utilization` 하향

## 재부팅 자동복원

```bash
sudo systemctl enable docker.service containerd.service
```

`restart: unless-stopped` → 직전 running 모델만 자동 복귀.

## 참고 — 토크나이저 패치 (적용 완료)

transformers 5.x가 Qwen2 계열 토크나이저를 slow로 잘못 로드하는 문제로 `tokenizer_class` 를 fast로 고정함(원본 `*.bak`). 모델 재다운로드 시 재적용:

```bash
for M in Qwen2.5-VL-72B-Instruct-AWQ Qwen2.5-VL-32B-Instruct-AWQ QwQ-32B; do
  f=/mnt/data/models/$M/tokenizer_config.json
  cp "$f" "$f.bak"
  python3 - "$f" <<'PY'
import json,sys
p=sys.argv[1]; d=json.load(open(p))
d["tokenizer_class"]="Qwen2TokenizerFast"
json.dump(d,open(p,"w"),ensure_ascii=False,indent=2)
PY
done
```
