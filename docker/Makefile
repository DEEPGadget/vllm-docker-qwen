# vLLM 모델 운영 (docker-compose.yml 래퍼)
# 포트 8000(GPU0): vl72b | coder    포트 8001(GPU1): vl32b | qwq
COMPOSE := docker compose
ALL_PROFILES := --profile vl72b --profile coder --profile vl32b --profile qwq

.PHONY: help up down ps 8000-vl72b 8000-coder 8001-vl32b 8001-qwq logs

help:           ## 사용법 출력
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

up:             ## 기본 조합 기동 (8000=vl72b, 8001=vl32b)
	@$(COMPOSE) --profile vl72b --profile vl32b up -d

8000-vl72b:     ## 8000 -> VL-72B
	@-$(COMPOSE) stop coder
	@$(COMPOSE) --profile vl72b up -d vl72b

8000-coder:     ## 8000 -> Coder-Next
	@-$(COMPOSE) stop vl72b
	@$(COMPOSE) --profile coder up -d coder

8001-vl32b:     ## 8001 -> VL-32B
	@-$(COMPOSE) stop qwq
	@$(COMPOSE) --profile vl32b up -d vl32b

8001-qwq:       ## 8001 -> QwQ
	@-$(COMPOSE) stop vl32b
	@$(COMPOSE) --profile qwq up -d qwq

ps:             ## 상태 확인
	@$(COMPOSE) ps

logs:           ## 로그 (예: make logs SVC=qwq)
	@$(COMPOSE) logs -f $(SVC)

down:           ## 전체 종료 + 정리
	@$(COMPOSE) $(ALL_PROFILES) down --remove-orphans
