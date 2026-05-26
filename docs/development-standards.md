# Embedding & Reranker 서비스 - 개발 표준

문서 버전: 1.0
대상 독자: 백엔드/AI 개발자, DevOps
관련 문서: [`./development-environment.md`](./development-environment.md)

> **본 서비스의 가장 큰 특징**: 단일 컨테이너 안에서 **임베딩(Embedding)** 과 **리랭커(Reranker)** 두 모델이 **공존**합니다.
> 두 모델은 아키텍처/입출력/메모리 특성이 다르므로 본 문서는 **공존 운영 규칙**을 별도 섹션으로 명시합니다(섹션 8).

---

## 목차

1. 개요
2. 개발환경
   2.1 개발환경 구성도
   2.2 개발절차
   2.3 개발자 PC 구성 내역
   2.4 IDE (Cursor / VSCode / PyCharm)
   2.5 소스 관리 (사내 Git + GitHub 미러)
   2.6 모델 / 패키지 / 이미지 저장소
   2.7 IDE 설정 및 런타임 설치
       2.7.1 IDE 설정 (Cursor / VSCode)
       2.7.2 Python / uv 설치
       2.7.3 CUDA 12.4 / NVIDIA Container Toolkit
       2.7.4 Docker / Compose
       2.7.5 Qwen3 임베딩 / 리랭커 모델 배치
       2.7.6 Hugging Face 인증
3. 디렉토리 & 모듈 표준
4. 의존성 / 패키지 관리 표준
5. 코드 스타일 표준
6. API 표준
7. 모델 운영 규칙 (공통)
8. **임베딩 ↔ 리랭커 공존 운영 규칙 (핵심)**
9. Docker / 배포 표준
10. 로깅 / 관측
11. 테스트 / 품질 SLA
12. 보안 / 데이터
13. Git / 브랜치 / PR
14. 백엔드 연동 시 주의

---

## 1. 개요

본 문서는 Embedding & Reranker 서비스(`/home/pps-nipa/jenkins/dev/embedding`)의 **개발 환경 / 모델 / 코드 / 배포** 표준을 정의합니다.
RAG 파이프라인의 (1) **임베딩 벡터화** 와 (2) **검색 결과 리랭킹** 을 단일 컨테이너의 FastAPI 로 제공합니다.

| 구분 | 기술 |
|------|------|
| 언어 | Python ≥ 3.11 |
| API | FastAPI + Uvicorn |
| 임베딩 모델 | `Qwen/Qwen3-Embedding-0.6B` (AutoModel, mean-pooling) |
| 리랭커 모델 | `Qwen/Qwen3-Reranker-0.6B` (AutoModelForCausalLM, yes/no logit) |
| 추론 백엔드 | PyTorch 2.6 + CUDA 12.4 (float16 GPU / float32 CPU) |
| 패키지 매니저 | uv |
| 컨테이너 | Docker / docker compose |
| CI | Jenkins (`/home/pps-nipa/jenkins/`) |

---

## 2. 개발환경

### 2.1 개발환경 구성도

```
┌──────────────────────────────────────────────────────────────────┐
│                           개발자 PC                                │
│   Cursor IDE  ──────────  Python 3.11 + uv (.venv)                 │
│        │                          │                                │
│        │ SSH/HTTPS                │ docker (GPU)                   │
└────────┼──────────────────────────┼────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────────┐   ┌──────────────────────────────────────────┐
│  사내 Git (Gitea)    │   │  Hugging Face Hub                          │
│  narea/embedding.git │   │  Qwen/Qwen3-Embedding-0.6B                 │
└────────┬────────────┘   │  Qwen/Qwen3-Reranker-0.6B                  │
         │                 └──────────────────────────────────────────┘
         ▼                                  │
┌────────────────────┐                       ▼
│ GitHub 미러         │            ┌──────────────────────────────────┐
│ GiJeongCho/         │            │ src/resources/model/             │
│ embedding           │            │  ├── embedding_qwen3_0_6b/        │
└────────────────────┘            │  └── reranker_qwen3_0_6b/         │
         │                         └──────────────────────────────────┘
         ▼
┌──────────────────────────────────────────────────────────────────┐
│                Jenkins 서버 (Build / Deploy)                      │
│  dev/docker.sh dev up embedding                                   │
│        │                                                          │
│        ▼                                                          │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │  embedding 컨테이너 (FastAPI + 2 models on one GPU)       │    │
│   │  ├─ EmbeddingService  → POST /embed                       │    │
│   │  └─ RerankerService    → POST /rerank                     │    │
│   └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 개발절차

1. 개발자 PC에 IDE, Python 3.11, uv, Docker, NVIDIA 드라이버를 설치한다.
2. SSH 키 등록(사내 Git, GitHub).
3. `git clone ssh://git@git.biz.ppsystem.co.kr:10022/narea/embedding.git`.
4. `uv sync` 로 의존성 설치.
5. `python test/download_model.py --all` 로 임베딩 + 리랭커 모델 다운로드.
6. `PYTHONPATH=. uv run uvicorn src.api:app --host 0.0.0.0 --port 6002 --reload` 실행 → `/health` 확인.
   - **`embedding.model_loaded`, `reranker.model_loaded` 가 모두 true 인지** 반드시 검사.
7. 기능 단위 PR → 사내 Git push → GitHub 미러(`GiJeongCho/embedding`) 동시 반영.
8. Jenkins Job 트리거 → 이미지 빌드 → dev/stg/prd 배포.
9. 배포 후:
   - `/embed` 짧은 문장 1건으로 워밍업
   - `/rerank` 3~5개 문서로 워밍업
   - VRAM 사용량(`nvidia-smi`) 측정

### 2.3 개발자 PC 구성 내역

| 항목 | 최소 | 권장 | 비고 |
|------|------|------|------|
| OS | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS | macOS는 CPU 추론만 가능 |
| CPU | 4 core | 8 core+ | |
| RAM | 16 GB | 32 GB | 모델 로드 시 CPU 메모리도 일시적으로 사용 |
| Disk | 30 GB | 200 GB SSD | 0.6B 모델 2개 ≈ 3 GB. 4B/8B 사용 시 +50GB 이상 |
| GPU | 없음(CPU 가능) | RTX 3060 12GB+ | float16 0.6B + 0.6B → VRAM 3~4 GB |
| Python | 3.11.x | 3.11.x | `pyproject` 가 ≥ 3.11 강제 |
| Docker | 24.x | 26.x | `--gpus all` 지원 |
| CUDA 드라이버 | 12.4 호환 (R545+) | 최신 LTS | |

### 2.4 IDE (Cursor / VSCode / PyCharm)

- 권장: Cursor 또는 VSCode.
- 필수 확장:
  - **Python**, **Pylance**, **Ruff**
  - **Docker**
  - **REST Client** 또는 **Thunder Client** (`/embed`, `/rerank` 호출 디버깅)
  - **Even Better TOML**
- 디버그 시 두 모델이 모두 메모리에 올라온 상태에서 작업하므로 `python.terminal.activateEnvironment=true` 권장.

### 2.5 소스 관리 (사내 Git + GitHub 미러)

- 사내 Git: `ssh://git@git.biz.ppsystem.co.kr:10022/narea/embedding.git`
- GitHub 미러: `https://github.com/GiJeongCho/embedding.git`
- `origin` 에 fetch 1 + push 2. `git push origin <branch>` 한 번으로 양쪽 반영.

### 2.6 모델 / 패키지 / 이미지 저장소

| 자원 | 저장소 | 비고 |
|------|--------|------|
| Qwen3 Embedding 가중치 | Hugging Face Hub `Qwen/Qwen3-Embedding-0.6B` / 사내 NAS 미러 | 4B/8B 모델은 별도 디스크 권장 |
| Qwen3 Reranker 가중치 | Hugging Face Hub `Qwen/Qwen3-Reranker-0.6B` / 사내 NAS 미러 | `allow_patterns` 로 필수 파일만 다운로드 (`download_model.py` 참조) |
| Python 패키지 | PyPI / 사내 Nexus | `[tool.uv.index].pytorch` 인덱스 사용 |
| Docker 이미지 | 사내 Registry (`IMG_EMBEDDING`) | dev/stg/prd 태그 분리 |

### 2.7 IDE 설정 및 런타임 설치

#### 2.7.1 IDE 설정 (Cursor / VSCode)

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.tabSize": 4
  },
  "files.watcherExclude": {
    "**/src/resources/model/**": true,
    "**/.venv/**": true,
    "**/uv.lock": true
  }
}
```

`.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Embedding API (uvicorn)",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": ["src.api:app", "--host", "0.0.0.0", "--port", "6002", "--reload"],
      "env": {
        "EMBEDDING_MODEL_PATH": "${workspaceFolder}/src/resources/model/embedding_qwen3_0_6b",
        "RERANKER_MODEL_PATH": "${workspaceFolder}/src/resources/model/reranker_qwen3_0_6b",
        "PYTHONPATH": "${workspaceFolder}"
      },
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

#### 2.7.2 Python / uv 설치

```bash
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /home/pps-nipa/jenkins/dev/embedding
uv sync
```

#### 2.7.3 CUDA 12.4 / NVIDIA Container Toolkit

```bash
nvidia-smi   # CUDA 12.4 호환 드라이버
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2.7.4 Docker / Compose

```bash
curl -fsSL https://get.docker.com | sh
docker compose version
```

#### 2.7.5 Qwen3 임베딩 / 리랭커 모델 배치

```bash
# 옵션 A) 프로젝트 헬퍼 스크립트
python test/download_model.py --all
# 결과:
# src/resources/model/embedding_qwen3_0_6b/
# src/resources/model/reranker_qwen3_0_6b/

# 옵션 B) huggingface-cli 직접
huggingface-cli download Qwen/Qwen3-Embedding-0.6B \
  --local-dir src/resources/model/embedding_qwen3_0_6b
huggingface-cli download Qwen/Qwen3-Reranker-0.6B \
  --local-dir src/resources/model/reranker_qwen3_0_6b
```

> 더 큰 임베딩 모델(4B/8B)로 교체 시 `download_model.py` 의 `EMBEDDING_MODELS` dict 주석을 해제하고 PR.

#### 2.7.6 Hugging Face 인증

- Qwen3 시리즈는 gated 가 아니지만 사내 망에서 Hub 접근 정책에 따라 토큰 필요.
- Jenkins → Credentials 의 `HF_TOKEN` 사용. `model_download.sh` 가 자동 호출.

---

## 3. 디렉토리 & 모듈 표준

| 레이어 | 위치 | 책임 |
|--------|------|------|
| API | `src/api.py` | FastAPI 라우팅, Pydantic 모델, lifespan, 로깅 |
| 임베딩 서비스 | `src/v1/service.py` | `EmbeddingService` (load / embed / mean-pooling) |
| 리랭커 서비스 | `src/v1/reranker.py` | `RerankerService` (load / rerank / yes-no scoring) |
| 모델 로더 | `src/v1/utils/model_loader.py` | `load_model` / `load_reranker` (두 함수 분리) |
| 자원 | `src/resources/model/` | 가중치 (Git 미추적) |
| 테스트 | `test/download_model.py` | 두 모델 모두 다운로드 가능 |

> **금지**: API 핸들러에서 `transformers` 직접 호출하지 않는다. 반드시 두 서비스 클래스를 거친다.
> **금지**: 임베딩 코드에서 리랭커 모델을, 리랭커 코드에서 임베딩 모델을 import 하지 않는다 (모듈 경계 유지).

### 3.1 API 버저닝
- 현재: `src/v1/`.
- 모델 교체로 인터페이스가 깨지면 `src/v2/` 신설 + `app.include_router(...)`.

### 3.2 네이밍
- 함수/모듈: `snake_case`
- 클래스: `PascalCase` (`EmbeddingService`, `RerankerService`)
- 상수: `UPPER_SNAKE` (`DEFAULT_MODEL_PATH`, `DEFAULT_RERANKER_PATH`)
- 모델 폴더명: `<task>_<family>_<size>` (`embedding_qwen3_0_6b`, `reranker_qwen3_0_6b`)

---

## 4. 의존성 / 패키지 관리 표준

- **`uv` 사용 강제** (로컬). Docker 빌드에서는 `pip install .` 로 `pyproject.toml` 직접 설치.
- 추가 패키지:
  1. `uv add <pkg>` → `pyproject.toml`, `uv.lock` 동시 갱신.
  2. CUDA 의존 패키지는 `[tool.uv.sources].torch` 의 `pytorch` 인덱스 사용.
- 핀: `torch>=2.6`, `transformers>=4.42.3`, `numpy>=1.26`.
- `trust_remote_code=True` 가 필요한 모델만 추가 가능. 임의로 활성화하지 않는다.

---

## 5. 코드 스타일 표준

### 5.1 일반
- PEP8, 4-space, 100자.
- docstring 필수.
- `print` 는 요청/응답 콘솔 추적에만 한정(현재 `api.py` 패턴). 그 외 `logger`.
- 임베딩/리랭커 양쪽 docstring 에 **모델 family / task** 명시 (예: `"""Qwen3 Embedding 서비스 레이어."""`).

### 5.2 타입 힌트
- 추론 함수의 인자/반환은 타입 힌트 필수.
- 반환 dict 는 가능한 한 `TypedDict` / `BaseModel`.
- numpy 배열 반환은 `np.ndarray` 명시.

### 5.3 로깅
- 포맷: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`.
- 임베딩 로딩과 리랭커 로딩은 **별도 try/except** 로 감싸서 한쪽 실패가 다른 쪽 부팅을 막지 않도록 한다 (현재 `lifespan` 패턴 유지).
- 향후 표준: `X-Request-ID` 헤더 → 로그/응답에 echo.

### 5.4 예외 처리
- `RuntimeError("X 모델이 로드되지 않았습니다.")` 패턴을 두 서비스 모두 사용.
- API 핸들러는 503 (`/embed` → 임베딩 503, `/rerank` → 리랭커 503) 로 매핑.
- 리랭커 배치 일부 실패는 0.0 점수로 채우고 경고 로그 (`reranker.py` 의 try/except 유지) — 사용자 응답을 막지 않는다.

---

## 6. API 표준

### 6.1 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | `embedding`, `reranker` 각각의 상태 + KST 시각 |
| POST | `/embed` | 텍스트 → 벡터 |
| POST | `/rerank` | (query, documents) → 정렬된 score |

> 두 모델 중 하나만 로드되어 있어도 다른 엔드포인트는 정상 동작해야 한다.

### 6.2 요청 모델
- 필드 추가 시 **기본값과 범위(`ge`, `le`)** 반드시 지정.
- `texts`, `documents` 는 `min_length=1` 강제(빈 입력 차단).

### 6.3 응답 표준
- 공통 응답에 `timestamp.request_kst`, `timestamp.response_kst`, `timestamp.total_elapsed_sec` 포함.
- 임베딩 응답: `embeddings`, `dimension`, `count`, `usage`.
- 리랭커 응답: `results[{index, score, document}]`, `count`, `usage`.
- `results.index` 는 항상 **요청 시 documents 배열에서의 원래 인덱스**여야 한다(클라이언트가 다시 매핑 가능).

---

## 7. 모델 운영 규칙 (공통)

### 7.1 모델 로딩
- `local_files_only=True`, `trust_remote_code=True` (Qwen3 필수) 유지.
- GPU 있으면 `torch.float16`, 없으면 `torch.float32` (코드에 자동 분기 — **변경 금지**).
- 로딩 시 `_load_lock`, `_reranker_load_lock` (threading.Lock) 으로 동시 로드 방지.
- 로드 후 `.eval()` 강제.

### 7.2 모델 경로
- `EMBEDDING_MODEL_PATH`, `RERANKER_MODEL_PATH` 로 외부 주입(직접 경로 하드코딩 금지).
- 기본 경로 계산은 `model_loader.py` 의 `DEFAULT_MODEL_PATH` / `DEFAULT_RERANKER_PATH` 만 사용.

### 7.3 임베딩 계산
- mean pooling 함수 `_mean_pooling` 만 사용. 다른 pooling(`[CLS]`, max 등) 도입 시 별도 PR + 사유.
- L2 정규화 옵션 유지 — RAG 인덱싱은 항상 `normalize=true` 권장.

### 7.4 리랭커 계산
- chat-template prefix(`_PREFIX`) / suffix(`_SUFFIX`) 와 `_format_pair` 는 **Qwen3-Reranker 공식 형식**. 임의로 수정 금지.
- 점수 = `softmax([no_logit, yes_logit])[1].exp()`. 다른 스코어링 함수 도입 시 별도 PR + 비교 평가.

---

## 8. 임베딩 ↔ 리랭커 공존 운영 규칙 (핵심)

### 8.1 메모리 / 라이프사이클

- 두 모델은 **lifespan 시점에 순차 로드**한다 (`embedding → reranker`).
- 한쪽 실패가 다른 쪽 부팅을 막지 않도록 **각 로드 블록을 별도의 try/except** 로 감싼다 (현재 `api.py` 패턴 유지).
- `/health` payload 는 두 모델 상태를 **독립적으로** 표시해야 한다 (`embedding`, `reranker` 키).
- **두 모델은 별도의 변수**(`EmbeddingService.model`, `RerankerService.model`)로 보관. 절대 공유 객체로 만들지 않는다.

### 8.2 디바이스 / VRAM
- 두 모델은 **같은 CUDA 디바이스에 올라간다**(동일 호스트의 단일 GPU 가정).
- VRAM 부족(OOM) 발생 시 표준 대응 순서:
  1. `batch_size` / `max_length` 축소(서비스 호출 측에서 조정)
  2. 리랭커 `max_length=8192` → `4096` 또는 `2048` 로 환경 정책 합의
  3. **임베딩 모델만 본 컨테이너에 두고, 리랭커는 별도 컨테이너로 분리** (compose 변경 PR)
- 다중 GPU 환경에서는 향후 표준에서 `EMBEDDING_DEVICE`, `RERANKER_DEVICE` 환경변수로 분리 운영 검토.

### 8.3 동시성 / 처리량
- 임베딩 `/embed` 는 보통 인덱싱 배치 호출(높은 throughput).
- 리랭커 `/rerank` 는 사용자 쿼리 응답 경로(낮은 latency 요구).
- **혼합 부하 시 우선순위 보장** 은 본 서비스가 수행하지 않는다. 백엔드가 큐/레이트 리미트로 조정.
- 단일 GPU 동시 호출은 PyTorch가 직렬화한다. 백엔드가 같은 컨테이너에 동시에 `/embed` + `/rerank` 를 쏘면 두 요청이 직렬로 처리됨을 가정.

### 8.4 모델 교체 / 업그레이드
- 임베딩 모델 교체(예: 0.6B → 4B):
  - `download_model.py` 의 `EMBEDDING_MODELS` 변경
  - `EMBEDDING_MODEL_PATH` 환경변수 갱신
  - 응답의 `dimension` 변경 → 백엔드/벡터DB(Milvus)도 함께 마이그레이션 필요 (메이저 버전 PR)
- 리랭커 모델 교체:
  - `RERANKER_MODELS` dict + `RERANKER_MODEL_PATH` 갱신
  - 토큰 ID(`yes`/`no`) 가 다른 모델은 `_compute_scores` 의 변환 로직 재검토 필요

### 8.5 분리 시점 가이드라인
다음 조건 중 하나라도 충족하면 **두 모델을 별도 컨테이너로 분리**한다.
- VRAM 사용량이 GPU 용량의 80% 를 지속적으로 초과
- 인덱싱(`/embed`) 트래픽이 쿼리(`/rerank`) latency 를 30% 이상 악화시킴
- 임베딩 모델 사이즈가 4B 이상으로 상승

---

## 9. Docker / 배포 표준

- Base image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`.
- `EXPOSE ${APP_PORT}` / `CMD ["sh","-c","uvicorn src.api:app --host 0.0.0.0 --port ${APP_PORT}"]` 패턴 유지.
- 모델 가중치는 이미지에 포함 금지. 호스트 read-only 볼륨 마운트.
- `.dockerignore`: `src/resources/model/`, `.venv/`, `__pycache__/`, `.git/`, `test/`.

### 9.1 Jenkins 연동
- `IMG_EMBEDDING=<registry>/embedding:<env>`.
- compose 서비스명: `embedding`.
- 호스트 모델 경로는 `EMBEDDING_MODEL_DIR`(또는 정해진 환경변수)로 통일하여 마운트.
- 환경별 포트(예시): dev 6004 / stg 9004 / prd 8004 — **포트 규칙(6/9/8000번대)** 준수.

### 9.2 헬스체크
```yaml
healthcheck:
  test: ["CMD", "curl", "-fsS", "http://localhost:${APP_PORT}/health"]
  interval: 30s
  timeout: 5s
  retries: 5
```

> 추가 권장: `/health` 의 `embedding.model_loaded == true && reranker.model_loaded == true` 를 검증하는 별도 모니터링 Job (Jenkins `90-health-check`)에서 alert.

---

## 10. 로깅 / 관측

| 항목 | 표준 |
|------|------|
| 요청 로그 | endpoint, 입력 길이/개수, max_length, batch_size, 앞 3개 미리보기 |
| 응답 로그 | dimension/count, inference_time, 총 소요시간 |
| 에러 로그 | `logger.error(..., exc_info=True)` (임베딩/리랭커 각각) |
| 리랭커 배치 부분 실패 | `logger.warning("리랭크 배치 실패 (idx=%d): %s", ...)` 후 0.0 채움 (현재 패턴) |
| `/health` | 200 보장. payload 의 두 모델 상태를 운영 모니터링이 파싱 |
| GPU 메트릭 | (향후 TODO) `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` 주기 수집 |

---

## 11. 테스트 / 품질 SLA

- 회귀 방지: 임베딩 모델 교체 시 동일 텍스트셋의 코사인 유사도 분포 비교 스냅샷.
- 리랭커 회귀: 골든 셋 (query, docs, expected ranking) 으로 NDCG@k 측정.
- 성능 SLA(가이드라인, 0.6B 단일 GPU 기준):
  - `/embed` 평균 latency ≤ 30 ms (배치 32, max_length 128)
  - `/rerank` 평균 latency ≤ 200 ms (문서 8개, max_length 2048)
  - 두 엔드포인트 혼합 부하 1시간 후 VRAM 변동 ≤ +5%
- 테스트 코드는 `test/` 에 추가. (현재 `download_model.py` 만 존재 → 회귀 테스트 추가는 TODO)

---

## 12. 보안 / 데이터

- 입력 텍스트/문서에 PII 가 포함될 수 있음 → 로그 미리보기는 100자 제한 (현재 패턴 유지).
- 본 서비스는 인증을 수행하지 않음 → 백엔드에서 권한/사용자 컨텍스트 검사.
- 모델 가중치/`HF_TOKEN` 등은 환경변수/Credentials 로만 전달.
- CORS 와일드카드는 dev/stg 한정. prd 에서는 ingress(nginx) 단에서 도메인 화이트리스트.

---

## 13. Git / 브랜치 / PR

- 브랜치: `feat/embedding-<topic>`, `fix/embedding-<topic>`, `model/embedding-<name>`(모델 교체).
- 커밋: `[embedding] <동사> <내용>` (예: `[embedding] add reranker batch fallback`).
- 모델 가중치는 절대 커밋 금지(`*.safetensors`, `*.bin`).
- 사내 Git + GitHub 미러(`GiJeongCho/embedding`) 동시 push (현재 push URL 이중 등록 완료).
- PR 본문에는 **두 모델 모두에 대한 영향 분석** 명시(임베딩만 바꾸더라도 리랭커 호환성 확인).

---

## 14. 백엔드 연동 시 주의

- RAG 표준 호출 흐름:
  1. **인덱싱**: `ocr_api` → 청크 → `/embed` (배치) → Milvus 저장
  2. **검색**: 사용자 질의 → `/embed` (1개) → Milvus top-K → `/rerank` (K개) → 상위 N → `llm_api`
- 본 서비스는 **stateless**. 캐시는 백엔드 책임.
- 임베딩 차원(`dimension`)은 모델에 종속 → 백엔드는 `/health` 또는 첫 `/embed` 응답으로 차원을 확인하고 Milvus 스키마와 정합성 검증.
- 리랭커 점수의 절대값은 모델 간 비교 불가 → 임계값 정책은 모델 교체 시마다 재튜닝.
- 타임아웃: 백엔드 측 connect 3s, read 60s 권장(긴 max_length + 큰 배치 시).
