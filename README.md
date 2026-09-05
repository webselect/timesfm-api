# timesfm-api

*[Version française](README.fr.md)*

REST service around **[TimesFM 3.0](https://github.com/google-research/timesfm)** (Google
Research) for time series forecasting, with built-in Swagger documentation.

Built to be called from an external client — a TypeScript trading bot, for instance: the API is
generic, carries no business logic, and exposes the **full** surface of the model.

---

## Requirements

- macOS on Apple Silicon (tested on an M1 Pro) or Linux
- Python ≥ 3.10 — the project uses `/opt/homebrew/bin/python3.12`
- ~1.2 GB of disk space for the weights, downloaded from HuggingFace on first start

## Install

```bash
make setup
cp .env.example .env
```

## Run

```bash
make run
```

The first start downloads the weights; later ones take a few seconds.

| URL | Contents |
|---|---|
| http://localhost:8000/docs | Swagger UI (with runnable examples) |
| http://localhost:8000/redoc | ReDoc documentation |
| http://localhost:8000/openapi.json | OpenAPI schema |

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/ready` | 200 once the weights are loaded, 503 otherwise |
| `GET` | `/v1/model` | Checkpoint, device and every effective option |
| `POST` | `/v1/forecast` | Forecast a batch of series |

### Example — univariate

```bash
curl -s localhost:8000/v1/forecast -H 'content-type: application/json' -d '{
  "horizon": 12,
  "series": [{"id": "sales", "values": [12.0, 13.5, 11.2, null, 14.8, 15.1]}]
}' | jq
```

`null` marks a missing value: TimesFM linearly interpolates internal gaps.

### Example — multivariate

A 2-D input `(variates, length)` turns on TimesFM 3.0's cross-variate attention:

```bash
curl -s localhost:8000/v1/forecast -H 'content-type: application/json' -d '{
  "horizon": 12,
  "series": [{"id": "sensors", "values": [[1.0, 1.2, 1.4], [8.0, 7.6, 7.9]]}]
}' | jq
```

### Example — covariates

```bash
curl -s localhost:8000/v1/forecast -H 'content-type: application/json' -d '{
  "horizon": 3,
  "series": [{
    "values": [100.0, 120.0, 90.0, 130.0],
    "past_only_covariates": [[5.0, 6.0, 4.0, 7.0]],
    "past_future_covariates": [[0, 0, 1, 1, 0, 0, 1]]
  }],
  "options": {"padding_mode": "edge"}
}' | jq
```

A future covariate covers the context **plus** the horizon. Since the model rounds the horizon
up to the next multiple of 64, `padding_mode: "edge"` extends the last value to that patch; the
API applies it automatically when you leave it unset, and reports it under `applied_options`.

### Response shape

The output mirrors the input:

| Input | `point` | `quantiles[level]` |
|---|---|---|
| 1-D `values` | `[horizon]` | `[horizon]` |
| 2-D `values` `(V, L)` | `[V][horizon]` | `[V][horizon]` |

The 9 quantiles run from `0.1` to `0.9`; `point` is the median. A non-finite value returned by
the model comes back as `null`.

## TimesFM 3.0 coverage

Every capability of the model is reachable through the API:

| TimesFM capability | Exposed as |
|---|---|
| Univariate forecasting | 1-D `values` |
| Multivariate forecasting (cross-variate attention) | 2-D `values` |
| Past-only covariates | `past_only_covariates` |
| Past **and** future covariates | `past_future_covariates` |
| Quantiles (9 levels) | `return_quantiles` |
| Z-normalization | `options.use_znorm` |
| Symmetric averaging | `options.use_symmetric_averaging` |
| Positivity constraint | `options.make_positive` |
| Quantile sorting | `options.sort_quantiles` |
| Univariate mode (unrolls variates) | `options.univariate` |
| Future-covariate padding | `options.padding_mode` |
| Automatic chunking beyond 32 variates | automatic |
| Series identifiers | `id` |
| Missing-value interpolation | `null` inside `values` |
| Model construction options (`use_stitching`, `use_linear_detrending`, `use_iterative_cpm_revin`, `use_variate_attention`, `input_transform`, `value_clip`, …) | `TIMESFM_*` variables, reported by `/v1/model` |

Those last options build the model itself, so they cannot vary per request: set them at startup.

**Not covered**: the Flax/JAX backend (the `flax` and `xreg` extras, which pull in `jax[cuda]`) —
unusable on Apple Silicon. The PyTorch backend offers the same features for 3.0.

## Model limits

| Parameter | Value |
|---|---|
| Maximum context | 15,360 points (longer inputs are truncated from the left by the model) |
| Input / output patch | 32 / 64 |
| Variates per forward pass | 32 (chunked automatically beyond that) |
| Quantiles | 0.1 … 0.9 |

API-side guardrails, all adjustable: `API_MAX_SERIES` (32), `API_MAX_HORIZON` (1024),
`API_MAX_VARIATES` (64).

## Configuration

Every option lives in [`.env.example`](.env.example). The main ones:

| Variable | Default | Purpose |
|---|---|---|
| `TIMESFM_CHECKPOINT` | `google/timesfm-3.0-pytorch` | Model to load |
| `TIMESFM_DEVICE` | `auto` | `auto` → `mps` on Apple Silicon, else `cuda`, else `cpu` |
| `TIMESFM_PER_CORE_BATCH_SIZE` | `4` | Internal batch size |
| `API_PRELOAD` | `true` | Load the weights at startup rather than on first request |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Route operations Metal does not implement to the CPU |

If initialization on `mps` fails, the service falls back to the CPU and says so in the logs.

### Measured latency (M1 Pro, 32 GB)

512-point context, horizon 64, one series:

| Device | Load | Inference |
|---|---|---|
| `mps` | 1.4 s | ~80 ms |
| `cpu` | 2.2 s | ~130 ms |

Inference is serialized behind a semaphore, so a burst of requests cannot saturate the GPU.

## Tests

```bash
make test
```

38 tests, no weights downloaded: the model is replaced by a double that checks every option
reaches `predict_batch`.

```bash
make test-model
```

Integration tests against the real weights: univariate and multivariate output shapes, quantile
ordering, future covariates, chunking beyond 32 variates, univariate mode.

```bash
make lint
```

## TypeScript client

The OpenAPI schema generates a typed client for the consumer:

```bash
make openapi
npx openapi-typescript openapi.json -o src/timesfm-api.d.ts
```

## Runtime notes

**Docker is not used.** Docker Desktop on Apple Silicon has no access to the Metal GPU, so
inference inside a container would be CPU-only, on top of the VM overhead. The service runs
natively instead.

## License

This repository is licensed under the **[Apache License 2.0](LICENSE)**.

Apache-2.0 was chosen because it is the license of the `timesfm` library itself, it carries an
explicit patent grant, and it is compatible with every dependency here — all of which are
permissive (Apache-2.0, MIT, BSD-3-Clause), with no copyleft obligation:

| Dependency | License |
|---|---|
| `timesfm` (library code) | Apache-2.0 |
| `torch` | BSD-3-Clause, with Apache-2.0 and other permissive components |
| `numpy` | BSD-3-Clause, with 0BSD, MIT, Zlib and CC0-1.0 components |
| `fastapi`, `pydantic`, `pydantic-settings`, `anyio` | MIT |
| `starlette`, `uvicorn` | BSD-3-Clause |
| `huggingface-hub`, `safetensors` | Apache-2.0 |
| `pytest`, `ruff` (dev) | MIT |
| `pytest-asyncio` (dev) | Apache-2.0 |
| `httpx` (dev) | BSD-3-Clause |

### The model weights are a separate matter

The TimesFM 3.0 **weights** are published by Google under the *TimesFM Non-Commercial License
v1.0* — **non-commercial, non-production use only**. They are **not** part of this repository:
they are downloaded from HuggingFace at runtime.

The Apache-2.0 license above covers the code written here and grants no rights over those
weights. Whoever runs this service is responsible for complying with Google's terms for the
checkpoint they load. See [NOTICE](NOTICE).

TimesFM 2.5 weights are Apache-2.0 and carry no such restriction, but its Python API differs
(`timesfm` instead of `timesfm3`): `app/forecaster.py` would need a dedicated adapter.
