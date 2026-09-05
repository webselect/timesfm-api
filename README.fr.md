# timesfm-api

*[English version](README.md) — la documentation de référence est en anglais.*

Service REST autour de **[TimesFM 3.0](https://github.com/google-research/timesfm)** (Google
Research) pour la prévision de séries temporelles, avec Swagger intégré.

Construit pour être appelé depuis un client externe (par exemple un robot de trading en
TypeScript) : l'API est générique, sans logique métier, et expose **toute** la surface du
modèle.

---

## Prérequis

- macOS Apple Silicon (testé sur M1 Pro) ou Linux
- Python ≥ 3.10 — le projet utilise `/opt/homebrew/bin/python3.12`
- ~1,2 Go d'espace disque pour les poids (téléchargés depuis HuggingFace au premier démarrage)

## Installation

```bash
make setup
cp .env.example .env
```

## Démarrage

```bash
make run
```

Le premier lancement télécharge les poids ; les suivants démarrent en quelques secondes.

| URL | Contenu |
|---|---|
| http://localhost:8000/docs | Swagger UI (« Try it out » utilisable directement) |
| http://localhost:8000/redoc | Documentation ReDoc |
| http://localhost:8000/openapi.json | Schéma OpenAPI |

## Endpoints

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/ready` | 200 si les poids sont chargés, 503 sinon |
| `GET` | `/v1/model` | Checkpoint, device et toutes les options effectives |
| `POST` | `/v1/forecast` | Prévision d'un lot de séries |

### Exemple — univarié

```bash
curl -s localhost:8000/v1/forecast -H 'content-type: application/json' -d '{
  "horizon": 12,
  "series": [{"id": "ventes", "values": [12.0, 13.5, 11.2, null, 14.8, 15.1]}]
}' | jq
```

`null` marque une valeur manquante : TimesFM interpole linéairement les trous internes.

### Exemple — multivarié

Une entrée 2-D `(variables, longueur)` active l'attention inter-variables de TimesFM 3.0 :

```bash
curl -s localhost:8000/v1/forecast -H 'content-type: application/json' -d '{
  "horizon": 12,
  "series": [{"id": "capteurs", "values": [[1.0, 1.2, 1.4], [8.0, 7.6, 7.9]]}]
}' | jq
```

### Exemple — covariables

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

Une covariable future couvre le contexte **plus** l'horizon. Comme le modèle arrondit l'horizon
au multiple de 64 supérieur, `padding_mode: "edge"` prolonge la dernière valeur jusqu'à ce
patch ; l'API l'applique automatiquement si vous ne précisez rien, et le signale dans
`applied_options`.

### Forme des réponses

La sortie reflète l'entrée :

| Entrée | `point` | `quantiles[niveau]` |
|---|---|---|
| `values` 1-D | `[horizon]` | `[horizon]` |
| `values` 2-D `(V, L)` | `[V][horizon]` | `[V][horizon]` |

Les 9 quantiles vont de `0.1` à `0.9` ; `point` est la médiane. Une valeur non finie renvoyée
par le modèle apparaît en `null`.

## Couverture de TimesFM 3.0

Toute la surface du modèle est atteignable via l'API :

| Capacité TimesFM | Exposition |
|---|---|
| Prévision univariée | `values` 1-D |
| Prévision multivariée (attention inter-variables) | `values` 2-D |
| Covariables passées | `past_only_covariates` |
| Covariables passées **et** futures | `past_future_covariates` |
| Quantiles (9 niveaux) | `return_quantiles` |
| Z-normalisation | `options.use_znorm` |
| Moyenne symétrique | `options.use_symmetric_averaging` |
| Contrainte de positivité | `options.make_positive` |
| Tri des quantiles | `options.sort_quantiles` |
| Mode univarié (déroulage des variables) | `options.univariate` |
| Padding des covariables futures | `options.padding_mode` |
| Découpage automatique au-delà de 32 variables | automatique |
| Identifiants de séries | `id` |
| Interpolation des valeurs manquantes | `null` dans `values` |
| Options de construction du modèle (`use_stitching`, `use_linear_detrending`, `use_iterative_cpm_revin`, `use_variate_attention`, `input_transform`, `value_clip`, …) | variables `TIMESFM_*`, visibles sur `/v1/model` |

Ces dernières construisent le modèle et ne peuvent donc pas varier d'une requête à l'autre :
elles se règlent au démarrage.

**Non couvert** : le backend Flax/JAX (extras `flax` et `xreg`, qui tirent `jax[cuda]`) —
inutilisable sur Apple Silicon. Le backend PyTorch offre les mêmes fonctionnalités pour la 3.0.

## Limites du modèle

| Paramètre | Valeur |
|---|---|
| Contexte maximum | 15 360 points (au-delà, tronqué à gauche par le modèle) |
| Patch d'entrée / de sortie | 32 / 64 |
| Variables par passe | 32 (découpage automatique au-delà) |
| Quantiles | 0.1 … 0.9 |

Garde-fous côté API, ajustables : `API_MAX_SERIES` (32), `API_MAX_HORIZON` (1024),
`API_MAX_VARIATES` (64).

## Configuration

Toutes les options sont dans [`.env.example`](.env.example). Les principales :

| Variable | Défaut | Rôle |
|---|---|---|
| `TIMESFM_CHECKPOINT` | `google/timesfm-3.0-pytorch` | Modèle chargé |
| `TIMESFM_DEVICE` | `auto` | `auto` → `mps` sur Apple Silicon, sinon `cuda`, sinon `cpu` |
| `TIMESFM_PER_CORE_BATCH_SIZE` | `4` | Taille de lot interne |
| `API_PRELOAD` | `true` | Charger les poids au démarrage plutôt qu'à la première requête |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | Bascule sur CPU les opérations non implémentées en Metal |

Si l'initialisation sur `mps` échoue, le service se rabat automatiquement sur le CPU et le
signale dans les logs.

### Latence mesurée (M1 Pro, 32 Go)

Contexte de 512 points, horizon 64, une série :

| Device | Chargement | Inférence |
|---|---|---|
| `mps` | 1,4 s | ~80 ms |
| `cpu` | 2,2 s | ~130 ms |

L'inférence est sérialisée par un sémaphore : une rafale de requêtes ne sature pas le GPU.

## Tests

```bash
make test
```

38 tests, sans téléchargement de poids : le modèle est remplacé par un double qui vérifie que
chaque option est bien transmise à `predict_batch`.

```bash
make test-model
```

Tests d'intégration avec les vrais poids : formes de sortie univariée et multivariée, ordre des
quantiles, covariables futures, découpage au-delà de 32 variables, mode univarié.

```bash
make lint
```

## Client TypeScript

Le schéma OpenAPI permet de générer un client typé pour le consommateur :

```bash
make openapi
npx openapi-typescript openapi.json -o src/timesfm-api.d.ts
```

## Notes d'exécution

**Docker n'est pas utilisé.** Docker Desktop sur Apple Silicon ne donne pas accès au GPU Metal :
l'inférence y serait CPU pur, avec en plus le surcoût de la VM. Le service tourne donc en natif.

## Licence

Le code de ce dépôt est libre d'usage. En revanche, les **poids** de TimesFM 3.0 sont distribués
par Google sous *TimesFM Non-Commercial License v1.0* — usage **non commercial et hors
production**. Le code de la librairie `timesfm` est, lui, sous Apache-2.0.

Pour un usage commercial, il faut se rabattre sur TimesFM 2.5 (poids Apache-2.0), dont l'API
Python diffère (`timesfm` au lieu de `timesfm3`) : `app/forecaster.py` devrait alors recevoir un
adaptateur dédié.
