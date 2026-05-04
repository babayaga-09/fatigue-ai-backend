# Fatigue AI — Backend

FastAPI server powering the Fatigue AI footballer performance predictor.

## Models
- **PyTorch LSTM** — predicts pass accuracy from 3-match workload sequences
- **XGBoost** — classifies injury risk from workload + rest features  
- **Gradient Boosting** — predicts form momentum trend

## Endpoints
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/leagues` | Top 5 European leagues |
| GET | `/teams?league=PL` | Teams in a league |
| GET | `/players?team_id=57` | Squad for a team |
| GET | `/player/{id}` | Full player profile + fixtures |
| POST | `/predict` | Run all 3 ML models |
| GET | `/compare` | Side-by-side player comparison |

## Stack
Python · FastAPI · PyTorch · XGBoost · scikit-learn · football-data.org API

## Setup
```bash
pip install fastapi uvicorn torch xgboost scikit-learn httpx numpy
python retrain_local.py        # regenerate model files
python -m uvicorn main:app --reload --port 8000
```

> Model files (.pth, .pkl) are not included in this repo due to size.  
> Run `retrain_local.py` to generate them locally.
