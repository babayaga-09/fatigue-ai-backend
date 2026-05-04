from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch, torch.nn as nn
import numpy as np, json, httpx, asyncio, os, pickle
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Config ──────────────────────────────────────────────────
FOOTBALL_API_KEY = "d5c336e36297444dbb69a9db9ef70a39"
FOOTBALL_BASE    = "https://api.football-data.org/v4"
CURRENT_SEASON   = 2024

LEAGUE_IDS = {
    "PL": 2021, "PD": 2014,
    "BL1": 2002, "SA": 2019, "FL1": 2015,
}
HEADERS = {"X-Auth-Token": FOOTBALL_API_KEY}

# ── LSTM Model ───────────────────────────────────────────────
class GlobalFatigueLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, pos_embed_dim=4):
        super().__init__()
        self.pos_embed = nn.Embedding(4, pos_embed_dim)
        self.lstm = nn.LSTM(
            input_size=input_size + pos_embed_dim,
            hidden_size=hidden_size,
            num_layers=2, batch_first=True, dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(32, 1),
        )
    def forward(self, x, pos_idx):
        emb = self.pos_embed(pos_idx).unsqueeze(1).expand(-1, x.size(1), -1)
        out, _ = self.lstm(torch.cat([x, emb], dim=-1))
        return self.head(out[:, -1, :])

# ── Load all models ──────────────────────────────────────────
lstm_model = GlobalFatigueLSTM()
lstm_model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "global_fatigue_lstm.pth"),
        map_location="cpu"
    )
)
lstm_model.eval()

with open(os.path.join(BASE_DIR, "injury_model.pkl"), "rb") as f:
    injury_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "form_model.pkl"), "rb") as f:
    form_model = pickle.load(f)

with open(os.path.join(BASE_DIR, "scaler_params.json")) as f:
    sp = json.load(f)

X_min      = np.array(sp["X_min"], dtype=np.float32)
X_max      = np.array(sp["X_max"], dtype=np.float32)
y_min      = sp["y_min"][0]
y_max      = sp["y_max"][0]
POS_IDX    = sp["pos_to_idx"]
SEQ_LEN    = sp["sequence_length"]
N_FEAT     = sp["n_features"]
WLOAD_75   = sp["workload_75"]
REST_25    = sp["rest_25"]

# ── FastAPI ──────────────────────────────────────────────────
app = FastAPI(title="Fatigue AI v2")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

# ── Schemas ──────────────────────────────────────────────────
class MatchWindow(BaseModel):
    sequence:  List[List[float]]
    position:  str
    rest_days: int = 4

# ── Helpers ──────────────────────────────────────────────────
def _difficulty(rank: Optional[int]) -> str:
    if rank is None: return "moderate"
    if rank <= 4:    return "hard"
    if rank <= 10:   return "moderate"
    return "easy"

def _difficulty_score(rank: Optional[int]) -> int:
    if rank is None: return 3
    if rank <= 4:    return 5
    if rank <= 7:    return 4
    if rank <= 12:   return 3
    if rank <= 16:   return 2
    return 1

def _map_position(raw: str) -> str:
    if not raw: return "MID"
    raw = raw.lower()
    if any(x in raw for x in ["goalkeeper","keeper"]): return "GK"
    if any(x in raw for x in ["back","defender","defence"]): return "DEF"
    if any(x in raw for x in ["forward","winger","striker","attacker"]): return "FWD"
    return "MID"

def _estimate_passes(pos): return {"GK":35,"DEF":55,"MID":65,"FWD":30}.get(pos,50)
def _estimate_def(pos):    return {"GK":2,"DEF":6,"MID":4,"FWD":1}.get(pos,3)
def _estimate_acc(pos, result):
    base = {"GK":72.0,"DEF":80.0,"MID":84.0,"FWD":76.0}.get(pos,80.0)
    return round(base + (2.0 if result=="W" else -1.5 if result=="L" else 0), 1)

def _compute_fitness(acc_pct, injury_prob, form_momentum, pos) -> float:
    """Combine all signals into a 0-100 fitness score."""
    baseline = {"GK":72,"DEF":80,"MID":84,"FWD":76}.get(pos, 80)
    acc_score    = min(acc_pct / baseline, 1.0) * 40       # 40 pts
    safety_score = (1 - injury_prob) * 30                  # 30 pts
    form_score   = min(form_momentum / baseline, 1.0) * 30 # 30 pts
    return round(acc_score + safety_score + form_score, 1)

def _compute_match_rating(result, passes, def_actions, acc, pos) -> float:
    """Per-match 0-10 player rating."""
    base = {"GK":6.0,"DEF":6.2,"MID":6.5,"FWD":6.3}.get(pos, 6.3)
    bonus = (2.0 if result=="W" else -0.5 if result=="L" else 0)
    pass_bonus = min(passes / 70, 1.0) * 1.0
    acc_bonus  = min(acc / 90, 1.0) * 1.0
    def_bonus  = min(def_actions / 8, 1.0) * 0.5
    return round(min(base + bonus + pass_bonus + acc_bonus + def_bonus, 10), 1)

# ── HTTP helper ───────────────────────────────────────────────
async def _fd_get(path: str, params: dict = None) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{FOOTBALL_BASE}{path}",
            headers=HEADERS,
            params=params or {},
            timeout=12,
        )
        print(f"GET {path} {params} → {r.status_code}")
        if r.status_code != 200:
            print(f"Body: {r.text[:300]}")
        r.raise_for_status()
        return r.json()

_standings_cache: dict = {}
_standings_ts:    dict = {}

async def get_standings(league_code: str) -> List[dict]:
    now = datetime.utcnow()
    if league_code in _standings_cache:
        if (now - _standings_ts[league_code]).seconds < 3600:
            return _standings_cache[league_code]
    data  = await _fd_get(
        f"/competitions/{LEAGUE_IDS[league_code]}/standings",
        {"season": CURRENT_SEASON}
    )
    table = data["standings"][0]["table"]
    _standings_cache[league_code] = table
    _standings_ts[league_code]    = now
    return table

async def team_rank(league_code: str, team_id: int) -> Optional[int]:
    try:
        for e in await get_standings(league_code):
            if e["team"]["id"] == team_id:
                return e["position"]
    except Exception:
        pass
    return None

# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models": ["lstm", "xgboost_injury", "gbm_form"]}

@app.get("/leagues")
async def list_leagues():
    return [
        {"code": k, "name": v, "id": LEAGUE_IDS[k]}
        for k, v in {
            "PL":"Premier League","PD":"La Liga",
            "BL1":"Bundesliga","SA":"Serie A","FL1":"Ligue 1",
        }.items()
    ]

@app.get("/teams")
async def list_teams(league: str = Query("PL")):
    lid = LEAGUE_IDS.get(league)
    if not lid: raise HTTPException(400, "Unknown league")
    try:
        data = await _fd_get(
            f"/competitions/{lid}/teams", {"season": CURRENT_SEASON}
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    return [
        {"id": t["id"], "name": t["name"],
         "crest": t.get("crest",""), "tla": t.get("tla","")}
        for t in data["teams"]
    ]

@app.get("/players")
async def search_players(team_id: int):
    data = await _fd_get(f"/teams/{team_id}")
    return [
        {
            "id":           p["id"],
            "name":         p["name"],
            "position":     p.get("position","Midfielder"),
            "nationality":  p.get("nationality",""),
            "shirt_number": p.get("shirtNumber"),
        }
        for p in data.get("squad", [])
    ]

@app.get("/player/{player_id}")
async def player_profile(player_id: int, league: str = Query("PL")):
    try:
        p_data = await _fd_get(f"/persons/{player_id}")
    except Exception:
        raise HTTPException(404, "Player not found")

    current_team = p_data.get("currentTeam") or {}
    team_id      = current_team.get("id")
    raw_pos      = p_data.get("position") or "Midfielder"
    pos_group    = _map_position(raw_pos)
    today        = datetime.utcnow().date()

    # ── Recent matches ────────────────────────────────────────
    finished = []
    if team_id:
        for attempt in [
            {"season": CURRENT_SEASON, "status": "FINISHED", "limit": 10},
            {"season": 2024,           "status": "FINISHED", "limit": 10},
            {"dateFrom": str(today - timedelta(days=180)),
             "dateTo": str(today), "limit": 10},
        ]:
            try:
                data = await _fd_get(f"/teams/{team_id}/matches", attempt)
                finished = [m for m in data.get("matches",[])
                            if m.get("status") == "FINISHED"]
                if finished: break
            except Exception:
                continue
        finished = finished[-5:]

    recent_matches = []
    ratings        = []
    for m in finished:
        try:
            home    = m["homeTeam"]
            away    = m["awayTeam"]
            s       = m.get("score",{}).get("fullTime",{})
            is_home = home["id"] == team_id
            opp     = away["name"] if is_home else home["name"]
            gf      = s.get("home") if is_home else s.get("away")
            ga      = s.get("away") if is_home else s.get("home")
            result  = ("W" if (gf or 0)>(ga or 0)
                       else "D" if gf==ga else "L")
            passes  = _estimate_passes(pos_group)
            defact  = _estimate_def(pos_group)
            acc     = _estimate_acc(pos_group, result)
            rating  = _compute_match_rating(result, passes, defact, acc, pos_group)
            ratings.append(rating)
            recent_matches.append({
                "opponent":       opp,
                "score":          f"{gf}–{ga}" if gf is not None else "–",
                "result":         result,
                "date":           m["utcDate"][:10],
                "home_away":      "H" if is_home else "A",
                "passes_made":    passes,
                "def_actions":    defact,
                "completion_pct": acc,
                "match_rating":   rating,
            })
        except Exception:
            continue

    # Placeholders if no real data
    if not recent_matches:
        for i in range(5):
            acc    = _estimate_acc(pos_group, "D")
            rating = _compute_match_rating("D",
                _estimate_passes(pos_group),
                _estimate_def(pos_group), acc, pos_group)
            ratings.append(rating)
            recent_matches.append({
                "opponent":"–","score":"–","result":"D",
                "date": str(today - timedelta(days=(i+1)*7)),
                "home_away":"H",
                "passes_made":    _estimate_passes(pos_group),
                "def_actions":    _estimate_def(pos_group),
                "completion_pct": acc,
                "match_rating":   rating,
            })

    # ── Season stats (from football-data scorers endpoint) ────
    season_stats = {
        "goals": 0, "assists": 0, "yellow_cards": 0, "red_cards": 0,
        "appearances": len(recent_matches),
        "avg_rating": round(np.mean(ratings), 2) if ratings else 6.0,
    }
    try:
        scorer_data = await _fd_get(
            f"/competitions/{LEAGUE_IDS.get(league, 2021)}/scorers",
            {"season": CURRENT_SEASON, "limit": 100}
        )
        for scorer in scorer_data.get("scorers", []):
            if scorer.get("player", {}).get("id") == player_id:
                season_stats["goals"]   = scorer.get("goals", 0) or 0
                season_stats["assists"] = scorer.get("assists", 0) or 0
                break
    except Exception:
        pass

    # ── Next fixture ──────────────────────────────────────────
    next_match   = None
    fixture_info = None
    if team_id:
        for attempt in [
            {"season": CURRENT_SEASON, "status": "SCHEDULED", "limit": 5},
            {"season": CURRENT_SEASON, "status": "TIMED",     "limit": 5},
            {"dateFrom": str(today),
             "dateTo": str(today+timedelta(days=60)), "limit": 5},
        ]:
            try:
                data = await _fd_get(f"/teams/{team_id}/matches", attempt)
                upcoming = data.get("matches", [])
                if upcoming:
                    next_match = upcoming[0]
                    break
            except Exception:
                continue

    if next_match:
        try:
            is_home  = next_match["homeTeam"]["id"] == team_id
            opp_id   = (next_match["awayTeam"]["id"] if is_home
                        else next_match["homeTeam"]["id"])
            opp_name = (next_match["awayTeam"]["name"] if is_home
                        else next_match["homeTeam"]["name"])
            opp_rank = await team_rank(league, opp_id)
            fixture_info = {
                "opponent":         opp_name,
                "date":             next_match["utcDate"][:10],
                "competition":      next_match.get(
                                        "competition",{}).get("name",""),
                "home_away":        "H" if is_home else "A",
                "difficulty":       _difficulty(opp_rank),
                "difficulty_score": _difficulty_score(opp_rank),
                "opponent_rank":    opp_rank,
            }
        except Exception:
            fixture_info = None

    return {
        "id":             player_id,
        "name":           p_data.get("name",""),
        "position":       raw_pos,
        "pos_group":      pos_group,
        "nationality":    p_data.get("nationality",""),
        "date_of_birth":  p_data.get("dateOfBirth"),
        "team":           current_team.get("name",""),
        "team_crest":     current_team.get("crest",""),
        "recent_matches": recent_matches,
        "next_fixture":   fixture_info,
        "season_stats":   season_stats,
    }

@app.post("/predict")
async def predict(body: MatchWindow):
    seq = np.array(body.sequence, dtype=np.float32)
    if seq.shape != (SEQ_LEN, N_FEAT):
        raise HTTPException(
            422, f"Need ({SEQ_LEN},{N_FEAT}), got {seq.shape}"
        )

    # ── LSTM: pass accuracy ───────────────────────────────────
    seq_scaled  = (seq - X_min) / (X_max - X_min + 1e-8)
    tensor_in   = torch.FloatTensor(seq_scaled).unsqueeze(0)
    pos_tensor  = torch.LongTensor([POS_IDX.get(body.position, 2)])
    with torch.no_grad():
        pred_scaled = lstm_model(tensor_in, pos_tensor).item()
    acc_pct = float(np.clip(
        pred_scaled * (y_max - y_min) + y_min, 0, 100
    ))

    # ── XGBoost: injury risk ──────────────────────────────────
    last_match     = seq[-1]  # most recent match features
    workload_score = (
        last_match[0] * 0.3 +   # passes
        last_match[1] * 0.4 +   # def_actions
        last_match[2] * 0.2 +   # carries
        last_match[5] * 0.5     # fouls
    )
    injury_features = np.array([[
        last_match[0], last_match[1], last_match[2],
        last_match[3], last_match[5], last_match[6],
        workload_score,
    ]])
    injury_prob = float(
        injury_model.predict_proba(injury_features)[0][1]
    )

    # ── GBM: form momentum ────────────────────────────────────
    form_features = np.array([[
        last_match[0], acc_pct,
        last_match[1], last_match[6], workload_score,
    ]])
    form_momentum = float(form_model.predict(form_features)[0])
    form_momentum = float(np.clip(form_momentum, 0, 100))

    # ── Fitness score (ensemble) ──────────────────────────────
    fitness = _compute_fitness(
        acc_pct, injury_prob, form_momentum, body.position
    )

    baseline = {"GK":72,"DEF":80,"MID":84,"FWD":76}.get(body.position,80)
    drop     = round(baseline - acc_pct, 1)
    fatigue  = "Low" if drop<=2 else "Moderate" if drop<=6 else "High"

    injury_label = (
        "High Risk"   if injury_prob > 0.6 else
        "Medium Risk" if injury_prob > 0.3 else
        "Low Risk"
    )
    form_label = (
        "In Form"      if form_momentum >= baseline else
        "Dipping"      if form_momentum >= baseline*0.95 else
        "Out of Form"
    )

    return {
        "predicted_accuracy_pct": round(acc_pct, 1),
        "fatigue_level":          fatigue,
        "performance_drop_pct":   max(drop, 0.0),
        "baseline_pct":           float(baseline),
        "position":               body.position,
        "injury_risk_pct":        round(injury_prob * 100, 1),
        "injury_label":           injury_label,
        "form_momentum_pct":      round(form_momentum, 1),
        "form_label":             form_label,
        "fitness_score":          fitness,
    }

@app.get("/compare")
async def compare_players(
    player_a: int = Query(...),
    player_b: int = Query(...),
    league:   str = Query("PL"),
):
    a, b = await asyncio.gather(
        player_profile(player_a, league),
        player_profile(player_b, league),
    )
    return {"player_a": a, "player_b": b}

# ── Utility ───────────────────────────────────────────────────
def _map_position(raw: str) -> str:
    if not raw: return "MID"
    raw = raw.lower()
    if any(x in raw for x in ["goalkeeper","keeper"]): return "GK"
    if any(x in raw for x in ["back","defender","defence"]): return "DEF"
    if any(x in raw for x in ["forward","winger","striker","attacker"]): return "FWD"
    return "MID"