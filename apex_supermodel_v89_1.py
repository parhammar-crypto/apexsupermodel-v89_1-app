# apex_supermodel_v89_1.py
# -*- coding: utf-8 -*-
"""
ApexEnsemble SuperModel V89.1
=============================
Uppgradering av V89 med:
- Set-piece & stil-proxys (hörnor, skott, fouls, kort) rullande for/against, mismatch-index.
- “Pace” (skott-tempo) och “Aggression” (fouls/kort) som features i Upset-ensemblen.
- Policy: mild relax av AH-filter vid stark set-piece-mismatch för underdog; hårdare block av BTTS vid låg pace + stark GK.
- I övrigt: samma konformala LBs, 4D-kalibrering och MoE som i v89.

Backtest: 2020–23 train, 2024 test (default).
"""

import os, re, io, sys, math, json, time, pickle, warnings, random
import datetime as dt
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
EPS = 1e-12

# -----------------------------------------------------------------------------
# API‑Football Integration
#
# För att utöka datakällorna med live‑ och historikstatistik från API‑Football
# (API‑Sports) läggs konfigurationskonstanter och hjälpfunktioner till.  Detta
# gör det möjligt att hämta matcher, lagstatistik och fixtures‑statistik via
# API‑Sports utan att störa befintliga datakällor (Understat och
# football‑data.co.uk).  API‑nyckeln läses från miljövariabeln
# `API_FOOTBALL_KEY` för att undvika att hårdkoda känsliga uppgifter.
#
# Som standard används den direkta API‑Sports‑ingången (`v3.football.api-sports.io`).
# Om du hellre vill använda RapidAPI (api-football-v1.p.rapidapi.com) kan du sätta
# `API_USE_RAPIDAPI = True` och ange `API_FOOTBALL_HOST` i miljön.
# -----------------------------------------------------------------------------

# Bas‑URL för API‑Sports (direktåtkomst).  För RapidAPI krävs annan host.
API_BASE_URL: str = "https://v3.football.api-sports.io"
# Flagga för att använda RapidAPI istället för API‑Sports.  RapidAPI kräver
# ytterligare header x‑rapidapi-host.  Standard är False för API‑Sports.
API_USE_RAPIDAPI: bool = False
# Värdnamn för RapidAPI.  Om du använder RapidAPI, sätt miljövariabeln
# API_FOOTBALL_HOST till t.ex. "api-football-v1.p.rapidapi.com".
API_HOST: str = os.environ.get("API_FOOTBALL_HOST", "api-football-v1.p.rapidapi.com")
# Din API‑nyckel.  För att köra modellen helt autonomt utan miljövariabler
# har nyckeln skrivits in direkt här.  Observera att det här inte är säkert
# att dela offentligt.  Den här nyckeln används med API‑Sports (direkt
# ingång).  Om du vill byta nyckel senare behöver du ändra denna rad.
API_KEY: str = "284b9691ee2c6923934d37135a975a2f"

# Mapping mellan liganamn och API‑Sports‑liga‑ID:n.  Dessa används när
# fixtures och statistik hämtas via API‑Sports.  Listan innehåller de
# största europeiska ligorna – utöka vid behov.
API_LEAGUE_IDS: Dict[str, int] = {
    "Premier League": 39,
    "LaLiga": 140,
    "Serie A": 135,
    "Ligue 1": 61,
    "Bundesliga": 78,
    "Eredivisie": 88,
    "Primeira Liga": 94,
    "Jupiler Pro League": 144,
}

# ML
try:
    import lightgbm as lgb
    _HAVE_LGB = True
except Exception:
    _HAVE_LGB = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

try:
    import requests
except Exception:
    requests = None

from concurrent.futures import ThreadPoolExecutor, as_completed

# ====================== Global Config ======================

TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEAR  = 2024

CACHE_ROOT = "./apex_cache_super_v89_1"
os.makedirs(CACHE_ROOT, exist_ok=True)

MAX_GOALS = 8

HIGH_THRESHOLD = 0.75
MEDIUM_RANGE   = (0.58, 0.75)

FD_CODES = {
    "Premier League":"E0","LaLiga":"SP1","Serie A":"I1","Ligue 1":"F1","Bundesliga":"D1",
    "Eredivisie":"N1","Primeira Liga":"P1","Jupiler Pro League":"B1"
}
LEAGUE_MAP = {
    "EPL":"Premier League","La liga":"LaLiga","La Liga":"LaLiga","LaLiga":"LaLiga",
    "Bundesliga":"Bundesliga","Serie A":"Serie A","Ligue 1":"Ligue 1",
    "Eredivisie":"Eredivisie","Primeira Liga":"Primeira Liga","Jupiler Pro League":"Jupiler Pro League"
}
RED_LEAGUES = {"LaLiga","Ligue 1","Eredivisie","Jupiler Pro League","Primeira Liga"}

TTL_DAYS_RECENT = 7
TTL_DAYS_STATIC = 3650

STAKE_CFG = {
    "kelly_frac": 0.25,
    "unit_cap_high": 1.50,
    "unit_cap_med":  0.90,
    "league_unit_adj": {
        "Bundesliga": +0.30,
        "Serie A":    +0.20,
        "LaLiga":     -0.30,
        "Ligue 1":    -0.50,
        "Premier League": -0.20,
        "Eredivisie": -0.10,
        "Primeira Liga": -0.10,
        "Jupiler Pro League": -0.20
    }
}

# Syntetiskt AH-pris för backtest.
SYNTH_AH_PRICING = {
    "ENABLE": True,  # sätt False för live
    "LEAGUE_MARGIN": {
        "Premier League": 0.055, "LaLiga": 0.060, "Serie A": 0.055, "Ligue 1": 0.060,
        "Bundesliga": 0.055, "Eredivisie": 0.060, "Primeira Liga": 0.060, "Jupiler Pro League": 0.060
    },
    "ODDS_CAP": 5.5
}

GUARD = {
    "knife_edge_gap_pp": 5.0,
    "knife_edge_min_px": 0.32,
    "draw_propensity": {
        "LaLiga": {"maxprob_cap": 0.72, "px_min": 0.29, "mu_total_max": 2.45, "boost_range": [0.014, 0.026]},
        "Ligue 1": {"maxprob_cap": 0.73, "px_min": 0.29, "mu_total_max": 2.40, "boost_range": [0.014, 0.026]},
        "Serie A": {"maxprob_cap": 0.74, "px_min": 0.28, "mu_total_max": 2.50, "boost_range": [0.012, 0.022]},
        "Bundesliga": {"maxprob_cap": 0.72, "px_min": 0.27, "mu_total_max": 2.55, "boost_range": [0.008, 0.014]},
        "Premier League":{"maxprob_cap": 0.71, "px_min": 0.27, "mu_total_max": 2.50, "boost_range": [0.008, 0.014]},
        "Primeira Liga":{"maxprob_cap": 0.73, "px_min": 0.28, "mu_total_max": 2.45, "boost_range": [0.010, 0.018]},
        "Eredivisie": {"maxprob_cap": 0.71, "px_min": 0.26, "mu_total_max": 2.70, "boost_range": [0.008, 0.014]},
        "Jupiler Pro League":{"maxprob_cap": 0.72, "px_min": 0.27, "mu_total_max": 2.55, "boost_range": [0.008, 0.014]}
    },
    "market_blend": {"enable": True, "abs_gap_pp": 18.0, "blend_w": 0.35},

    "medium_gate": {
        "upset_win_lb_min": 0.20,
        "upset_cover_lb_min": 0.55,
        "late_drift_bps_max": 15.0
    },

    "late_drift_guard": {"window_hours": 6.0, "drift_bps": 15.0},

    "fav_bigline_gate": {
        "mu_total_min": 2.95,
        "upset_cover_lb_min": 0.62
    },

    "btts_gate": {
        "p_min": 0.64,
        "mu_total_min": 2.60,
        "gk_soft_block": True
    }
}

OVERDISP = {
    "Ligue 1":0.22,"Eredivisie":0.28,"Bundesliga":0.15,
    "Serie A":0.06,"LaLiga":0.10,"Premier League":0.14,
    "Primeira Liga":0.12,"Jupiler Pro League":0.14
}

# ---- Stil/set-piece trösklar ----
STYLE_CFG = {
    "sp_mismatch_pp": 0.18,     # rullande corners_for minus corners_against (normaliserad) – tröskel för "stark mismatch"
    "pace_low": 7.8,            # låg match-pace (sum skott per lag senaste rull) ~ per match
    "pace_high": 12.5,          # hög pace
    "allow_ah_relax_delta": 0.03,  # sänkning av upset_cover_lb_min vid stark SP-mismatch
    "gk_block_extra_when_lowpace": True
}

# ===================== Utils =====================

def clamp(x, lo, hi): return max(lo, min(hi, x))
def normalize_probs(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1.0); s = float(p.sum())
    return p/s if s>0 else np.array([1/3,1/3,1/3], float)
def overround_correction(odds):
    if (not odds) or any((o is None) or (not np.isfinite(o)) or (o<=1e-9) for o in odds):
        return np.array([1/3,1/3,1/3], float)
    imps = np.array([1.0/float(o) for o in odds], float); s = float(imps.sum())
    return imps/s if s>0 else np.array([1/3,1/3,1/3], float)
def fuzzy(s): return re.sub(r"[^a-z0-9]+","", (s or "").lower())
def prob_entropy(p): p=np.clip(np.asarray(p,float),1e-12,1.0); return float(-np.sum(p*np.log(p)))
def top2_gap(p): q=np.sort(np.asarray(p,float)); return float(q[-1]-q[-2]) if q.size>=2 else 0.0
def book_overround(odds):
    try: imps=np.array([1.0/float(o) for o in odds], float); return float(imps.sum())-1.0
    except Exception: return 0.0

# =================== NB goal-matrix ===================

def nb_pmf(k, mu, rshape):
    from math import lgamma
    k=int(max(0,k)); r=max(float(rshape),1e-6); mu=max(float(mu),1e-6)
    p=r/(r+mu); logpmf=lgamma(k+r)-lgamma(r)-lgamma(k+1)+r*math.log(p)+k*math.log(1-p)
    return math.exp(logpmf)

def dc_corr(h,a,rho=0.05):
    if h==0 and a==0: return 1-rho
    if (h,a) in [(0,1),(1,0)]: return 1+rho/2
    if h==1 and a==1: return 1-rho/2
    return 1.0

def joint_nb(mu_h, mu_a, rdisp, H=MAX_GOALS):
    H1=H+1; M=np.zeros((H1,H1),float); rshape=1.0/max(rdisp,1e-6)
    for h in range(H1):
        ph=nb_pmf(h, mu_h, rshape)
        for a in range(H1):
            pa=nb_pmf(a, mu_a, rshape); M[h,a]=ph*pa*dc_corr(h,a,0.05)
    s=float(M.sum()); return M/s if s>0 else np.ones((H1,H1),float)/(H1*H1)

def one_x_two_from_joint(M):
    p1=float(np.sum(np.tril(M,-1))); px=float(np.sum(np.diag(M))); p2=float(np.sum(np.triu(M,1)))
    return normalize_probs([p1,px,p2])

def btts_from_joint(M):
    p_h0=float(np.sum(M[0,:])); p_a0=float(np.sum(M[:,0])); p_00=float(M[0,0])
    return clamp(1.0-p_h0-p_a0+p_00,0.0,1.0)

def under25_from_joint(M):
    return float(np.sum([M[i,j] for i in range(M.shape[0]) for j in range(M.shape[1]) if (i+j)<=2]))

# =================== Data Manager ===================

class QuantumDataManager:
    UA = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36"
    ]
    def __init__(self, cache_root: str, timeout: float = 15.0, rate_limit: float = 1.6):
        self.cache_root = cache_root; self.timeout=timeout; self.rate_limit=rate_limit
        self.session = requests.Session() if requests else None
        if self.session: self.session.headers.update({"User-Agent": random.choice(self.UA)})
    def _sleep(self): time.sleep(self.rate_limit + np.random.random()*0.6)
    def _cache_path(self, league, season, tag):
        safe = re.sub(r"[^A-Za-z0-9]+","", league)
        return os.path.join(self.cache_root, f"{tag}_{safe}_{season}.parquet")
    def _get(self, url, retries=3):
        if not self.session: return None
        for _ in range(retries):
            try:
                r = self.session.get(url, timeout=self.timeout)
                if r.status_code==200 and r.content:
                    return r
            except Exception: pass
            if self.session: self.session.headers.update({"User-Agent": random.choice(self.UA)})
            self._sleep()
        return None

    # ---------------------------------------------------------------------
    # API‑Football helpers
    #
    # Följande metoder möjliggör åtkomst till API‑Football (API‑Sports) för
    # fixtures, statistik och lagdata.  De använder instansens timeout
    # och rate‑limit för att undvika att överskrida API:ets begränsningar.

    def _get_api(self, endpoint: str, params: Optional[dict] = None, retries: int = 3):
        """
        Gör ett GET‑anrop mot API‑Sports (eller RapidAPI) och returnerar JSON.

        Denna hjälpfunktion bygger en fullständig URL baserad på den valda
        API‑ingången och skickar med nödvändiga headers (x‑apisports‑key
        eller x‑rapidapi‑key/host).  Den hanterar även temporära 429‑fel
        genom att göra ett kort uppehåll och försöka igen.  Om svaret är
        lyckat (HTTP 200) returneras JSON‑datan, annars None.

        :param endpoint: Endpointen ska börja med ett snedstreck, t.ex. "/fixtures".
        :param params: URL‑parametrar som dict.
        :param retries: Antal återförsök vid fel eller 429‑svar.
        :return: Parsat JSON‑objekt eller None.
        """
        if not requests or not API_KEY:
            return None
        # Bestäm bas‑URL och headers beroende på vald ingång
        base = API_BASE_URL
        headers: Dict[str, str] = {}
        if API_USE_RAPIDAPI:
            # När RapidAPI används måste värd‑headern sättas och versionen läggas till i bas‑URL
            base = f"https://{API_HOST}/v3"
            headers["x-rapidapi-host"] = API_HOST
            headers["x-rapidapi-key"] = API_KEY
        else:
            # Direkt mot API‑Sports
            headers["x-apisports-key"] = API_KEY
        url = f"{base}{endpoint}"
        for attempt in range(max(1, retries)):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
                # 429 = Too Many Requests.  Vänta längre och försök igen.
                if resp.status_code == 429:
                    # Exponentiell backoff: öka väntetid beroende på vilket försök vi är på
                    wait_sec = self.rate_limit * (attempt + 1) + 1.0
                    time.sleep(wait_sec)
                    continue
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                # Ignorera temporära nätverkfel och försök igen efter rate limit
                pass
            # Respektera rate‑limit mellan försök för att undvika att spamma API:et
            time.sleep(self.rate_limit)
        return None

    def _get_fixture_api(self, fixture_id: int):
        """Hämtar detaljer om ett enskilt fixture (match) från API‑Sports."""
        data = self._get_api("/fixtures", {"id": fixture_id})
        try:
            resp = data.get("response") if data else None
            return resp[0] if resp else None
        except Exception:
            return None

    def _get_fixture_statistics_api(self, fixture_id: int):
        """Hämtar matchstatistik för ett fixture från API‑Sports."""
        data = self._get_api("/fixtures/statistics", {"fixture": fixture_id})
        try:
            resp = data.get("response") if data else None
            return resp if resp else None
        except Exception:
            return None

    def _parse_fixture_stats(self, fixture_info: dict, stats: List[dict]):
        """
        Konverterar API‑Sports fixtures + statistik till en rad med modellens
        kolumnnamn (HG, AG, HST, AST, HS, AS, HC, AC, HF, AF, HY, AY).
        Om röd kort registreras läggs det till i HY/AY som ytterligare en gul.
        """
        try:
            if not fixture_info or not stats:
                return None
            home_id = fixture_info["teams"]["home"]["id"]
            away_id = fixture_info["teams"]["away"]["id"]
            date = pd.to_datetime(fixture_info["fixture"]["date"]).date()
            home = fixture_info["teams"]["home"]["name"]
            away = fixture_info["teams"]["away"]["name"]
            row = {"Date": date, "Home": home, "Away": away,
                   "HG": fixture_info["goals"]["home"], "AG": fixture_info["goals"]["away"]}
            # Initiera statistik med NaN
            for c in ["HST","AST","HS","AS","HC","AC","HF","AF","HY","AY"]:
                row[c] = np.nan
            for entry in stats:
                team_id = entry.get("team", {}).get("id")
                side = "home" if team_id == home_id else ("away" if team_id == away_id else None)
                if not side:
                    continue
                for item in entry.get("statistics", []):
                    typ = item.get("type")
                    val = item.get("value")
                    if val is None:
                        continue
                    if typ == "Shots on Goal":
                        row["HST" if side=="home" else "AST"] = val
                    elif typ == "Total Shots":
                        row["HS" if side=="home" else "AS"] = val
                    elif typ == "Corner Kicks":
                        row["HC" if side=="home" else "AC"] = val
                    elif typ == "Fouls":
                        row["HF" if side=="home" else "AF"] = val
                    elif typ == "Yellow Cards":
                        col = "HY" if side=="home" else "AY"
                        row[col] = val
                    elif typ == "Red Cards":
                        # Rött kort ≈ ett extra gult; lägg till i HY/AY om redan finns
                        col = "HY" if side=="home" else "AY"
                        prev = row.get(col)
                        row[col] = (prev if pd.notna(prev) else 0) + val
            return row
        except Exception:
            return None

    def fetch_season_api(self, league_name: str, season: int) -> pd.DataFrame:
        """
        Hämtar matchdata för en liga och säsong via API‑Sports och returnerar
        en DataFrame med dina basala kolumner (Datum, Home, Away, HG/AG,
        HST/AST, HS/AS, HC/AC, HF/AF, HY/AY).  I stället för att göra
        ett enskilt anrop per fixture använder denna metod batch‑hämtning
        där upp till 20 fixture‑ID hämtas samtidigt via `fixtures?ids=...`.
        Status filtreras så att endast färdigspelade matcher (FT, AET, PEN)
        inkluderas, och tidszonen sätts till Europe/Stockholm för att
        säkerställa korrekta datum.  Funktionen hanterar även rate‑limits
        genom att pausa mellan anrop samt göra backoff vid 429‑svar.

        :param league_name: Namnet på ligan enligt API_LEAGUE_IDS.
        :param season: Säsongens år (t.ex. 2023 för 2023/24).
        :return: DataFrame med matchrader.  Tom om inga data hämtades.
        """
        league_id = API_LEAGUE_IDS.get(league_name)
        if not league_id or not API_KEY:
            return pd.DataFrame()
        # Hämta en lista över fixture‑ID för denna liga och säsong.  Vi filtrerar
        # på status så att endast färdigspelade matcher tas med (FT, AET, PEN).
        status_filter = "FT-AET-PEN"
        params = {
            "league": league_id,
            "season": season,
            "status": status_filter,
            "timezone": "Europe/Stockholm",
            # Inga övriga filter; standardhämtar alla matcher för säsongen.
        }
        data = self._get_api("/fixtures", params)
        if not data or "response" not in data:
            return pd.DataFrame()
        # Extrahera alla fixture‑ID
        fixture_ids: List[int] = []
        fixture_info_map: Dict[int, dict] = {}
        for fixt in data["response"]:
            try:
                fid = fixt.get("fixture", {}).get("id")
                if fid is None:
                    continue
                fixture_ids.append(int(fid))
                # Spara fixture‑info för senare (goals, teams, datum etc.)
                fixture_info_map[int(fid)] = fixt
            except Exception:
                continue
        if not fixture_ids:
            return pd.DataFrame()
        # Dela upp ID:n i block om max 20 (API‑Sports tillåter upp till 20
        # fixtures per anrop).  Detta reducerar antalet anrop och är
        # rate‑limit‑vänligt.
        def _chunks(it, n: int):
            it = list(it)
            for i in range(0, len(it), n):
                yield it[i:i+n]
        rows: List[dict] = []
        for grp in _chunks(fixture_ids, 20):
            ids_param = "-".join(str(i) for i in grp)
            params2 = {
                "ids": ids_param,
                "timezone": "Europe/Stockholm",
            }
            batch_data = self._get_api("/fixtures", params2)
            # Om vi inte får något svar fortsätt; backoff hanteras i _get_api
            if not batch_data or "response" not in batch_data:
                continue
            for itm in batch_data["response"]:
                try:
                    fid = itm.get("fixture", {}).get("id")
                    if fid is None:
                        continue
                    fixture_info = fixture_info_map.get(int(fid), itm)
                    stats = itm.get("statistics")
                    # Om statistik inte medföljer (kan ske om coverage saknas)
                    # försök hämta separat med fallback
                    if not stats:
                        stats = self._get_fixture_statistics_api(int(fid))
                    if not stats:
                        continue
                    row = self._parse_fixture_stats(fixture_info, stats)
                    if row:
                        row["League"] = league_name
                        row["Season"] = season
                        rows.append(row)
                except Exception:
                    continue
            # Pausa en kort stund mellan batchar för att undvika rate‑limit
            time.sleep(self.rate_limit)
        return pd.DataFrame(rows)
    def _fd(self, league, season):
        code = FD_CODES.get(league);
        if not code: return None
        yy = f"{str(season)[-2:]}{str(season+1)[-2:]}"
        url = f"https://www.football-data.co.uk/mmz4281/{yy}/{code}.csv"
        r = self._get(url, retries=3)
        if r is None: return None
        try: return pd.read_csv(io.StringIO(r.text), encoding="ISO-8859-1")
        except Exception: return None
    def _parse_understat_matches(self, html):
        try:
            m = re.search(r"matchesData\s*=\s*JSON\.parse\('(.+)'\)", html, re.DOTALL)
            if not m: return None
            raw = m.group(1); decoded = bytes(raw, "utf-8").decode("unicode_escape")
            data = json.loads(decoded); df = pd.DataFrame(data)
            def gx(side): return lambda d: float(d.get(side, 0.0)) if isinstance(d, dict) else np.nan
            def gg(side): return lambda d: int(d.get(side, 0)) if isinstance(d, dict) else np.nan
            df["xG_h"] = df["xG"].apply(gx("h")); df["xG_a"] = df["xG"].apply(gx("a"))
            df["goals_h"] = df["goals"].apply(gg("h")); df["goals_a"] = df["goals"].apply(gg("a"))
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df["HomeTeam_us"] = df["h"].apply(lambda o: o.get("title") if isinstance(o, dict) else None)
            df["AwayTeam_us"] = df["a"].apply(lambda o: o.get("title") if isinstance(o, dict) else None)
            keep = ["datetime","HomeTeam_us","AwayTeam_us","xG_h","xG_a","goals_h","goals_a"]
            return df[keep].copy()
        except Exception: return None
    def _understat(self, league, season):
        key = None
        for k, v in LEAGUE_MAP.items():
            if v.lower()==league.lower() or k.lower()==league.lower():
                key = k if len(k)>2 else v
                if key.lower()=="laliga": key = "La%20Liga"
                break
        if not key: key = league.replace(" ", "%20")
        url = f"https://understat.com/league/{key}/{season}"
        r = self._get(url, retries=3)
        if r is None: return None
        html = r.text
        return self._parse_understat_matches(html)
    def _merge_us_fd(self, us, fd):
        if us is None and fd is None: return pd.DataFrame()
        if us is None:
            df = fd.copy()
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
            df["Home"] = df.get("HomeTeam", df.get("Home")).astype(str)
            df["Away"] = df.get("AwayTeam", df.get("Away")).astype(str)
            return df
        if fd is None:
            df = us.copy()
            df["Date"] = pd.to_datetime(df["datetime"].dt.date)
            df["Home"] = df["HomeTeam_us"].astype(str); df["Away"] = df["AwayTeam_us"].astype(str)
            return df
        us = us.copy(); fd = fd.copy()
        us["Date"] = pd.to_datetime(us["datetime"].dt.date)
        us["Home"] = us["HomeTeam_us"].astype(str); us["Away"] = us["AwayTeam_us"].astype(str)
        us["_hn"] = us["Home"].map(fuzzy); us["_an"] = us["Away"].map(fuzzy)
        fd["Date"] = pd.to_datetime(fd["Date"], dayfirst=True, errors="coerce")
        fd["Home"] = fd.get("HomeTeam", fd.get("Home")).astype(str)
        fd["Away"] = fd.get("AwayTeam", fd.get("Away")).astype(str)
        fd["_hn"] = fd["Home"].map(fuzzy); fd["_an"] = fd["Away"].map(fuzzy)
        merged = pd.merge(us, fd, on=["Date","_hn","_an"], how="left", suffixes=("","_fd"))
        # Fuzzy fallback per datum
        miss = merged["Home"].isna() | merged["Away"].isna()
        if miss.any():
            for i, r in merged[miss].iterrows():
                date = r["Date"]; pool = fd[fd["Date"]==date]
                if pool.empty: continue
                h = r["Home"]; a = r["Away"]
                from difflib import get_close_matches
                h_cand = get_close_matches(h, pool["Home"].astype(str).tolist(), n=1, cutoff=0.7)
                a_cand = get_close_matches(a, pool["Away"].astype(str).tolist(), n=1, cutoff=0.7)
                if h_cand and a_cand:
                    row = pool[(pool["Home"]==h_cand[0]) & (pool["Away"]==a_cand[0])]
                    if not row.empty:
                        for col in ["Home","Away","B365H","B365D","B365A","AvgH","AvgD","AvgA","Open_B365H","Open_B365D","Open_B365A"]:
                            if col in row.columns: merged.at[i, col] = row.iloc[0][col]
        return merged
    def fetch_season(self, league, season):
        cp = self._cache_path(league, season, "merged")
        ttl = TTL_DAYS_RECENT if season >= TEST_YEAR-1 else TTL_DAYS_STATIC
        if os.path.exists(cp) and ((time.time()-os.path.getmtime(cp))/86400.0) <= ttl:
            try: return pd.read_parquet(cp)
            except Exception: pass
        us = self._understat(league, season); self._sleep()
        fd = self._fd(league, season)
        merged = self._merge_us_fd(us, fd)
        # Försök hämta statistik via API‑Sports och fyll i saknade värden.
        api_df = None
        if API_KEY:
            try:
                api_df = self.fetch_season_api(league, season)
            except Exception:
                api_df = None
        if api_df is not None and not api_df.empty:
            # Slå samman på datum, hemmalag och bortalag.  Behåll befintliga värden
            # och använd API‑värden där underliggande kolumner saknas.
            cols = ["Date","Home","Away","HST","AST","HS","AS","HC","AC","HF","AF","HY","AY"]
            try:
                merged = merged.merge(api_df[cols], on=["Date","Home","Away"], how="left", suffixes=("","_api"))
                for c in ["HST","AST","HS","AS","HC","AC","HF","AF","HY","AY"]:
                    if f"{c}_api" in merged.columns:
                        merged[c] = merged[c].fillna(merged[f"{c}_api"])
                        merged.drop(columns=[f"{c}_api"], inplace=True)
            except Exception:
                pass
        # xG-proxy från odds om saknas
        if merged is not None and not merged.empty:
            if ("xG_h" not in merged.columns or merged["xG_h"].isna().all()) and {"B365H","B365D","B365A"}.issubset(merged.columns):
                rows=[]
                for _,r in merged.iterrows():
                    odds=[r.get("B365H"), r.get("B365D"), r.get("B365A")]
                    Pm = overround_correction(odds)
                    mh_best, ma_best, err_best = 1.4, 1.2, 1e9
                    for mh in np.arange(0.2,3.8,0.06):
                        for ma in np.arange(0.2,3.8,0.06):
                            M = joint_nb(mh, ma, 0.15)
                            p = one_x_two_from_joint(M)
                            err = float(np.sum((p - Pm)**2))
                            if err < err_best:
                                mh_best, ma_best, err_best = mh, ma, err
                    rows.append((mh_best, ma_best))
                if rows:
                    merged["xG_h"] = [x[0] for x in rows]
                    merged["xG_a"] = [x[1] for x in rows]
        if merged is not None and not merged.empty:
            merged["League"]=league; merged["Season"]=season
            try: merged.to_parquet(cp)
            except Exception: pass
        return merged

# =================== Feature Engineering ===================

def prep_base(df):
    if df is None or df.empty: return pd.DataFrame()
    d = df.copy()
    ren = {
        "HomeTeam":"Home","AwayTeam":"Away","goals_h":"HG","goals_a":"AG",
        "xG_h":"xGH","xG_a":"xGA","B365H":"O_H","B365D":"O_X","B365A":"O_A",
        "AvgH":"O_H","AvgD":"O_X","AvgA":"O_A"
    }
    for a,b in ren.items():
        if a in d.columns and b not in d.columns:
            d.rename(columns={a:b}, inplace=True)
    d["Date"] = pd.to_datetime(d.get("Date", d.get("datetime")), errors="coerce")
    d = d.dropna(subset=["Date"])
    d["Home"] = d.get("Home", d.get("HomeTeam", "")).astype(str)
    d["Away"] = d.get("Away", d.get("AwayTeam", "")).astype(str)

    # Targets
    if {"HG","AG"}.issubset(d.columns):
        d["Target"] = np.select([d["HG"]>d["AG"], d["HG"]==d["AG"]],[0,1], default=2)
        d["BTTS_T"]  = ((d["HG"]>0) & (d["AG"]>0)).astype(int)
        d["U25_T"]   = ((d["HG"]+d["AG"])<=2).astype(int)

    # Odds → implied
    for c in ["O_H","O_X","O_A"]:
        if c not in d.columns: d[c]=np.nan
    d[["O_H","O_X","O_A"]] = d[["O_H","O_X","O_A"]].apply(pd.to_numeric, errors="coerce")
    mask = d[["O_H","O_X","O_A"]].notnull().all(axis=1)
    if mask.any():
        arr = d.loc[mask,["O_H","O_X","O_A"]].values
        imps = np.vstack([overround_correction(x) for x in arr])
        d.loc[mask,"Imp_H"]=imps[:,0]; d.loc[mask,"Imp_X"]=imps[:,1]; d.loc[mask,"Imp_A"]=imps[:,2]
    else:
        d["Imp_H"]=d["Imp_X"]=d["Imp_A"]=np.nan

    d = d.sort_values("Date")

    # Rest
    for side in ["Home","Away"]:
        d[f"{side}_Rest"] = d.groupby(side)["Date"].diff().dt.days.clip(lower=1, upper=14)
    d["Home_Rest"]=d["Home_Rest"].fillna(7.0); d["Away_Rest"]=d["Away_Rest"].fillna(7.0)

    # FD extras (shots/corners/fouls/cards) – skapa tomma om saknas
    for c in ["HST","AST","HS","AS","HC","AC","HF","AF","HY","AY"]:
        if c not in d.columns: d[c]=np.nan
    for c in ["HST","AST","HS","AS","HC","AC","HF","AF","HY","AY"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    return d.reset_index(drop=True)

def _roll_mean(series, w, minp=2): return series.rolling(w, min_periods=minp).mean()

def add_venue_form(d):
    if d is None or d.empty: return d
    df=d.copy().sort_values("Date")
    if not {"xGH","xGA"}.issubset(df.columns):
        df["xGH"]=df.get("HG", np.nan); df["xGA"]=df.get("AG", np.nan)

    # xG-baserade rullor
    for w in [6,10]:
        df[f"Home_xGF_home_r{w}"]=df.groupby("Home")["xGH"].transform(lambda s: _roll_mean(s, w)).shift(1)
        df[f"Home_xGA_home_r{w}"]=df.groupby("Home")["xGA"].transform(lambda s: _roll_mean(s, w)).shift(1)
        df[f"Home_xGD_home_r{w}"]=df[f"Home_xGF_home_r{w}"]-df[f"Home_xGA_home_r{w}"]
        df[f"Away_xGF_away_r{w}"]=df.groupby("Away")["xGA"].transform(lambda s: _roll_mean(s, w)).shift(1)
        df[f"Away_xGA_away_r{w}"]=df.groupby("Away")["xGH"].transform(lambda s: _roll_mean(s, w)).shift(1)
        df[f"Away_xGD_away_r{w}"]=df[f"Away_xGF_away_r{w}"]-df[f"Away_xGA_away_r{w}"]

    # Defensiv instabilitet (ewm)
    span=5
    df["Home_EWM_DefInst"]=df.groupby("Home")["xGA"].transform(lambda s: s.ewm(span=span, min_periods=3).std()).shift(1).fillna(0.0)
    df["Away_EWM_DefInst"]=df.groupby("Away")["xGH"].transform(lambda s: s.ewm(span=span, min_periods=3).std()).shift(1).fillna(0.0)

    # xG-residual std
    df["Home_xG_res"] = (df.get("HG",0)-df.get("xGH",0)).astype(float)
    df["Away_xG_res"] = (df.get("AG",0)-df.get("xGA",0)).astype(float)
    for w in [5,8]:
        df[f"Home_xG_res_rstd{w}"]=df.groupby("Home")["Home_xG_res"].transform(lambda s: s.rolling(w, min_periods=3).std()).shift(1).fillna(0.0)
        df[f"Away_xG_res_rstd{w}"]=df.groupby("Away")["Away_xG_res"].transform(lambda s: s.rolling(w, min_periods=3).std()).shift(1).fillna(0.0)

    # GK-residual (SoT against vs GA), rullande 6
    k = 0.30
    df["SoT_against_home"] = df.groupby("Home")["AST"].transform(lambda s: s.rolling(6, min_periods=3).mean()).shift(1)
    df["SoT_against_away"] = df.groupby("Away")["HST"].transform(lambda s: s.rolling(6, min_periods=3).mean()).shift(1)
    df["GA_home_r"] = df.groupby("Home")["AG"].transform(lambda s: s.rolling(6, min_periods=3).mean()).shift(1)
    df["GA_away_r"] = df.groupby("Away")["HG"].transform(lambda s: s.rolling(6, min_periods=3).mean()).shift(1)
    df["GK_res_home"] = (df["GA_home_r"] - k*df["SoT_against_home"]).fillna(0.0)
    df["GK_res_away"] = (df["GA_away_r"] - k*df["SoT_against_away"]).fillna(0.0)

    # ---- NYTT: set-piece & stil ----
    # Hörnor (HC/AC), skott (HS/AS), fouls (HF/AF), kort (HY/AY)
    def rmean(team_col, stat_col, w=6):
        return df.groupby(team_col)[stat_col].transform(lambda s: s.rolling(w, min_periods=3).mean()).shift(1)

    # For (Home lagets egna siffror hemma; Away lagets egna siffror borta)
    df["Home_Corners_For_r6"] = rmean("Home","HC")
    df["Home_Corners_Ag_r6"]  = rmean("Home","AC")
    df["Away_Corners_For_r6"] = rmean("Away","AC")
    df["Away_Corners_Ag_r6"]  = rmean("Away","HC")

    df["Home_Shots_For_r6"] = rmean("Home","HS")
    df["Home_Shots_Ag_r6"]  = rmean("Home","AS")
    df["Away_Shots_For_r6"] = rmean("Away","AS")
    df["Away_Shots_Ag_r6"]  = rmean("Away","HS")

    df["Home_Fouls_r6"] = rmean("Home","HF")
    df["Away_Fouls_r6"] = rmean("Away","AF")
    df["Home_Cards_r6"] = rmean("Home","HY")
    df["Away_Cards_r6"] = rmean("Away","AY")

    # Pace proxys (skott per match)
    df["Home_Pace_r6"] = (df["Home_Shots_For_r6"].fillna(0)+df["Home_Shots_Ag_r6"].fillna(0))
    df["Away_Pace_r6"] = (df["Away_Shots_For_r6"].fillna(0)+df["Away_Shots_Ag_r6"].fillna(0))
    df["Match_Pace_r6"] = (df["Home_Pace_r6"].fillna(0)+df["Away_Pace_r6"].fillna(0))/2.0

    # Set-piece mismatch index (corners for - against, normaliserad)
    def norm(x):
        try:
            return float(x)
        except Exception:
            return 0.0
    h_sp = (df["Home_Corners_For_r6"].fillna(0.0) - df["Home_Corners_Ag_r6"].fillna(0.0))
    a_sp = (df["Away_Corners_For_r6"].fillna(0.0) - df["Away_Corners_Ag_r6"].fillna(0.0))
    # positiv SP_Mismatch => gynnar hemmalagets set-piece-profil
    df["SP_Mismatch"] = (h_sp - a_sp).apply(norm)

    # Aggression proxys (fouls + cards)
    df["Home_Agg_r6"] = (df["Home_Fouls_r6"].fillna(0.0) + 0.5*df["Home_Cards_r6"].fillna(0.0))
    df["Away_Agg_r6"] = (df["Away_Fouls_r6"].fillna(0.0) + 0.5*df["Away_Cards_r6"].fillna(0.0))

    # Fyll nans
    fillcols = [
        "Home_Corners_For_r6","Home_Corners_Ag_r6","Away_Corners_For_r6","Away_Corners_Ag_r6",
        "Home_Shots_For_r6","Home_Shots_Ag_r6","Away_Shots_For_r6","Away_Shots_Ag_r6",
        "Home_Fouls_r6","Away_Fouls_r6","Home_Cards_r6","Away_Cards_r6",
        "Home_Pace_r6","Away_Pace_r6","Match_Pace_r6","SP_Mismatch","Home_Agg_r6","Away_Agg_r6"
    ]
    for c in fillcols: df[c]=df[c].fillna(0.0)

    return df.reset_index(drop=True)

def add_market_drift(d):
    if d is None or d.empty: return d
    df=d.copy()
    for c in ["Open_B365H","Open_B365D","Open_B365A","B365H","B365D","B365A"]:
        if c not in df.columns: df[c]=np.nan
    for side in ["H","X","A"]:
        df[f"Open_Imp_{side}"]=1/df[f"Open_B365{side}"]
        df[f"Close_Imp_{side}"]=1/df[f"B365{side}"]
    df["CLV_H"]=df["Close_Imp_H"]-df["Open_Imp_H"]
    df["CLV_X"]=df["Close_Imp_X"]-df["Open_Imp_X"]
    df["CLV_A"]=df["Close_Imp_A"]-df["Open_Imp_A"]
    df["Drift_Asym_HA"] = (df["Close_Imp_H"]-df["Open_Imp_H"]) - (df["Close_Imp_A"]-df["Open_Imp_A"])
    df["Drift_Asym_HX"] = (df["Close_Imp_H"]-df["Open_Imp_H"]) - (df["Close_Imp_X"]-df["Open_Imp_X"])
    df["Drift_Asym_AX"] = (df["Close_Imp_A"]-df["Open_Imp_A"]) - (df["Close_Imp_X"]-df["Open_Imp_X"])
    df["Overround_Open"]  = (1/df["Open_B365H"] + 1/df["Open_B365D"] + 1/df["Open_B365A"]) - 1.0
    df["Overround_Close"] = (1/df["B365H"] + 1/df["B365D"] + 1/df["B365A"]) - 1.0
    df["Overround_Diff"]  = df["Overround_Close"] - df["Overround_Open"]
    df["DriftBps"] = (df["Close_Imp_H"].fillna(0)-df["Open_Imp_H"].fillna(0))*10000.0
    df["LateDriftBps"]=df["DriftBps"].fillna(0.0)
    return df

def rest_asym_tanh(h_rest, a_rest):
    try: return math.tanh((float(h_rest)-float(a_rest))/3.0)
    except Exception: return 0.0

# Placeholder (ingen extern nyhetskälla här)
def fetch_key_absences_score(row): return 0.0, "off"

# =================== GoalModel (NB) med per-lag HFA ===================

class GoalModelNB:
    def __init__(self, league_dispersion: Dict[str,float], max_goals:int=MAX_GOALS):
        self.k = league_dispersion; self.max_goals=max_goals
        self.team_att={}; self.team_def={}; self.home_adv={}; self.team_home_adv={}
        self.rho_dc={}
    def fit(self, df_league, league_name):
        teams = pd.unique(df_league[["Home","Away"]].values.ravel("K"))
        att={}; deff={}
        for t in teams:
            h = (df_league["Home"]==t); a = (df_league["Away"]==t)
            xgf=(df_league.loc[h,"xGH"].mean()+df_league.loc[a,"xGA"].mean())/2.0
            xga=(df_league.loc[h,"xGA"].mean()+df_league.loc[a,"xGH"].mean())/2.0
            xgf=1.25 if (not np.isfinite(xgf)) else max(0.12,float(xgf))
            xga=1.10 if (not np.isfinite(xga)) else max(0.12,float(xga))
            att[t]=math.log(xgf+1e-3); deff[t]=-math.log(xga+1e-3)
        team_hfa={}
        for t in teams:
            h = (df_league["Home"]==t); a = (df_league["Away"]==t)
            mu_h = df_league.loc[h,"xGH"].mean(); mu_a = df_league.loc[a,"xGA"].mean()
            team_hfa[t] = clamp((mu_h - mu_a), -0.25, 0.45) if (np.isfinite(mu_h) and np.isfinite(mu_a)) else 0.10
        self.team_att[league_name]=att; self.team_def[league_name]=deff
        self.team_home_adv[league_name]=team_hfa
        self.home_adv[league_name]=np.mean(list(team_hfa.values())) if team_hfa else 0.10
        self.rho_dc[league_name]=0.05
    def team_mu(self, league, home, away, rest_h, rest_a, news_shock=0.0):
        att=self.team_att.get(league,{}); deff=self.team_def.get(league,{})
        ha=att.get(home,0.0); hd=deff.get(home,0.0); aa=att.get(away,0.0); ad=deff.get(away,0.0)
        base_hfa = self.home_adv.get(league,0.0); team_hfa = self.team_home_adv.get(league,{}).get(home,0.0)
        rh=math.log(max(rest_h,1)/7.0+1); ra=math.log(max(rest_a,1)/7.0+1)
        mu_h=math.exp(ha+ad+base_hfa+0.5*team_hfa)*rh; mu_a=math.exp(aa+hd)*ra
        k=min(max(news_shock, -0.5), 0.5)*0.01
        mu_h=max(0.05, mu_h*(1.0 + k)); mu_a=max(0.05, mu_a*(1.0 - k))
        return float(mu_h), float(mu_a)
    def joint(self, league, home, away, rest_h, rest_a, news_shock=0.0):
        mu_h, mu_a = self.team_mu(league, home, away, rest_h, rest_a, news_shock=news_shock)
        rdisp=max(self.k.get(league,0.12),0.05)
        return joint_nb(mu_h, mu_a, rdisp, H=self.max_goals)

# =================== Classifier 1X2 ===================

class Classifier1X2:
    def __init__(self):
        self.model=None; self.cols=[]; self.class_weight=None
    def _get(self):
        if _HAVE_LGB:
            return lgb.LGBMClassifier(
                n_estimators=900, learning_rate=0.020,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                class_weight=self.class_weight
            )
        return GradientBoostingClassifier(random_state=42)
    def fit(self, X, y):
        self.cols=list(X.columns)
        classes, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        cw = {int(c): float(total/(3*cnt)) for c,cnt in zip(classes, counts)}
        self.class_weight = cw if _HAVE_LGB else None
        self.model=self._get()
        if _HAVE_LGB: self.model.fit(X.values, y)
        else:
            w = np.array([cw.get(int(t),1.0) for t in y], float)
            self.model.fit(X.values, y, sample_weight=w)
    def predict_proba(self, X):
        if self.model is None: return np.tile(np.array([1/3,1/3,1/3]), (len(X),1))
        X2=X.reindex(columns=self.cols, fill_value=0.0)
        p=self.model.predict_proba(X2.values)
        return p if p.shape[1]==3 else np.tile(np.array([1/3,1/3,1/3]), (len(X2),1))

# =================== Upset Models (MoE) ===================

def fav_bucket(x: float) -> str:
    if x < 0.60: return "50-60"
    if x < 0.70: return "60-70"
    if x < 0.80: return "70-80"
    return "80-100"

def month_bucket(d: pd.Timestamp) -> str:
    try:
        m = int(pd.to_datetime(d).month)
        if m in (8,9,10): return "autumn"
        if m in (11,12,1): return "winter"
        if m in (2,3,4): return "spring"
        return "summer"
    except Exception: return "na"

class MoE_Upset:
    def __init__(self, name="upset", alpha=1.2, max_bonus=3.0):
        self.name=name; self.experts={}; self.cols={}; self.alpha=alpha; self.max_bonus=max_bonus
        self.feature_order=[]
    def _lgb(self, n_feats: int, monotonic_idx: Dict[str,int]):
        if not _HAVE_LGB: return None
        mono=[monotonic_idx.get(c,0) for c in self.feature_order]
        return lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.035, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            monotone_constraints=mono
        )
    def _gbm(self): return GradientBoostingClassifier(n_estimators=600, max_depth=3, learning_rate=0.035, random_state=42)
    def _key(self, bk: str, is_away_fav: int) -> str: return f"{bk}|{int(is_away_fav)}"
    def fit(self, X: pd.DataFrame, y: np.ndarray, fav_imps: np.ndarray, fav_is_away: np.ndarray, leagues: np.ndarray):
        y = np.asarray(y, int); fav_imps = np.clip(np.asarray(fav_imps, float), 0.0, 1.0)
        n1 = (y == 1).sum(); n0 = (y == 0).sum()
        base_w1 = ((n0 + n1) / (2 * max(1, n1))); base_w0 = ((n0 + n1) / (2 * max(1, n0)))
        bonus = 1.0 + np.minimum(self.max_bonus - 1.0, np.power(np.maximum(0.0, fav_imps - 0.50) / 0.50, self.alpha))
        sw = np.array([(base_w1 * bonus[i]) if y[i] == 1 else base_w0 for i in range(len(y))], float)
        mono_idx = {
            "mkt_vs_model_gap": +1, "gap_top2_mkt": +1, "fav_imp": +1, "overround": +1, "rest_asym_tanh": +1,
            # nya features: milda +
            "SP_Mismatch": +1, "Home_Pace_r6": +1, "Away_Pace_r6": +1
        }
        self.feature_order = list(X.columns)
        bks = np.array([fav_bucket(fv) for fv in fav_imps])
        for level in sorted(set((bk, int(a)) for bk,a in zip(bks, fav_is_away))):
            bk, away = level; key = self._key(bk, away)
            mask = (bks==bk) & (fav_is_away==away)
            if mask.sum() < 100: continue
            Xsub = X.loc[mask]; ysub = y[mask]; wsub = sw[mask]
            if _HAVE_LGB:
                m = self._lgb(Xsub.shape[1], mono_idx); m.fit(Xsub.values, ysub, sample_weight=wsub)
            else:
                m = self._gbm(); m.fit(Xsub.values, ysub, sample_weight=wsub)
            self.experts[key]=m; self.cols[key]=list(Xsub.columns)
        if "global" not in self.experts:
            if _HAVE_LGB:
                m = self._lgb(X.shape[1], mono_idx); m.fit(X.values, y, sample_weight=sw)
            else:
                m = self._gbm(); m.fit(X.values, y, sample_weight=sw)
            self.experts["global"]=m; self.cols["global"]=list(X.columns)
    def predict(self, X_row: pd.DataFrame, fav_imp: float, is_away_fav: int) -> float:
        bk = fav_bucket(fav_imp); key = self._key(bk, is_away_fav)
        if key in self.experts: m = self.experts[key]; cols = self.cols[key]
        else: m = self.experts["global"]; cols = self.cols["global"]
        x = X_row.reindex(columns=cols, fill_value=0.0)
        try:
            return float(m.predict_proba(x.values)[:,1][0])
        except Exception:
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(x.values);
                if p.shape[1]==2: return float(p[:,1][0])
            return 0.0

# =================== Calibrators ===================

class DirichletCalibrator:
    def __init__(self): self.models: Dict[str, LogisticRegression] = {}
    def fit(self, leagues, P, y):
        df = pd.DataFrame(P, columns=["p1","px","p2"]); df["y"]=y; df["lg"]=leagues
        for lg, part in df.groupby("lg"):
            X = np.log(np.clip(part[["p1","px","p2"]].values, 1e-12, 1)).astype(float)
            Y = part["y"].astype(int).values
            if len(np.unique(Y))<2: continue
            lr = LogisticRegression(max_iter=2000, multi_class="multinomial"); lr.fit(X, Y)
            self.models[lg]=lr
    def transform(self, leagues, P):
        out = np.zeros_like(P, float)
        for i, lg in enumerate(leagues):
            lr = self.models.get(lg); row = np.log(np.clip(P[i], 1e-12, 1)).reshape(1,-1)
            out[i,:]=normalize_probs(lr.predict_proba(row)[0]) if lr is not None else normalize_probs(P[i])
        return out

class ProbCalibratorND:
    def __init__(self): self.iso = {}; self.lr  = {}
    def _key(self, lg, bk, away, mon): return f"{lg}|{bk}|{away}|{mon}"
    def fit(self, leagues, p_raw, target, fav_imps, fav_is_away, dates):
        df = pd.DataFrame({
            "lg": leagues, "p": p_raw, "y": target,
            "bk": [fav_bucket(f) for f in fav_imps],
            "aw": fav_is_away.astype(int),
            "mon": [month_bucket(d) for d in dates]
        })
        for (lg,bk,aw,mon), part in df.groupby(["lg","bk","aw","mon"]):
            key = self._key(lg,bk,aw,mon); x = part["p"].values.reshape(-1,1); y = part["y"].astype(int).values
            if np.unique(y).size<2: self.iso[key]=None; self.lr[key]=None; continue
            if np.unique(x).size<5:
                self.iso[key]=None; lr=LogisticRegression(max_iter=1000); lr.fit(x,y); self.lr[key]=lr
            else:
                ir=IsotonicRegression(out_of_bounds="clip"); ir.fit(x.ravel(), y)
                self.iso[key]=ir; lr=LogisticRegression(max_iter=1000); lr.fit(x,y); self.lr[key]=lr
    def _transform_single(self, lg,bk,aw,mon, pr):
        key=f"{lg}|{bk}|{aw}|{mon}"; ir=self.iso.get(key); lr=self.lr.get(key)
        if ir is not None: return float(ir.predict([pr])[0])
        if lr is not None: return float(lr.predict_proba([[pr]])[0,1])
        key=f"{lg}|{bk}|{aw}|*"; ir=self.iso.get(key); lr=self.lr.get(key)
        if ir is not None: return float(ir.predict([pr])[0])
        if lr is not None: return float(lr.predict_proba([[pr]])[0,1])
        key=f"{lg}|{bk}"; ir=self.iso.get(key); lr=self.lr.get(key)
        if ir is not None: return float(ir.predict([pr])[0])
        if lr is not None: return float(lr.predict_proba([[pr]])[0,1])
        key=f"{lg}"; ir=self.iso.get(key); lr=self.lr.get(key)
        if ir is not None: return float(ir.predict([pr])[0])
        if lr is not None: return float(lr.predict_proba([[pr]])[0,1])
        return pr
    def transform(self, leagues, p_raw, fav_imps, fav_is_away, dates):
        out=np.zeros_like(p_raw, float)
        for i,(lg,pr) in enumerate(zip(leagues, p_raw)):
            bk=fav_bucket(fav_imps[i]); aw=int(fav_is_away[i]); mon=month_bucket(dates[i])
            out[i]=self._transform_single(lg,bk,aw,mon,float(pr))
        return np.clip(out, 0.0, 1.0)

class ConformalLB:
    def __init__(self, level=0.85): self.level=level; self.q=0.0
    def fit(self, y_true: np.ndarray, p_hat: np.ndarray):
        resid = np.abs(np.asarray(p_hat,float) - np.asarray(y_true,int))
        self.q=float(np.quantile(resid, 1.0-self.level)) if resid.size>0 else 0.0
    def lower(self, p_hat: float) -> float: return float(clamp(p_hat - self.q, 0.0, 1.0))

# =================== Upset label builder ===================

def dynamic_fav_threshold_v2(p_market: np.ndarray, league: str, is_away_fav: bool) -> float:
    gap = top2_gap(p_market)
    thr = min(0.60, 0.52 + 0.35*gap)
    if league in {"LaLiga","Ligue 1"}: thr -= 0.02
    if is_away_fav: thr += 0.015
    return clamp(thr, 0.50, 0.62)

def build_upset_labels(y_true: int, p_market: np.ndarray, hg: Optional[int], ag: Optional[int], league: str) -> Tuple[int,int]:
    fav_idx = int(np.argmax(p_market)); fav_imp = float(p_market[fav_idx])
    is_away_fav = (fav_idx==2)
    thr = dynamic_fav_threshold_v2(p_market, league, is_away_fav)
    upset_win = 1 if (y_true in (0,1,2) and y_true != fav_idx and fav_imp >= thr) else 0
    if hg is None or ag is None:
        upset_cover = 0
    else:
        cover = int((y_true!=fav_idx) or (y_true==fav_idx and abs(int(hg)-int(ag))==1))
        upset_cover = 1 if (fav_imp >= thr and cover==1) else 0
    return upset_win, upset_cover

# =================== AH engine ===================

def ah_cover_prob(M: np.ndarray, underdog_side: str, line: float) -> float:
    H, A = M.shape; p = 0.0
    for h in range(H):
        for a in range(A):
            d = h - a
            ok = (d >= -line) if underdog_side=="home" else (d <= line)
            if ok: p += M[h,a]
    return float(p)

def synthetic_ah_odds(league: str, p_cover: float) -> Optional[float]:
    if not SYNTH_AH_PRICING["ENABLE"]: return None
    if not (0.0 < p_cover < 1.0): return None
    m = SYNTH_AH_PRICING["LEAGUE_MARGIN"].get(league, 0.06)
    o_fair = 1.0/p_cover
    o_book = max(1.01, min(o_fair*(1.0 - m), SYNTH_AH_PRICING["ODDS_CAP"]))
    return float(o_book)

def ev_from_prob(p: float, o: Optional[float]) -> Optional[float]:
    if o is None or not np.isfinite(o) or o<=1.0 or not (0<=p<=1): return None
    return float(p*(o-1.0) - (1.0-p))

def kelly_stake(p: float, o: float, frac=0.25) -> float:
    if not (np.isfinite(p) and np.isfinite(o) and o>1.0): return 0.0
    b = o-1.0; q=1.0-p; edge=b*p - q
    if edge<=0: return 0.0
    return float(frac * edge / b)

# =================== Policies ===================

def _draw_propensity(league, mu_total, pvec):
    cfg=GUARD["draw_propensity"].get(league)
    if not cfg: return pvec, None
    p1,px,p2=map(float,pvec); top=max(p1,px,p2)
    if (top<=cfg["maxprob_cap"]) and (px>=cfg["px_min"]) and (mu_total<=cfg["mu_total_max"]):
        lo,hi=cfg["boost_range"]; boost=float(np.random.uniform(lo,hi))
        px2=min(0.95, px+boost); take=px2-px
        p1=max(0.0,p1-take/2.0); p2=max(0.0,p2-take/2.0); px=px2
        return normalize_probs([p1,px,p2]), "DrawPropensity"
    return pvec, None

def policy_post_adjust(pvec, league, pick_idx, upset_risk, mu_total):
    flags=[]
    p1,px,p2=map(float,pvec)
    top=max(p1,px,p2); second=sorted([p1,px,p2], reverse=True)[1]
    topgap_pp=(top-second)*100.0
    cfg=GUARD["draw_propensity"].get(league, None)
    px_floor = max(GUARD["knife_edge_min_px"], (cfg["px_min"]+0.01) if cfg else 0.0)
    if topgap_pp<GUARD["knife_edge_gap_pp"] and px<px_floor:
        take=px_floor-px
        if p1>=p2 and p1>=px: p1=max(0.0,p1-take)
        elif p2>=p1 and p2>=px: p2=max(0.0,p2-take)
        px=px_floor; flags.append("KnifeEdge_XFloor")
    (p1,px,p2), tag = _draw_propensity(league, mu_total, [p1,px,p2])
    if tag: flags.append(tag)
    if league in RED_LEAGUES and pick_idx==2:
        mix=0.92; p2=mix*p2+(1.0-mix)*px; flags.append("RedAwayShrink_Mix")
    if pick_idx==2 and (0.45<=p2<=0.60) and (topgap_pp<=6.0):
        delta=min(0.02, p2-0.01); p2-=delta; px+=delta; flags.append("AwayEven_ShrinkToX")
    return normalize_probs([p1,px,p2]), flags

def cup_unseen_whitelist_filter(league: str, season: int, h: str, a: str, seen_by_season: Dict[Tuple[str,int], set], manual_whitelist: Dict[str,set]) -> bool:
    if season is None: return True
    seen = seen_by_season.get((league,int(season)), set())
    if (h in seen) and (a in seen): return False
    allow = manual_whitelist.get(league, set())
    if (h in allow) and (a in allow): return False
    return True

def medium_band_gate(ok_range: Tuple[float,float], p_top: float, upset_win_lb: float, upset_cover_lb: float, late_drift_bps: float, relax_delta: float=0.0) -> bool:
    if ok_range[0] <= p_top < ok_range[1]:
        g=GUARD["medium_gate"]
        if upset_win_lb < g["upset_win_lb_min"]: return True
        if upset_cover_lb+relax_delta < g["upset_cover_lb_min"]: return True
        if late_drift_bps > g["late_drift_bps_max"]: return True
    return False

def btts_gate(p_btts: float, mu_total: float, gk_home: float, gk_away: float, lowpace: bool=False) -> bool:
    g=GUARD["btts_gate"]
    if p_btts < g["p_min"]: return True
    if mu_total < g["mu_total_min"]: return True
    if g["gk_soft_block"] and (gk_home < -0.25 or gk_away < -0.25) and (not lowpace or STYLE_CFG["gk_block_extra_when_lowpace"]):
        return True
    return False

def big_fav_line_block(mu_total: float, upset_cover_lb: float) -> bool:
    g=GUARD["fav_bigline_gate"]
    return not (mu_total >= g["mu_total_min"] and upset_cover_lb >= g["upset_cover_lb_min"])

# =================== Main Model ===================

BASE_UPSET_FEATS = [
    "Imp_H","Imp_X","Imp_A","Home_Rest","Away_Rest",
    "Home_xGF_home_r6","Home_xGA_home_r6","Home_xGD_home_r6",
    "Away_xGF_away_r6","Away_xGA_away_r6","Away_xGD_away_r6",
    "Home_EWM_DefInst","Away_EWM_DefInst",
    "DriftBps","LateDriftBps",
    "CLV_H","CLV_X","CLV_A","Drift_Asym_HA","Drift_Asym_HX","Drift_Asym_AX",
    "Overround_Open","Overround_Close","Overround_Diff",
    "Home_xG_res_rstd5","Home_xG_res_rstd8","Away_xG_res_rstd5","Away_xG_res_rstd8",
    "GK_res_home","GK_res_away",
    # Nya stil/SP-features:
    "Home_Corners_For_r6","Home_Corners_Ag_r6","Away_Corners_For_r6","Away_Corners_Ag_r6",
    "Home_Shots_For_r6","Home_Shots_Ag_r6","Away_Shots_For_r6","Away_Shots_Ag_r6",
    "Home_Pace_r6","Away_Pace_r6","Match_Pace_r6","SP_Mismatch","Home_Agg_r6","Away_Agg_r6"
]

class ApexSuperModelV891:
    def __init__(self, conformal_level=0.85):
        self.data=QuantumDataManager(CACHE_ROOT, timeout=15.0, rate_limit=1.6)
        self.goal=GoalModelNB(OVERDISP, max_goals=MAX_GOALS)
        self.main=Classifier1X2()
        self.dir_cal=DirichletCalibrator()
        self.upset_win = MoE_Upset(name="upset_win")
        self.upset_cover = MoE_Upset(name="upset_cover")
        self.upset_cal_nd_win   = ProbCalibratorND()
        self.upset_cal_nd_cover = ProbCalibratorND()
        self.conf_win  = ConformalLB(level=conformal_level)
        self.conf_cover= ConformalLB(level=conformal_level)
        self.iso_pre={lg:[None,None,None] for lg in OVERDISP.keys()}
        self.btts_cal=ProbCalibratorND(); self.u25_cal=ProbCalibratorND()
        self.seen_by_season={}; self.manual_whitelist={lg:set() for lg in OVERDISP.keys()}

    def _artifacts_path(self, tag): return os.path.join(CACHE_ROOT, f"artifacts_{tag}.pkl")
    def save_artifacts(self, tag="latest"):
        state={
            "goal_att":self.goal.team_att, "goal_def":self.goal.team_def,
            "goal_home":self.goal.home_adv, "goal_rho":self.goal.rho_dc,
            "team_home_adv": self.goal.team_home_adv,
            "seen_by_season": self.seen_by_season,
            "manual_whitelist": self.manual_whitelist
        }
        with open(self._artifacts_path(tag),"wb") as f: pickle.dump(state,f)
    def load_artifacts(self, tag="latest", ttl_days=TTL_DAYS_RECENT):
        path=self._artifacts_path(tag)
        if os.path.exists(path) and ((time.time()-os.path.getmtime(path))/86400.0)<=ttl_days:
            try:
                with open(path,"rb") as f: state=pickle.load(f)
                self.goal.team_att=state.get("goal_att",{})
                self.goal.team_def=state.get("goal_def",{})
                self.goal.home_adv=state.get("goal_home",{})
                self.goal.rho_dc=state.get("goal_rho",{})
                self.goal.team_home_adv=state.get("team_home_adv",{})
                self.seen_by_season=state.get("seen_by_season",{})
                self.manual_whitelist=state.get("manual_whitelist", self.manual_whitelist)
                return True
            except Exception:
                return False
        return False

    def _build_upset_feats_row(self, r, pgen, p_market):
        fav_idx = int(np.argmax(p_market)); fav_imp = float(p_market[fav_idx])
        gap_top2_mkt = top2_gap(p_market); gap_top2_mod = top2_gap(pgen)
        ent_mkt = prob_entropy(p_market); ent_mod = prob_entropy(pgen)
        model_on_fav = float(pgen[fav_idx]); mkt_vs_model_gap = float(fav_imp - model_on_fav)
        asym_mkt = float(p_market[0] - p_market[2]); asym_mod = float(pgen[0] - pgen[2])
        ovrr = book_overround([r.get("O_H"), r.get("O_X"), r.get("O_A")])
        is_away_fav = 1.0 if fav_idx == 2 else 0.0
        mu_total = float(r.get("mu_h", 0.0) + r.get("mu_a", 0.0))
        rest_delta = float(r.get("Home_Rest", 7.0) - r.get("Away_Rest", 7.0))
        rest_tanh = rest_asym_tanh(r.get("Home_Rest",7.0), r.get("Away_Rest",7.0))
        feats = {**{k: float(r.get(k, 0.0)) for k in BASE_UPSET_FEATS},
            "fav_imp": fav_imp, "gap_top2_mkt": gap_top2_mkt, "gap_top2_mod": gap_top2_mod,
            "ent_mkt": ent_mkt, "ent_mod": ent_mod, "mkt_vs_model_gap": mkt_vs_model_gap,
            "is_away_fav": is_away_fav, "asym_mkt": asym_mkt, "asym_mod": asym_mod,
            "overround": ovrr, "pgen_x": float(pgen[1]), "mu_total": mu_total,
            "rest_delta": rest_delta, "rest_asym_tanh": rest_tanh
        }
        return feats, fav_imp, is_away_fav

    def fit(self, full_data: pd.DataFrame):
        df=full_data.copy()
        df=add_venue_form(df); df=add_market_drift(df)
        for (lg,season), part in df.groupby(["League","Season"]):
            if len(part)<30: continue
            try:
                self.goal.fit(part, lg)
                teams = set(pd.unique(part[["Home","Away"]].values.ravel("K")))
                self.seen_by_season[(lg,int(season))] = teams
            except Exception as e:
                print(f"[warn] Goal fit failed for {lg} {season}: {e}")
        upset_feat_rows=[]; fav_imps=[]; fav_is_away=[]; leagues=[]; dates=[]
        y_all=[]; btts_raw=[]; u25_raw=[]; upset_targets_win=[]; upset_targets_cover=[]
        for _,r in df.iterrows():
            lg=r["League"]; h=r["Home"]; a=r["Away"]
            rest_h=r.get("Home_Rest",7.0); rest_a=r.get("Away_Rest",7.0)
            key_abs, _ = fetch_key_absences_score(r); news_shock = float(r.get("LateDriftBps",0.0))*0.0001 * float(key_abs)
            M=self.goal.joint(lg,h,a,rest_h,rest_a, news_shock=news_shock)
            pgen=one_x_two_from_joint(M); mu_h,mu_a=self.goal.team_mu(lg,h,a,rest_h,rest_a, news_shock=news_shock)
            p_btts_raw=btts_from_joint(M); p_u25_raw=under25_from_joint(M)
            p_market = overround_correction([r.get("O_H"),r.get("O_X"),r.get("O_A")])
            upset_win, upset_cover = build_upset_labels(int(r.get("Target",-1)) if "Target" in r else -1,
                                                        p_market, r.get("HG"), r.get("AG"), lg)
            feats, fav_imp, awayfav = self._build_upset_feats_row({**r.to_dict(), **{"mu_h":mu_h,"mu_a":mu_a}}, pgen, p_market)
            upset_feat_rows.append(feats); fav_imps.append(fav_imp); fav_is_away.append(int(awayfav)); leagues.append(lg); dates.append(r.get("Date"))
            y_all.append(int(r.get("Target",-1)) if "Target" in r else -1)
            btts_raw.append(p_btts_raw); u25_raw.append(p_u25_raw)
            upset_targets_win.append(upset_win); upset_targets_cover.append(upset_cover)

        X_up = pd.DataFrame(upset_feat_rows).fillna(0.0)
        ok=(np.array(y_all)>=0)&(np.array(y_all)<=2)
        X_main = X_up.copy()
        self.main.fit(X_main[ok], np.array(y_all)[ok])
        P_main=self.main.predict_proba(X_main[ok])

        leagues_ok=np.array(leagues)[ok]; y_ok=np.array(y_all)[ok]
        # isotonic per klass → dirichlet
        self.iso_pre={lg:[None,None,None] for lg in set(leagues_ok)}
        for lg, part_idx in pd.Series(leagues_ok).reset_index().groupby(0)["index"]:
            lgname = part_idx.name; cal_list=[]
            for k in range(3):
                x=P_main[part_idx.values, k]; t=(y_ok[part_idx.values]==k).astype(int)
                if np.unique(x).size<3: cal_list.append(None)
                else:
                    ir=IsotonicRegression(out_of_bounds="clip"); ir.fit(x,t); cal_list.append(ir)
            self.iso_pre[lgname]=cal_list
        P_iso=[]
        for i,lg in enumerate(leagues_ok):
            row=P_main[i].copy(); cal_list=self.iso_pre.get(lg,[None,None,None]); adj=[]
            for k in range(3):
                ir=cal_list[k]; adj.append(row[k] if ir is None else float(ir.predict([row[k]])[0]))
            P_iso.append(normalize_probs(np.array(adj,float)))
        P_iso=np.vstack(P_iso)
        self.dir_cal.fit(leagues_ok, P_iso, y_ok)

        upset_targets_win = np.array(upset_targets_win,int); upset_targets_cover = np.array(upset_targets_cover,int)
        fav_imps_arr=np.array(fav_imps,float); fav_is_away_arr=np.array(fav_is_away,int); leagues_arr=np.array(leagues); dates_arr=np.array(dates)
        self.upset_win.fit(X_up, upset_targets_win, fav_imps_arr, fav_is_away_arr, leagues_arr)
        self.upset_cover.fit(X_up, upset_targets_cover, fav_imps_arr, fav_is_away_arr, leagues_arr)

        # OOF för conformal
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_win = np.zeros(len(X_up), float); oof_cover = np.zeros(len(X_up), float)
        for tr, va in kf.split(X_up):
            mw = MoE_Upset(name="upset_win_fold"); mc = MoE_Upset(name="upset_cover_fold")
            mw.fit(X_up.iloc[tr], upset_targets_win[tr], fav_imps_arr[tr], fav_is_away_arr[tr], leagues_arr[tr])
            mc.fit(X_up.iloc[tr], upset_targets_cover[tr], fav_imps_arr[tr], fav_is_away_arr[tr], leagues_arr[tr])
            for idx in va:
                xrow = X_up.iloc[[idx]]
                oof_win[idx]=mw.predict(xrow, fav_imps_arr[idx], fav_is_away_arr[idx])
                oof_cover[idx]=mc.predict(xrow, fav_imps_arr[idx], fav_is_away_arr[idx])
        self.upset_cal_nd_win.fit(leagues_arr, oof_win, upset_targets_win, fav_imps_arr, fav_is_away_arr, dates_arr)
        self.upset_cal_nd_cover.fit(leagues_arr, oof_cover, upset_targets_cover, fav_imps_arr, fav_is_away_arr, dates_arr)
        self.conf_win.fit(upset_targets_win, oof_win); self.conf_cover.fit(upset_targets_cover, oof_cover)

        btts_raw=np.array(btts_raw,float); u25_raw=np.array(u25_raw,float)
        self.btts_cal.fit(leagues_arr[ok], btts_raw[ok], (df.loc[ok,"BTTS_T"].astype(int).values if "BTTS_T" in df.columns else np.zeros(ok.sum(),int)),
                          fav_imps_arr[ok], fav_is_away_arr[ok], dates_arr[ok])
        self.u25_cal.fit(leagues_arr[ok], u25_raw[ok], (df.loc[ok,"U25_T"].astype(int).values if "U25_T" in df.columns else np.zeros(ok.sum(),int)),
                         fav_imps_arr[ok], fav_is_away_arr[ok], dates_arr[ok])
        self.save_artifacts(tag="latest")

    def _calibrate_main(self, lg: str, p_main: np.ndarray) -> np.ndarray:
        cal_list=self.iso_pre.get(lg,[None,None,None]); adj=[]
        for k in range(3):
            ir=cal_list[k]; adj.append(p_main[k] if ir is None else float(ir.predict([p_main[k]])[0]))
        P_iso=normalize_probs(np.array(adj,float))
        return self.dir_cal.transform([lg], np.array([P_iso]))[0]

    def predict_row(self, row: dict, season:int=None):
        lg=row.get("League"); h=row.get("Home"); a=row.get("Away")
        rest_h=row.get("Home_Rest",7.0); rest_a=row.get("Away_Rest",7.0)
        key_abs, _ = fetch_key_absences_score(row); news_shock = float(row.get("LateDriftBps",0.0))*0.0001 * float(key_abs)
        is_block = cup_unseen_whitelist_filter(lg, season or int(row.get("Season",TEST_YEAR)), h, a, self.seen_by_season, self.manual_whitelist)

        M=self.goal.joint(lg,h,a,rest_h,rest_a, news_shock=news_shock)
        pgen=one_x_two_from_joint(M)
        mu_h,mu_a=self.goal.team_mu(lg,h,a,rest_h,rest_a, news_shock=news_shock)
        mu_total=mu_h+mu_a
        p_btts_raw=btts_from_joint(M); p_u25_raw=under25_from_joint(M)
        p_market = overround_correction([row.get("O_H"),row.get("O_X"),row.get("O_A")])

        feats, fav_imp, is_away_fav = self._build_upset_feats_row({**row, **{"mu_h":mu_h,"mu_a":mu_a}}, pgen, p_market)
        Xu = pd.DataFrame([feats])
        up_win_raw   = self.upset_win.predict(Xu, fav_imp, int(is_away_fav))
        up_cover_raw = self.upset_cover.predict(Xu, fav_imp, int(is_away_fav))
        up_win_cal   = float(self.upset_cal_nd_win.transform([lg], np.array([up_win_raw]), np.array([fav_imp]), np.array([int(is_away_fav)]), np.array([row.get("Date", pd.Timestamp("2024-01-01"))]))[0])
        up_cover_cal = float(self.upset_cal_nd_cover.transform([lg], np.array([up_cover_raw]), np.array([fav_imp]), np.array([int(is_away_fav)]), np.array([row.get("Date", pd.Timestamp("2024-01-01"))]))[0])
        up_win_lb    = self.conf_win.lower(up_win_cal)
        up_cover_lb  = self.conf_cover.lower(up_cover_cal)

        meta = pd.DataFrame([{**feats, **{"pgen_1":pgen[0],"pgen_x":pgen[1],"pgen_2":pgen[2]}}])
        p_main = self.main.predict_proba(meta)[0]
        gap_pp=abs((p_main[np.argmax(p_main)]-p_market[np.argmax(p_main)])*100.0)
        if GUARD["market_blend"]["enable"] and gap_pp>=GUARD["market_blend"]["abs_gap_pp"]:
            w=GUARD["market_blend"]["blend_w"]; p_main=normalize_probs((1.0-w)*p_main + w*p_market)
        p_adj, flags1 = policy_post_adjust(p_main, lg, int(np.argmax(p_main)), up_win_cal, mu_total)
        P_cal = self._calibrate_main(lg, p_adj)

        P_btts=np.array([p_btts_raw]); P_u25=np.array([p_u25_raw])
        p_btts = float(self.btts_cal.transform([lg], P_btts, np.array([fav_imp]), np.array([int(is_away_fav)]), np.array([row.get("Date", pd.Timestamp("2024-01-01"))]))[0])
        p_u25c = float(self.u25_cal.transform([lg], P_u25, np.array([fav_imp]), np.array([int(is_away_fav)]), np.array([row.get("Date", pd.Timestamp("2024-01-01"))]))[0])

        # Bestäm preliminärt pick baserat på högsta sannolikheten
        pick_idx=int(np.argmax(P_cal))
        p_top=float(P_cal[pick_idx])
        conf = "High" if p_top>=HIGH_THRESHOLD else ("Medium" if MEDIUM_RANGE[0]<=p_top<MEDIUM_RANGE[1] else "Low")
        # -------------------
        # Extra draw‑heuristik
        # I backtesten visade modellerna en tendens att underskatta oavgjort
        # (Draw).  För att förbättra detta införs en enkel regel: om
        # skillnaden mellan hemmaseger- och bortaseger‑sannolikheten är
        # liten och drawsannolikheten ligger över ett tröskelvärde så
        # väljer modellen X.  Detta ska fånga matcher där utfallet är
        # jämnt och en oavgjord utgång är plausibel.  Konfidensnivån
        # justeras ned något vid sådana fall.
        p1, px, p2 = float(P_cal[0]), float(P_cal[1]), float(P_cal[2])
        # Skillnaden mellan de två segerchanserna (hemma vs borta)
        diff12 = abs(p1 - p2)
        # Dynamiska trösklar baserade på ligans draw‑propensity.  Vi tar
        # ligaspecifik px_min och höjer den något för att utlösa X i jämna
        # matcher.  diff‑tröskeln anger hur nära de två segerchanserna måste
        # vara.
        cfg_draw = GUARD.get("draw_propensity", {}).get(lg, {})
        draw_px_min = float(cfg_draw.get("px_min", 0.27)) + 0.01
        draw_diff_max = 0.07
        if conf != "No-bet" and px >= draw_px_min and diff12 <= draw_diff_max:
            # Välj oavgjort som pick
            pick_idx = 1
            p_top = px
            # Anpassa konfidens: medel om drawprob ≥ (px_min + 0.05), annars låg
            conf = "Medium" if px >= (draw_px_min + 0.05) else "Low"

        flags=flags1[:]
        # Medium gate med ev. relax vid stark set-piece-mismatch
        sp_mismatch = float(row.get("SP_Mismatch", feats.get("SP_Mismatch", 0.0)))
        relax = STYLE_CFG["allow_ah_relax_delta"] if abs(sp_mismatch)>=STYLE_CFG["sp_mismatch_pp"] else 0.0
        if medium_band_gate(MEDIUM_RANGE, p_top, up_win_lb, up_cover_lb, float(row.get("LateDriftBps",0.0)), relax_delta=relax):
            conf="No-bet"; flags.append("MediumBand_Gate" + ("_Relaxed" if relax>0 else ""))

        if is_block:
            conf="No-bet"; flags.append("Cup/Unseen_NoBet")

        lowpace = (float(feats.get("Match_Pace_r6",0.0)) <= STYLE_CFG["pace_low"])
        if conf!="No-bet" and p_btts>=0.60:
            if btts_gate(p_btts, mu_total, float(feats.get("GK_res_home",0.0)), float(feats.get("GK_res_away",0.0)), lowpace=lowpace):
                flags.append("BTTS_Gate_Block")

        # -------- AH policy --------
        try:
            fav_idx=int(np.argmax([row.get("Imp_H",np.nan), row.get("Imp_X",np.nan), row.get("Imp_A",np.nan)]))
        except Exception:
            fav_idx=int(np.argmax(p_market))
        underdog_side = "away" if fav_idx==0 else ("home" if fav_idx==2 else None)
        if underdog_side is None:
            underdog_side = "home" if P_cal[0]<P_cal[2] else "away"

        p05 = ah_cover_prob(M, underdog_side, 0.5)
        p15 = ah_cover_prob(M, underdog_side, 1.5)
        o05 = row.get("AH_H_+0.5" if underdog_side=="home" else "AH_A_+0.5")
        o15 = row.get("AH_H_+1.5" if underdog_side=="home" else "AH_A_+1.5")
        if (o05 is None or not np.isfinite(o05)): o05 = synthetic_ah_odds(lg, p05)
        if (o15 is None or not np.isfinite(o15)): o15 = synthetic_ah_odds(lg, p15)

        fav = float(fav_imp); ah_rec=None
        # Relax upset_cover_lb-min om SP-mismatch tydlig till underdogens fördel
        sp_relax = STYLE_CFG["allow_ah_relax_delta"] if ((underdog_side=="home" and sp_mismatch>=STYLE_CFG["sp_mismatch_pp"]) or (underdog_side=="away" and -sp_mismatch>=STYLE_CFG["sp_mismatch_pp"])) else 0.0
        cover_gate = (self.conf_cover.lower(up_cover_cal) + sp_relax)

        if fav<0.65:
            if cover_gate>=0.60 and float(P_cal[1])>=0.27:
                ah_rec = { "product":"+0.5", "side":underdog_side, "line":0.5, "p":p05, "o":o05 }
        elif fav<0.75:
            if cover_gate>=0.58:
                ev05 = ev_from_prob(p05, o05); ev15 = ev_from_prob(p15, o15)
                if float(P_cal[1])>=0.28 and up_win_lb>=0.22 and (ev05 or -1) > (ev15 or -1):
                    ah_rec = { "product":"+0.5", "side":underdog_side, "line":0.5, "p":p05, "o":o05 }
                else:
                    ah_rec = { "product":"+1.5", "side":underdog_side, "line":1.5, "p":p15, "o":o15 }
        else:
            if cover_gate>=0.55:
                ah_rec = { "product":"+1.5", "side":underdog_side, "line":1.5, "p":p15, "o":o15 }

        # Sidoråd
        if conf in ("High","Medium"):
            side_rec = {0:"Home win",1:"Draw",2:"Away win"}[pick_idx] + f" ({conf})"
        else:
            if p_btts>=0.66 and mu_total>=2.5 and not btts_gate(p_btts, mu_total, feats.get("GK_res_home",0.0), feats.get("GK_res_away",0.0), lowpace=lowpace):
                side_rec = "BTTS: Yes"
            elif p_u25c>=0.62 or mu_total<=2.35:
                side_rec = "Under 2.5"
            else:
                side_rec = "No bet"

        stake=None
        if ah_rec and ah_rec.get("o"):
            kbase = 0.35 if ah_rec["product"]=="+0.5" else 0.45
            units = kelly_stake(ah_rec["p"], ah_rec["o"], frac=STAKE_CFG["kelly_frac"]*kbase/0.35)
            cap = STAKE_CFG["unit_cap_high"] if conf=="High" else STAKE_CFG["unit_cap_med"]
            units = clamp(units, 0.0, cap)
            units += STAKE_CFG["league_unit_adj"].get(lg, 0.0)
            units = max(units, 0.0)
            if units>0: stake = round(units, 2)

        return {
            "League":lg, "Home":h, "Away":a,
            "P1":float(P_cal[0]), "PX":float(P_cal[1]), "P2":float(P_cal[2]),
            "Pick":{0:"1",1:"X",2:"2"}[pick_idx], "Confidence":conf,
            "UpsetWin":up_win_cal, "UpsetCover":up_cover_cal,
            "UpsetWin_LB":up_win_lb, "UpsetCover_LB":up_cover_lb,
            "Pred_xG_H":float(mu_h), "Pred_xG_A":float(mu_a),
            "P_BTTS":float(p_btts), "P_Under2.5":float(p_u25c),
            "AH_underdog": ah_rec, "AH_stake_units": stake,
            "Recommendation": side_rec,
            "Flags": ",".join(flags) if flags else ""
        }

    def predict_df(self, fixtures_df: pd.DataFrame):
        out=[]
        with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as ex:
            futs = {ex.submit(self.predict_row, r, int(r.get("Season",TEST_YEAR))): idx
                    for idx,(_,r) in enumerate(fixtures_df.iterrows())}
            for fut in as_completed(futs):
                try: out.append(fut.result())
                except Exception: pass
        return pd.DataFrame(out)

# ================ Evaluation & ROI =================

def evaluate_backtest(preds_df, test_df):
    merged=preds_df.merge(test_df, on=["League","Season","Home","Away"], how="left", suffixes=("","_T"))
    rows=[]; deciles=[]
    for lg, part in merged.groupby("League"):
        if "Target" not in part.columns or part["Target"].isna().all(): continue
        y_true = part["Target"].astype(int).values
        P = part[["P1","PX","P2"]].values
        acc = (part["Pick"]==part["Target"].map({0:"1",1:"X",2:"2"})).mean()
        brier=np.mean(np.sum((P - np.eye(3)[y_true])**2, axis=1))
        ll = -np.mean(np.log(np.maximum([P[i, y_true[i]] for i in range(len(y_true))], 1e-12)))
        for col in ["UpsetWin","UpsetCover"]:
            try:
                q = pd.qcut(part[col], 10, labels=False, duplicates="drop")
                is_fail = (P.argmax(1)!=y_true).astype(int)
                df = pd.DataFrame({"dec":q, "fail":is_fail})
                d = df.groupby("dec")["fail"].mean().reset_index()
                d["League"]=lg; d["Metric"]=col; deciles.append(d)
            except Exception: pass
        rows.append({"League":lg,"Matches":len(part),"Accuracy%":round(acc*100,1),"Brier":round(brier,3),"LogLoss":round(ll,3)})
    dec_out = pd.concat(deciles, ignore_index=True) if deciles else pd.DataFrame()
    return pd.DataFrame(rows), dec_out

def roi_sim(preds_df, raw_df, stake_cfg=STAKE_CFG, use_synthetic_ah=SYNTH_AH_PRICING["ENABLE"]):
    df=preds_df.merge(raw_df, on=["League","Season","Home","Away"], how="left", suffixes=("","_raw"))
    bets=[]
    for _,r in df.iterrows():
        ah = r.get("AH_underdog", None)
        if isinstance(ah, dict) and ah.get("o") and r.get("AH_stake_units",0)>0:
            hg, ag = int(r.get("HG",np.nan)), int(r.get("AG",np.nan))
            if not np.isfinite(hg) or not np.isfinite(ag): continue
            side = ah["side"]; line=float(ah["line"]); o=float(ah["o"]); units=float(r.get("AH_stake_units",0.0))
            margin = (hg - ag) if side=="home" else (ag - hg)
            win = (margin + line > 0)
            pnl = units*(o-1.0) if win else -units
            bets.append({"League":r["League"],"Product":f"AH {side} {line:+}", "Units":units, "PNL":pnl})
            continue
        conf=r["Confidence"]
        if conf not in ("High","Medium"): continue
        idx={"1":0,"X":1,"2":2}[r["Pick"]]
        p = float(r[["P1","PX","P2"]].values[idx])
        o = float(r[["O_H","O_X","O_A"]].values[idx])
        if not np.isfinite(o) or o<=1.0: continue
        units = kelly_stake(p, o, frac=stake_cfg["kelly_frac"])
        cap = stake_cfg["unit_cap_high"] if conf=="High" else stake_cfg["unit_cap_med"]
        units = clamp(units, 0.0, cap)
        units += stake_cfg["league_unit_adj"].get(r["League"], 0.0)
        units = max(units, 0.0)
        if units<=0: continue
        res = r.get("Target", np.nan)
        won = (res == {0:"1",1:"X",2:"2"}[idx])
        pnl = units*(o-1.0) if won else -units
        bets.append({"League":r["League"],"Product":f"1X2 {r['Pick']}", "Units":units, "PNL":pnl})
    if not bets: return pd.DataFrame(), {}
    B=pd.DataFrame(bets)
    report={"Total_Bets":len(B), "Total_Units":float(B["Units"].sum()), "Total_PNL":float(B["PNL"].sum()),
            "ROI%":float(100.0*B["PNL"].sum()/max(1e-9,B["Units"].sum()))}
    by_lg = B.groupby("League").agg(Bets=("Units","size"), Units=("Units","sum"), PNL=("PNL","sum"))
    by_lg["ROI%"]=100.0*by_lg["PNL"]/by_lg["Units"].clip(lower=1e-9)
    return B, {"overall":report, "by_league":by_lg.reset_index()}

# =================== Fixtures parser ===================

def parse_fixtures(text):
    lines=[ln.strip() for ln in text.strip().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    fixtures=[]; league=""
    for line in lines:
        if ":" in line:
            parts=line.split(":",1); league=parts[1].strip()
            league=LEAGUE_MAP.get(league, league)
        else:
            teams=[t.strip() for t in line.split("-",1)]
            if len(teams)==2: fixtures.append({"League":league,"Home":teams[0],"Away":teams[1]})
    return pd.DataFrame(fixtures)

# =================== MAIN ===================

FIXTURES_TEXT = """
# Exempel:
# ENGLAND: Premier League
# Aston Villa - Crystal Palace
"""

def _load_hist(leagues, years):
    dm = QuantumDataManager(CACHE_ROOT, timeout=15.0, rate_limit=1.6)
    hist_raw=[]
    with ThreadPoolExecutor(max_workers=min(8, len(leagues)*len(years))) as ex:
        futs=[ex.submit(dm.fetch_season, lg, yr) for lg in leagues for yr in years]
        for fut in as_completed(futs):
            try:
                df=fut.result()
                if df is not None and not df.empty: hist_raw.append(df)
            except Exception: pass
    return pd.concat(hist_raw, ignore_index=True) if hist_raw else pd.DataFrame()

if __name__=="__main__":
    fixtures_df=parse_fixtures(FIXTURES_TEXT)
    if fixtures_df.empty:
        leagues=["Premier League","LaLiga","Bundesliga","Serie A","Ligue 1","Eredivisie","Primeira Liga","Jupiler Pro League"]
    else:
        leagues = sorted(set(fixtures_df["League"].astype(str).tolist()))
    hist = _load_hist(leagues, TRAIN_YEARS + [TEST_YEAR])
    if hist.empty: print("[warn] Ingen historik laddades – modellen kan ej tränas korrekt.")
    data = prep_base(hist); data = add_venue_form(data); data = add_market_drift(data)
    model=ApexSuperModelV891(conformal_level=0.85)
    if not data.empty: model.fit(data)

    if fixtures_df.empty:
        dm = QuantumDataManager(CACHE_ROOT, timeout=15.0, rate_limit=1.6)
        test_raw=[]
        with ThreadPoolExecutor(max_workers=min(8, len(leagues))) as ex:
            futs=[ex.submit(dm.fetch_season, lg, TEST_YEAR) for lg in leagues]
            for fut in as_completed(futs):
                try:
                    df=fut.result()
                    if df is not None and not df.empty: test_raw.append(df)
                except Exception: pass
        test=pd.concat(test_raw, ignore_index=True) if test_raw else pd.DataFrame()
        test=prep_base(test); test=add_venue_form(test); test=add_market_drift(test)
        preds=model.predict_df(test.dropna(subset=["Home","Away","League"]))
        if preds.empty:
            print("No predictions."); sys.exit(0)

        perf, dec = evaluate_backtest(preds, test)
        print("\n=== Backtest Performance (",TEST_YEAR,") ===")
        if not perf.empty: print(perf[["League","Matches","Accuracy%","Brier","LogLoss"]].to_string(index=False))
        if not dec.empty:
            print("\n=== Upset deciles (empirical fav-failure rate) ===")
            print(dec.groupby(["Metric","dec"])["fail"].mean().reset_index().to_string(index=False))

        bets, report = roi_sim(preds, test, stake_cfg=STAKE_CFG, use_synthetic_ah=SYNTH_AH_PRICING["ENABLE"])
        if report:
            print("\n=== ROI Simulator ===")
            print(f"Total Bets: {report['overall']['Total_Bets']}, Units: {report['overall']['Total_Units']:.2f}, "
                  f"PNL: {report['overall']['Total_PNL']:.2f}, ROI: {report['overall']['ROI%']:.1f}%")
            if "by_league" in report and not report["by_league"].empty:
                print("\nPer league:")
                print(report["by_league"][["League","Bets","Units","PNL","ROI%"]].to_string(index=False))

        preds.to_csv("predictions_supermodel_v89_1_backtest.csv", index=False)
        if isinstance(bets, pd.DataFrame) and not bets.empty:
            bets.to_csv("bets_supermodel_v89_1_backtest.csv", index=False)
        print("\nSaved: predictions_supermodel_v89_1_backtest.csv",
              "(+ bets_supermodel_v89_1_backtest.csv)" if isinstance(bets, pd.DataFrame) and not bets.empty else "")
    else:
        fixtures_df["Season"]=TEST_YEAR+1
        fixtures_df["Date"]=pd.Timestamp.today()+pd.Timedelta(days=1)
        fixtures_df["Home_Rest"]=7.0; fixtures_df["Away_Rest"]=7.0
        fixtures_df["Imp_H"]=fixtures_df["Imp_X"]=fixtures_df["Imp_A"]=1/3
        for c in BASE_UPSET_FEATS:
            if c not in fixtures_df.columns: fixtures_df[c]=0.0
        preds=model.predict_df(fixtures_df)
        if preds.empty:
            print("No predictions for fixtures.")
        else:
            view=preds.copy()
            for c in ["P1","PX","P2","UpsetWin","UpsetCover","UpsetWin_LB","UpsetCover_LB","P_BTTS","P_Under2.5"]:
                view[c]=(view[c].astype(float)*100).round(1).astype(str)+"%"
            print("\n=== Predictions (V89.1) ===")
            cols=["League","Home","Away","P1","PX","P2","Pick","Confidence",
                  "UpsetWin","UpsetCover","UpsetWin_LB","UpsetCover_LB",
                  "P_BTTS","P_Under2.5","Recommendation","AH_underdog","AH_stake_units","Flags"]
            print(view[cols].to_string(index=False))
            preds.to_csv("predictions_supermodel_v89_1.csv", index=False)
            print("\nSaved: predictions_supermodel_v89_1.csv")
