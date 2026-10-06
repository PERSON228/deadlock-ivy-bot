"""
Deadlock Style Map bot.

Строит 2D-карту стиля игры по Steam ID32 на выбранном герое (по умолчанию Ivy).
Серые/цветные точки: реальные игроки с этого героя из Deadlock API.
Белые кольца: топ игроки на герое. Белые ромбы с подписями: киберспортсмены из pros.json.

Запрос боту:  <steam id32> [герой]    например:  291654238   или   291654238 haze

Переменные окружения:
  BOT_TOKEN        обязательно, токен из @BotFather
  PORT             порт для health-check (Render ставит сам)
  REF_ACCOUNT_IDS  необязательно, список account_id через запятую для базы
  PRO_PLAYERS      необязательно, "Ник:account_id,Ник:account_id" (дополняет pros.json)

Файл pros.json рядом с bot.py:  {"Ник": 123456789, "Другой ник": 987654321}
"""
import os
import io
import json
import asyncio
import logging
import time

import numpy as np
import aiohttp
from aiohttp import web

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import BufferedInputFile

log = logging.getLogger("deadlock-style")

API = "https://api.deadlock-api.com/v1"
ASSETS = "https://assets.deadlock-api.com/v2"
DEFAULT_HERO_ID = 20  # Ivy
HERO_FALLBACK = {20: "Ivy"}
REGIONS = ["Europe", "Asia", "NAmerica", "SAmerica", "Oceania"]
TOP_PER_REGION = 10
TOP_N = 30
MIN_MATCHES_REF = 10    # минимум матчей на герое, чтобы попасть в базу
MIN_MATCHES_USER = 3
MIN_MATCHES_PRO = 5
MIN_MATCHES_TOP_WR = 50  # для запасного топа по винрейту
REF_TTL = 6 * 3600
USER_COOLDOWN = 20

LABELS = {
    "kills_per_min": "убийства/мин",
    "deaths_per_min": "смерти/мин",
    "assists_per_min": "ассисты/мин",
    "denies_per_min": "дени/мин",
    "networth_per_min": "души/мин",
    "net_worth_per_min": "души/мин",
    "last_hits_per_min": "добивания/мин",
    "damage_per_min": "урон/мин",
    "damage_taken_per_min": "полученный урон/мин",
    "creeps_per_min": "крипы/мин",
    "creep_kills_per_min": "крипы/мин",
    "neutral_damage_per_min": "урон по нейтралам/мин",
    "boss_damage_per_min": "урон по боссам/мин",
    "shots_hit_per_min": "попадания/мин",
    "shots_missed_per_min": "промахи/мин",
    "hero_bullets_hit_per_min": "попадания по героям/мин",
    "hero_bullets_hit_crit_per_min": "криты по героям/мин",
    "enemy_bullets_hit_per_min": "попадания по героям/мин",
    "enemy_bullets_hit_crit_per_min": "криты по героям/мин",
    "accuracy": "точность",
}

WORDS = {
    "damage": "урон", "taken": "полученный", "mitigated": "поглощённый", "creeps": "крипы",
    "creep": "крипы", "kills": "убийства", "deaths": "смерти", "assists": "ассисты",
    "denies": "дени", "networth": "души", "net": "", "worth": "души", "souls": "души",
    "last": "добивания", "hits": "", "healing": "лечение", "heal": "лечение",
    "neutral": "нейтралы", "boss": "боссы", "shots": "выстрелы", "hit": "попадания",
    "missed": "промахи", "hero": "по героям", "bullets": "", "crit": "криты",
    "accuracy": "точность", "health": "здоровье", "barriers": "барьеры",
    "self": "урон себе", "orb": "сферы", "objective": "объекты",
}

# поля, которые не описывают стиль
SKIP_KEYS = {"account_id", "hero_id", "wins", "losses", "matches", "matches_played",
             "winrate", "win_rate", "ending_level", "max_health", "duration_s"}


# ---------------------------------------------------------------- API

_session: aiohttp.ClientSession | None = None
_sem = asyncio.Semaphore(4)


async def session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=40),
            headers={"User-Agent": "deadlock-style-bot/1.1"},
        )
    return _session


async def get_json(url: str, params: dict | None = None, tries: int = 3):
    """GET с повторами. Возвращает распарсенный JSON или None."""
    for attempt in range(tries):
        try:
            async with _sem:
                s = await session()
                async with s.get(url, params=params) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    if r.status in (429, 502, 503, 504):
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    log.warning("GET %s -> %s", url, r.status)
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("GET %s failed: %s", url, e)
            await asyncio.sleep(1.0 * (attempt + 1))
    return None


_heroes_cache: dict = {}


async def load_heroes() -> dict:
    """{'by_name': {имя: id}, 'by_id': {id: имя}}. Без сети откатывается на одну Ivy."""
    if _heroes_cache:
        return _heroes_cache
    data = await get_json(f"{ASSETS}/heroes")
    by_name, by_id = {}, {}
    if isinstance(data, list):
        for h in data:
            if not isinstance(h, dict) or "id" not in h:
                continue
            try:
                hid = int(h["id"])
            except (TypeError, ValueError):
                continue
            name = str(h.get("name") or h.get("class_name") or "").strip()
            if not name:
                continue
            by_id[hid] = name
            by_name[name.lower()] = hid
            cn = str(h.get("class_name") or "").lower().replace("hero_", "")
            if cn:
                by_name[cn] = hid
    if by_id:
        _heroes_cache.update({"by_name": by_name, "by_id": by_id})
        return _heroes_cache
    return {"by_name": {"ivy": 20}, "by_id": dict(HERO_FALLBACK)}


async def resolve_hero(query: str | None):
    """Возвращает (hero_id, имя) или None, если героя не нашли."""
    heroes = await load_heroes()
    if not query:
        return DEFAULT_HERO_ID, heroes["by_id"].get(DEFAULT_HERO_ID, "Ivy")
    q = query.strip().lower()
    if q.isdigit():
        hid = int(q)
        return hid, heroes["by_id"].get(hid, f"герой {hid}")
    if q in heroes["by_name"]:
        hid = heroes["by_name"][q]
        return hid, heroes["by_id"][hid]
    cands = {hid for n, hid in heroes["by_name"].items() if n.startswith(q)}
    if len(cands) == 1:
        hid = cands.pop()
        return hid, heroes["by_id"][hid]
    return None


async def hero_stats(account_ids: list[int], hero_id: int) -> list[dict]:
    """Агрегаты игрок+герой через bulk-эндпоинт (до 1000 аккаунтов за запрос)."""
    out: list[dict] = []
    for i in range(0, len(account_ids), 1000):
        chunk = account_ids[i:i + 1000]
        data = await get_json(
            f"{API}/players/hero-stats",
            {"account_ids": ",".join(map(str, chunk)), "hero_ids": str(hero_id)},
        )
        if isinstance(data, list):
            out.extend(d for d in data if isinstance(d, dict))
    return out


def _extract_ids(data) -> list[int]:
    ids = []
    if isinstance(data, dict):
        data = data.get("data") or data.get("results") or data.get("players") or []
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict) and d.get("account_id") is not None:
                try:
                    ids.append(int(d["account_id"]))
                except (TypeError, ValueError):
                    pass
    return ids


async def candidate_accounts(hero_id: int) -> list[int]:
    """Список account_id игроков с этим героем для базы."""
    env_ids = os.getenv("REF_ACCOUNT_IDS", "").strip()
    if env_ids:
        return [int(x) for x in env_ids.split(",") if x.strip().isdigit()]

    attempts = [
        (f"{API}/analytics/scoreboards/players",
         {"hero_id": hero_id, "sort_by": "matches", "sort_direction": "desc", "limit": 1000}),
        (f"{API}/analytics/scoreboards/players",
         {"hero_id": hero_id, "sort_by": "matches", "limit": 1000}),
        (f"{API}/players/scoreboard",
         {"hero_id": hero_id, "sort_by": "matches", "limit": 1000}),
    ]
    for url, params in attempts:
        ids = _extract_ids(await get_json(url, params, tries=1))
        if ids:
            return list(dict.fromkeys(ids))
    return []


def _aid(row: dict):
    try:
        return int(row.get("account_id"))
    except (TypeError, ValueError):
        return None


_ref_cache: dict[int, tuple[float, list[dict]]] = {}


async def reference_rows(hero_id: int) -> list[dict]:
    cached = _ref_cache.get(hero_id)
    if cached and time.time() - cached[0] < REF_TTL:
        return cached[1]
    ids = await candidate_accounts(hero_id)
    rows = await hero_stats(ids, hero_id) if ids else []
    rows = [r for r in rows if matches_of(r) >= MIN_MATCHES_REF]
    if rows:
        _ref_cache[hero_id] = (time.time(), rows)
    log.info("reference for hero %s: %d players", hero_id, len(rows))
    return rows


# --- топ лидерборда и про-игроки

async def leaderboard_top(hero_id: int):
    """Топ лидерборда героя по регионам: (список account_id, {id: ник})."""
    async def one(region: str):
        data = await get_json(f"{API}/leaderboard/{region}/{hero_id}", tries=1)
        entries = data.get("entries") if isinstance(data, dict) else data
        out = []
        if isinstance(entries, list):
            for e in entries[:TOP_PER_REGION]:
                if not isinstance(e, dict):
                    continue
                ids = e.get("possible_account_ids") or ([e["account_id"]] if e.get("account_id") else [])
                try:
                    out.append((int(ids[0]), str(e.get("account_name") or "")))
                except (TypeError, ValueError, IndexError):
                    continue
        return out

    results = await asyncio.gather(*(one(r) for r in REGIONS))
    names: dict[int, str] = {}
    for part in results:
        for acc, name in part:
            names.setdefault(acc, name)
    return list(names.keys()), names


_top_cache: dict[int, tuple[float, tuple]] = {}


async def top_rows(hero_id: int):
    cached = _top_cache.get(hero_id)
    if cached and time.time() - cached[0] < REF_TTL:
        return cached[1]
    ids, names = await leaderboard_top(hero_id)
    rows = []
    if ids:
        rows = [r for r in await hero_stats(ids, hero_id) if matches_of(r) >= MIN_MATCHES_USER]
    out = (rows, names)
    if rows:
        _top_cache[hero_id] = (time.time(), out)
    return out


def load_pros() -> dict[str, int]:
    pros: dict[str, int] = {}
    for item in os.getenv("PRO_PLAYERS", "").split(","):
        name, _, acc = item.rpartition(":")
        if name.strip() and acc.strip().isdigit():
            pros[name.strip()] = int(acc)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pros.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if str(v).isdigit():
                        pros[str(k)] = int(v)
        except Exception:
            log.exception("не удалось прочитать pros.json")
    return pros


# ---------------------------------------------------------------- анализ

def matches_of(row: dict) -> int:
    for k in ("matches_played", "matches"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, list):
            return len(v)
    return 0


def _num(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def style_keys(rows: list[dict]) -> list[str]:
    """Автовыбор признаков стиля: нормированные показатели (в минуту, точность, на душу)."""
    if not rows:
        return []
    keys = []
    for k, v in rows[0].items():
        if k in SKIP_KEYS or isinstance(v, (list, dict, bool)):
            continue
        if not isinstance(v, (int, float)):
            continue
        if "per_min" in k or "per_soul" in k or k == "accuracy":
            keys.append(k)
    good = []
    for k in keys:
        col = np.array([_num(r.get(k)) for r in rows], dtype=float)
        if np.isfinite(col).mean() > 0.9 and np.nanstd(col) > 1e-9:
            good.append(k)
    return good


def matrix(rows: list[dict], keys: list[str]) -> np.ndarray:
    return np.array([[_num(r.get(k)) for k in keys] for r in rows], dtype=float)


def pretty(k: str) -> str:
    if k in LABELS:
        return LABELS[k]
    left, _, right = k.partition("_per_")
    words = [WORDS.get(w, w) for w in left.split("_")]
    words.sort(key=lambda w: w not in ("полученный", "поглощённый"))  # прилагательные вперёд
    name = " ".join(w for w in words if w) or left
    if right == "min":
        return f"{name}/мин"
    if right == "soul":
        return f"{name} на душу"
    if right:
        return f"{name} на {right}"
    return name


def pick_k(X: np.ndarray) -> int:
    n = len(X)
    best_k, best_s = 3, -1.0
    for k in range(3, min(7, n // 8 + 1)):
        try:
            labels = KMeans(k, n_init=5, random_state=0).fit_predict(X)
            s = silhouette_score(X, labels)
        except Exception:
            continue
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def build_map(ref_rows: list[dict], user_row: dict | None, keys: list[str],
              groups: list[dict] | None = None) -> dict:
    """Все расчёты карты без сети (удобно тестировать).

    groups: [{"label": str, "rows": [...], "names": [...] | None, "style": "top" | "pro"}]
    Группы проецируются в ту же систему осей, что и база, но в обучении осей не участвуют.
    """
    R = matrix(ref_rows, keys)
    med = np.nanmedian(R, axis=0)
    R = np.where(np.isnan(R), med, R)
    lo, hi = np.percentile(R, 1, axis=0), np.percentile(R, 99, axis=0)
    R = np.clip(R, lo, hi)  # выбросы базы не должны тянуть оси

    scaler = StandardScaler().fit(R)
    Rz = scaler.transform(R)

    def to_z(rows: list[dict]) -> np.ndarray:
        M = matrix(rows, keys)
        M = np.where(np.isnan(M), med, M)
        return scaler.transform(M)  # без обрезки: про и вы могут быть экстремальнее базы

    pca = PCA(n_components=2, random_state=0).fit(Rz)
    R2 = pca.transform(Rz)

    # кластеры считаем в пространстве до 6 компонент, а не в 2D
    ncomp = min(6, Rz.shape[1], Rz.shape[0] - 1)
    pc = PCA(n_components=ncomp, random_state=0).fit(Rz)
    Rc = pc.transform(Rz)
    k = pick_k(Rc)
    km = KMeans(k, n_init=10, random_state=0).fit(Rc)
    labels = km.labels_

    res = {"ref2": R2, "labels": labels, "k": k, "keys": keys,
           "evr": pca.explained_variance_ratio_}

    def axis_name(comp):
        order = np.argsort(comp)
        neg = [pretty(keys[i]) for i in order[:2]]
        pos = [pretty(keys[i]) for i in order[::-1][:2]]
        return " + ".join(neg), " + ".join(pos)
    res["xaxis"] = axis_name(pca.components_[0])
    res["yaxis"] = axis_name(pca.components_[1])

    uz = None
    if user_row is not None:
        uz = to_z([user_row])
        res["user2"] = pca.transform(uz)[0]
        res["user_z"] = uz[0]
        res["user_cluster"] = int(km.predict(pc.transform(uz))[0])
        res["percentile_rank"] = (Rz < uz).mean(axis=0)

    res["groups"] = []
    res["nearest"] = {}
    for g in groups or []:
        if not g["rows"]:
            continue
        gz = to_z(g["rows"])
        entry = {"label": g["label"], "style": g["style"], "names": g.get("names"),
                 "xy": pca.transform(gz)}
        res["groups"].append(entry)
        if uz is not None and g.get("names"):
            d = np.linalg.norm(gz - uz, axis=1)
            seen, near = set(), []
            for i in np.argsort(d):
                nm = g["names"][i]
                if nm and nm not in seen:
                    seen.add(nm)
                    near.append(nm)
                if len(near) == 3:
                    break
            if near:
                res["nearest"][g["label"]] = near

    desc = {}
    for c in range(k):
        mask = labels == c
        m = Rz[mask].mean(axis=0)
        top = np.argsort(-np.abs(m))[:2]
        desc[c] = ", ".join(
            f"{pretty(keys[i])} {'выше' if m[i] > 0 else 'ниже'} среднего" for i in top)
        desc[c] += f" ({mask.sum()} игр.)"
    res["cluster_desc"] = desc
    return res


PALETTE = ["#89b4fa", "#a6e3a1", "#f9e2af", "#cba6f7", "#94e2d5", "#fab387"]


def render_map(res: dict, title: str, ref_n: int) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#1e1e2e")
    ax.set_facecolor("#181825")
    R2, labels = res["ref2"], res["labels"]
    for c in range(res["k"]):
        m = labels == c
        ax.scatter(R2[m, 0], R2[m, 1], s=22, alpha=0.38,
                   color=PALETTE[c % len(PALETTE)], label=f"кластер {c + 1}")

    for g in res.get("groups", []):
        xy = g["xy"]
        label = f"{g['label']} ({len(xy)})"
        if g["style"] == "pro":
            ax.scatter(xy[:, 0], xy[:, 1], s=120, marker="D", color="white",
                       edgecolor="#11111b", linewidth=1.0, zorder=4, label=label)
            for (x, y), name in zip(xy, g["names"] or []):
                ax.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points",
                            color="white", fontsize=8, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.15", fc="#11111b", ec="none", alpha=0.75))
        else:
            ax.scatter(xy[:, 0], xy[:, 1], s=85, facecolors="none", edgecolors="white",
                       linewidth=1.5, zorder=3, label=label)

    if "user2" in res:
        ux, uy = res["user2"]
        ax.scatter([ux], [uy], s=340, marker="*", color="#f38ba8",
                   edgecolor="white", linewidth=1.2, zorder=7, label="Вы")

    ax.axhline(0, color="#585b70", ls="--", lw=1, alpha=0.5)
    ax.axvline(0, color="#585b70", ls="--", lw=1, alpha=0.5)
    ev = res["evr"]
    xn, xp = res["xaxis"]
    yn, yp = res["yaxis"]
    ax.set_xlabel(f"← {xn}      ось 1 ({ev[0]:.0%})      {xp} →", color="#cdd6f4", fontsize=9)
    ax.set_ylabel(f"← {yn}      ось 2 ({ev[1]:.0%})      {yp} →", color="#cdd6f4", fontsize=9)
    ax.set_title(f"{title}  ·  база {ref_n} игроков", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="#a6adc8")
    ax.grid(True, color="#313244", ls=":", alpha=0.5)
    for s in ax.spines.values():
        s.set_color("#45475a")
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="white", fontsize=8, loc="best")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def describe_user(res: dict, user_row: dict) -> str:
    keys, z, pr = res["keys"], res["user_z"], res["percentile_rank"]
    c = res["user_cluster"]
    parts = [f"Ваш кластер {c + 1}: {res['cluster_desc'][c]}"]
    for label, names in res.get("nearest", {}).items():
        parts.append(f"Ближе всего по стилю ({label}): {', '.join(names)}")
    order = np.argsort(-np.abs(z))[:4]
    lines = [f"• {pretty(keys[i])}: {_num(user_row.get(keys[i])):.2f} "
             f"(выше, чем у {pr[i]:.0%} игроков)" for i in order]
    parts.append("Самое необычное в вашей игре:\n" + "\n".join(lines))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- бот

async def _no_rows() -> list:
    return []


async def analyze(account_id: int, hero_id: int, hero_name: str):
    """Возвращает (png_bytes | None, caption)."""
    pros = load_pros()
    id_to_pro = {v: k for k, v in pros.items()}
    ref, mine, (top, top_names), pro_rows = await asyncio.gather(
        reference_rows(hero_id),
        hero_stats([account_id], hero_id),
        top_rows(hero_id),
        hero_stats(list(pros.values()), hero_id) if pros else _no_rows(),
    )
    user_row = mine[0] if mine else None

    if user_row is None:
        return None, f"Нет данных по этому I
