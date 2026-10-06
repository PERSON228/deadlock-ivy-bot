"""
Deadlock Style Map bot.

Строит 2D-карту стиля игры по Steam ID32 на выбранном герое (по умолчанию Ivy).
Точки "базы" это реальные игроки с этого героя из Deadlock API, а не случайные числа.

Переменные окружения:
  BOT_TOKEN        обязательно, токен из @BotFather
  PORT             порт для health-check (Render ставит сам)
  REF_ACCOUNT_IDS  необязательно, список account_id через запятую для базы,
                   если автоматическая выгрузка топа не сработает
"""
import os
import io
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
MIN_MATCHES_REF = 10   # минимум матчей на герое, чтобы попасть в базу
MIN_MATCHES_USER = 3
REF_TTL = 6 * 3600
USER_COOLDOWN = 20

# русские подписи для известных полей, остальные показываем как есть
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
    "creep_kills_per_min": "крипы/мин",
    "neutral_damage_per_min": "урон по нейтралам/мин",
    "boss_damage_per_min": "урон по боссам/мин",
    "shots_hit_per_min": "попадания/мин",
    "shots_missed_per_min": "промахи/мин",
    "hero_bullets_hit_per_min": "попадания по героям/мин",
    "hero_bullets_hit_crit_per_min": "криты по героям/мин",
    "accuracy": "точность",
    "enemy_bullets_hit_per_min": "попадания по героям/мин",
    "enemy_bullets_hit_crit_per_min": "криты по героям/мин",
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
            headers={"User-Agent": "deadlock-style-bot/1.0"},
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
    """{имя в нижнем регистре: id} и {id: имя}."""
    if _heroes_cache:
        return _heroes_cache
    data = await get_json(f"{ASSETS}/heroes")
    by_name, by_id = {}, {}
    if isinstance(data, list):
        for h in data:
            if isinstance(h, dict) and "id" in h and "name" in h:
                by_name[str(h["name"]).lower()] = int(h["id"])
                by_id[int(h["id"])] = str(h["name"])
    _heroes_cache.update({"by_name": by_name, "by_id": by_id})
    return _heroes_cache


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


async def user_all_heroes(account_id: int) -> list[dict]:
    data = await get_json(f"{API}/players/hero-stats", {"account_ids": str(account_id)})
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


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


_ref_cache: dict[int, tuple[float, list[dict]]] = {}


async def reference_rows(hero_id: int) -> list[dict]:
    cached = _ref_cache.get(hero_id)
    if cached and time.time() - cached[0] < REF_TTL:
        return cached[1]
    ids = await candidate_accounts(hero_id)
    rows = await hero_stats(ids, hero_id) if ids else []
    rows = [r for r in rows if matches_of(r) >= MIN_MATCHES_REF]
    _ref_cache[hero_id] = (time.time(), rows)
    log.info("reference for hero %s: %d players", hero_id, len(rows))
    return rows


# ---------------------------------------------------------------- анализ

def matches_of(row: dict) -> int:
    for k in ("matches_played", "matches"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, list):
            return len(v)
    return 0


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
    # оставляем только те, что есть у большинства строк и не константа
    good = []
    for k in keys:
        col = np.array([_num(r.get(k)) for r in rows], dtype=float)
        if np.isfinite(col).mean() > 0.9 and np.nanstd(col) > 1e-9:
            good.append(k)
    return good


def _num(v) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def matrix(rows: list[dict], keys: list[str]) -> np.ndarray:
    return np.array([[_num(r.get(k)) for k in keys] for r in rows], dtype=float)


def pretty(k: str) -> str:
    return LABELS.get(k, k.replace("_", " "))


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


def build_map(ref_rows: list[dict], user_row: dict | None, keys: list[str]) -> dict:
    """Чистая функция без сети: все расчёты карты. Удобно тестировать."""
    R = matrix(ref_rows, keys)
    med = np.nanmedian(R, axis=0)
    R = np.where(np.isnan(R), med, R)
    lo, hi = np.percentile(R, 1, axis=0), np.percentile(R, 99, axis=0)
    R = np.clip(R, lo, hi)

    scaler = StandardScaler().fit(R)
    Rz = scaler.transform(R)

    pca = PCA(n_components=2, random_state=0).fit(Rz)
    R2 = pca.transform(Rz)

    # кластеризуем в пространстве до 6 компонент, не в 2D
    ncomp = min(6, Rz.shape[1], Rz.shape[0] - 1)
    pc = PCA(n_components=ncomp, random_state=0).fit(Rz)
    Rc = pc.transform(Rz)
    k = pick_k(Rc)
    km = KMeans(k, n_init=10, random_state=0).fit(Rc)
    labels = km.labels_

    res = {"ref2": R2, "labels": labels, "k": k, "pca": pca, "keys": keys,
           "evr": pca.explained_variance_ratio_}

    # оси: какие признаки сильнее всего тянут влево/вправо и вниз/вверх
    def axis_name(comp):
        order = np.argsort(comp)
        neg = [pretty(keys[i]) for i in order[:2]]
        pos = [pretty(keys[i]) for i in order[::-1][:2]]
        return " + ".join(neg), " + ".join(pos)
    res["xaxis"] = axis_name(pca.components_[0])
    res["yaxis"] = axis_name(pca.components_[1])

    if user_row is not None:
        u = np.array([_num(user_row.get(kk)) for kk in keys], dtype=float)
        u = np.where(np.isnan(u), med, u)
        u = np.clip(u, lo, hi)
        uz = scaler.transform(u.reshape(1, -1))
        res["user2"] = pca.transform(uz)[0]
        res["user_z"] = uz[0]
        res["user_cluster"] = int(km.predict(pc.transform(uz))[0])
        # похожие игроки: ближайшие в полном стандартизованном пространстве
        d = np.linalg.norm(Rz - uz, axis=1)
        res["nearest_idx"] = np.argsort(d)[:3]
        res["percentile_rank"] = (Rz < uz).mean(axis=0)

    # описание кластеров: самые выраженные признаки относительно общего среднего
    desc = {}
    for c in range(k):
        mask = res["labels"] == c
        m = Rz[mask].mean(axis=0)
        top = np.argsort(-np.abs(m))[:2]
        desc[c] = ", ".join(
            f"{'выше' if m[i] > 0 else 'ниже'} {pretty(keys[i])}" for i in top)
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
        ax.scatter(R2[m, 0], R2[m, 1], s=26, alpha=0.55,
                   color=PALETTE[c % len(PALETTE)], label=f"кластер {c + 1}")
    if "user2" in res:
        ux, uy = res["user2"]
        ax.scatter([ux], [uy], s=320, marker="*", color="#f38ba8",
                   edgecolor="white", linewidth=1.2, zorder=5, label="Вы")
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
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="white", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def describe_user(res: dict, user_row: dict) -> str:
    keys, z, pr = res["keys"], res["user_z"], res["percentile_rank"]
    order = np.argsort(-np.abs(z))[:5]
    lines = []
    for i in order:
        lines.append(f"• {pretty(keys[i])}: {_num(user_row.get(keys[i])):.2f} "
                     f"(лучше {pr[i]:.0%} игроков)")
    c = res["user_cluster"]
    out = f"Ваш кластер {c + 1}: {res['cluster_desc'][c]}\n\nСамое необычное в вашей игре:\n" + "\n".join(lines)
    return out


# ---------------------------------------------------------------- бот

async def analyze(account_id: int, hero_id: int, hero_name: str):
    """Возвращает (png_bytes | None, caption)."""
    ref = await reference_rows(hero_id)
    mine = [r for r in await hero_stats([account_id], hero_id)]
    user_row = mine[0] if mine else None

    if user_row is not None and matches_of(user_row) < MIN_MATCHES_USER:
        return None, (f"На {hero_name} у вас всего {matches_of(user_row)} матч(ей) в базе API. "
                      f"Нужно хотя бы {MIN_MATCHES_USER}, лучше 20+.")

    if len(ref) >= 40 and user_row is not None:
        keys = style_keys(ref)
        if len(keys) < 3:
            return None, "API вернул слишком мало подходящих показателей для карты."
        res = await asyncio.to_thread(build_map, ref, user_row, keys)
        png = await asyncio.to_thread(render_map, res, f"Стиль на {hero_name}", len(ref))
        cap = (f"{hero_name}: {matches_of(user_row)} матчей\n\n" + describe_user(res, user_row))
        return png, cap[:1020]

    # запасной режим: нет базы, строим карту по вашим героям
    allh = [r for r in await user_all_heroes(account_id) if matches_of(r) >= MIN_MATCHES_USER]
    if len(allh) < 4:
        return None, ("Не удалось получить базу игроков и у вас мало героев с матчами, "
                      "карту построить не из чего.")
    keys = style_keys(allh)
    if len(keys) < 3:
        return None, "API вернул слишком мало подходящих показателей."
    res = await asyncio.to_thread(build_map, allh, None, keys)
    png = await asyncio.to_thread(render_map, res, "Ваши герои по стилю", len(allh))
    cap = ("База игроков недоступна, поэтому показаны ваши герои: близкие точки это "
           "герои, на которых вы играете похоже.\nГероев на карте: " + str(len(allh)))
    return png, cap


_last_call: dict[int, float] = {}


def make_dispatcher() -> Dispatcher:
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def start(m: types.Message):
        await m.answer(
            "Пришли Steam ID32 (например 291654238) и карту стиля получишь на Ivy.\n"
            "Другой герой: ID и имя, например `291654238 haze`.",
            parse_mode="Markdown")

    @dp.message(F.text)
    async def handle(m: types.Message):
        text = (m.text or "").strip()
        if text.startswith("/"):
            return
        parts = text.split(maxsplit=1)
        if not parts[0].isdigit():
            await m.answer("Нужен числовой Steam ID32, например 291654238.")
            return
        account_id = int(parts[0])

        now = time.time()
        uid = m.from_user.id if m.from_user else 0
        if now - _last_call.get(uid, 0) < USER_COOLDOWN:
            await m.answer("Подожди немного перед следующим запросом.")
            return
        _last_call[uid] = now

        heroes = await load_heroes()
        hero_id, hero_name = DEFAULT_HERO_ID, heroes.get("by_id", {}).get(DEFAULT_HERO_ID, "Ivy")
        if len(parts) > 1:
            q = parts[1].strip().lower()
            if q in heroes.get("by_name", {}):
                hero_id = heroes["by_name"][q]
                hero_name = heroes["by_id"][hero_id]
            else:
                await m.answer(f"Не знаю героя «{parts[1]}».")
                return

        await m.answer("Собираю данные, это может занять до минуты...")
        try:
            png, caption = await analyze(account_id, hero_id, hero_name)
        except Exception:
            log.exception("analyze failed")
            await m.answer("Что-то сломалось при расчёте. Попробуй позже.")
            return
        if png is None:
            await m.answer(caption)
            return
        await m.answer_photo(BufferedInputFile(png, filename="style.png"), caption=caption)

    return dp


async def ping(_request):
    return web.Response(text="ok")


async def main():
    logging.basicConfig(level=logging.INFO)
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("Задай переменную окружения BOT_TOKEN")
    bot = Bot(token=token)
    dp = make_dispatcher()

    app = web.Application()
    app.router.add_get("/", ping)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080))).start()

    try:
        await dp.start_polling(bot)
    finally:
        if _session and not _session.closed:
            await _session.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
