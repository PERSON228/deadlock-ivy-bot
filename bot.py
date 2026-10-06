"""Deadlock Style Map bot: карта стиля игры по Steam ID32 на выбранном герое.
Запрос: <steam id32> [герой], например 291654238 или 291654238 haze.
Env: BOT_TOKEN (обязательно), PORT, REF_ACCOUNT_IDS, PRO_PLAYERS="Ник:id,Ник:id".
Про-игроки: pros.json рядом с файлом, {"Ник": 123456789}."""
import os, io, json, asyncio, logging, time
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

log = logging.getLogger("dl")
API = "https://api.deadlock-api.com/v1"
ASSETS = "https://assets.deadlock-api.com/v2"
IVY = 20
REGIONS = ["Europe", "Asia", "NAmerica", "SAmerica", "Oceania"]
TOP_PER_REGION, TOP_N = 10, 30
MIN_REF, MIN_USER, MIN_PRO, MIN_WR = 10, 3, 5, 50
TTL, COOLDOWN = 6 * 3600, 20

LABELS = {
    "hero_bullets_hit_per_min": "попадания по героям/мин",
    "hero_bullets_hit_crit_per_min": "криты по героям/мин",
    "enemy_bullets_hit_per_min": "попадания по героям/мин",
    "enemy_bullets_hit_crit_per_min": "криты по героям/мин",
    "creep_kills_per_min": "крипы/мин", "creeps_per_min": "крипы/мин",
    "neutral_damage_per_min": "урон по нейтралам/мин",
    "boss_damage_per_min": "урон по боссам/мин",
    "shots_hit_per_min": "попадания/мин", "shots_missed_per_min": "промахи/мин",
    "networth_per_min": "души/мин", "net_worth_per_min": "души/мин",
    "last_hits_per_min": "добивания/мин",
}
WORDS = {"damage": "урон", "taken": "полученный", "mitigated": "поглощённый",
         "creeps": "крипы", "kills": "убийства", "deaths": "смерти", "assists": "ассисты",
         "denies": "дени", "networth": "души", "worth": "души", "souls": "души",
         "healing": "лечение", "heal": "лечение", "accuracy": "точность",
         "last": "добивания", "net": "", "hits": "", "bullets": ""}
SKIP = {"account_id", "hero_id", "wins", "losses", "matches", "matches_played",
        "winrate", "win_rate", "ending_level", "max_health", "duration_s"}

# ---------------------------------------------------------------- API
_s = None
_sem = asyncio.Semaphore(4)


async def get_json(url, params=None, tries=3):
    global _s
    for i in range(tries):
        try:
            async with _sem:
                if _s is None or _s.closed:
                    _s = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40))
                async with _s.get(url, params=params) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    if r.status not in (429, 502, 503, 504):
                        log.warning("GET %s -> %s", url, r.status)
                        return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("GET %s: %s", url, e)
        await asyncio.sleep(1.5 * (i + 1))
    return None


_heroes_cache = {}


async def load_heroes():
    if _heroes_cache:
        return _heroes_cache
    by_name, by_id = {}, {}
    for h in await get_json(f"{ASSETS}/heroes") or []:
        try:
            hid, name = int(h["id"]), str(h.get("name") or h["class_name"]).strip()
        except (TypeError, KeyError, ValueError, AttributeError):
            continue
        by_id[hid] = name
        by_name[name.lower()] = hid
        cn = str(h.get("class_name", "")).lower().replace("hero_", "")
        if cn:
            by_name[cn] = hid
    if not by_id:
        return {"by_name": {"ivy": IVY}, "by_id": {IVY: "Ivy"}}
    _heroes_cache.update(by_name=by_name, by_id=by_id)
    return _heroes_cache


async def resolve_hero(q):
    """(hero_id, имя) или None. Принимает имя, начало имени или числовой ID."""
    h = await load_heroes()
    q = (q or "").strip().lower()
    if not q:
        return IVY, h["by_id"].get(IVY, "Ivy")
    if q.isdigit():
        return int(q), h["by_id"].get(int(q), f"герой {q}")
    hid = h["by_name"].get(q)
    if hid is None:
        c = {v for n, v in h["by_name"].items() if n.startswith(q)}
        hid = c.pop() if len(c) == 1 else None
    return (hid, h["by_id"][hid]) if hid is not None else None


async def hero_stats(ids, hero_id):
    """Агрегаты игрок+герой, bulk до 1000 аккаунтов за запрос."""
    out = []
    for i in range(0, len(ids), 1000):
        d = await get_json(f"{API}/players/hero-stats", {
            "account_ids": ",".join(map(str, ids[i:i + 1000])), "hero_ids": str(hero_id)})
        out += [x for x in d if isinstance(x, dict)] if isinstance(d, list) else []
    return out


def _aid(r):
    try:
        return int(r.get("account_id"))
    except (TypeError, ValueError):
        return None


def _ids(data):
    if isinstance(data, dict):
        data = data.get("data") or data.get("results") or data.get("players")
    return [a for a in (_aid(d) for d in data if isinstance(d, dict))
            if a is not None] if isinstance(data, list) else []


async def candidate_accounts(hero_id):
    env = os.getenv("REF_ACCOUNT_IDS", "")
    if env.strip():
        return [int(x) for x in env.split(",") if x.strip().isdigit()]
    base = {"hero_id": hero_id, "sort_by": "matches", "limit": 1000}
    for url, extra in ((f"{API}/analytics/scoreboards/players", {"sort_direction": "desc"}),
                       (f"{API}/analytics/scoreboards/players", {}),
                       (f"{API}/players/scoreboard", {})):
        ids = _ids(await get_json(url, base | extra, tries=1))
        if ids:
            return list(dict.fromkeys(ids))
    return []


def matches_of(r):
    for k in ("matches_played", "matches"):
        v = r.get(k)
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, list):
            return len(v)
    return 0


_ref, _top = {}, {}


async def reference_rows(hero_id):
    c = _ref.get(hero_id)
    if c and time.time() - c[0] < TTL:
        return c[1]
    ids = await candidate_accounts(hero_id)
    rows = [r for r in (await hero_stats(ids, hero_id) if ids else []) if matches_of(r) >= MIN_REF]
    if rows:
        _ref[hero_id] = (time.time(), rows)
    log.info("reference hero %s: %d players", hero_id, len(rows))
    return rows


async def leaderboard_top(hero_id):
    """Топ лидерборда героя по регионам: (список account_id, {id: ник})."""
    async def one(region):
        d = await get_json(f"{API}/leaderboard/{region}/{hero_id}", tries=1)
        es = d.get("entries") if isinstance(d, dict) else d
        out = []
        for e in (es[:TOP_PER_REGION] if isinstance(es, list) else []):
            try:
                ids = e.get("possible_account_ids") or [e["account_id"]]
                out.append((int(ids[0]), str(e.get("account_name") or "")))
            except (TypeError, KeyError, ValueError, IndexError, AttributeError):
                pass
        return out
    names = {}
    for part in await asyncio.gather(*(one(r) for r in REGIONS)):
        for a, n in part:
            names.setdefault(a, n)
    return list(names), names


async def top_rows(hero_id):
    c = _top.get(hero_id)
    if c and time.time() - c[0] < TTL:
        return c[1]
    ids, names = await leaderboard_top(hero_id)
    rows = [r for r in (await hero_stats(ids, hero_id) if ids else []) if matches_of(r) >= MIN_USER]
    if rows:
        _top[hero_id] = (time.time(), (rows, names))
    return rows, names


def load_pros():
    pros = {}
    for item in os.getenv("PRO_PLAYERS", "").split(","):
        name, _, acc = item.rpartition(":")
        if name.strip() and acc.strip().isdigit():
            pros[name.strip()] = int(acc)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pros.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                pros.update({str(k): int(v) for k, v in json.load(f).items() if str(v).isdigit()})
        except Exception:
            log.exception("pros.json не прочитан")
    return pros


# ---------------------------------------------------------------- анализ
def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def matrix(rows, keys):
    return np.array([[_num(r.get(k)) for k in keys] for r in rows], dtype=float)


def style_keys(rows):
    """Признаки стиля: только нормированные показатели (в минуту, на душу, точность)."""
    if not rows:
        return []
    keys = [k for k, v in rows[0].items() if k not in SKIP and isinstance(v, (int, float))
            and not isinstance(v, bool) and ("per_min" in k or "per_soul" in k or k == "accuracy")]
    good = []
    for k in keys:
        col = np.array([_num(r.get(k)) for r in rows])
        if np.isfinite(col).mean() > 0.9 and np.nanstd(col) > 1e-9:
            good.append(k)
    return good


def pretty(k):
    if k in LABELS:
        return LABELS[k]
    left, _, right = k.partition("_per_")
    words = [WORDS.get(w, w) for w in left.split("_")]
    words.sort(key=lambda w: w not in ("полученный", "поглощённый"))
    name = " ".join(w for w in words if w) or left
    return {"min": f"{name}/мин", "soul": f"{name} на душу", "": name}.get(right, f"{name} на {right}")


def pick_k(X):
    best_k, best_s = 3, -1.0
    for k in range(3, min(7, len(X) // 8 + 1)):
        try:
            s = silhouette_score(X, KMeans(k, n_init=5, random_state=0).fit_predict(X))
        except Exception:
            continue
        if s > best_s:
            best_k, best_s = k, s
    return best_k


def build_map(ref_rows, user_row, keys, groups=None):
    """Расчёт карты без сети. groups: [{"label","rows","names","style": "top"|"pro"}].
    Группы и пользователь проецируются на оси, обученные только на базе."""
    R = matrix(ref_rows, keys)
    med = np.nanmedian(R, axis=0)
    R = np.where(np.isnan(R), med, R)
    R = np.clip(R, np.percentile(R, 1, axis=0), np.percentile(R, 99, axis=0))
    sc = StandardScaler().fit(R)
    Rz = sc.transform(R)

    def to_z(rows):  # без обрезки: про и вы могут быть экстремальнее базы
        M = matrix(rows, keys)
        return sc.transform(np.where(np.isnan(M), med, M))

    pca = PCA(n_components=2, random_state=0).fit(Rz)
    nc = min(6, Rz.shape[1], Rz.shape[0] - 1)  # кластеры считаем не в 2D, а в 6 компонентах
    pc = PCA(n_components=nc, random_state=0).fit(Rz)
    k = pick_k(pc.transform(Rz))
    km = KMeans(k, n_init=10, random_state=0).fit(pc.transform(Rz))
    labels = km.labels_

    def axis(comp):
        o = np.argsort(comp)
        return (" + ".join(pretty(keys[i]) for i in o[:2]),
                " + ".join(pretty(keys[i]) for i in o[::-1][:2]))

    res = {"ref2": pca.transform(Rz), "labels": labels, "k": k, "keys": keys,
           "evr": pca.explained_variance_ratio_, "xaxis": axis(pca.components_[0]),
           "yaxis": axis(pca.components_[1]), "groups": [], "nearest": {}}
    uz = None
    if user_row is not None:
        uz = to_z([user_row])
        res.update(user2=pca.transform(uz)[0], user_z=uz[0],
                   user_cluster=int(km.predict(pc.transform(uz))[0]),
                   percentile_rank=(Rz < uz).mean(axis=0))
    for g in groups or []:
        if not g["rows"]:
            continue
        gz = to_z(g["rows"])
        res["groups"].append({"label": g["label"], "style": g["style"],
                              "names": g.get("names"), "xy": pca.transform(gz)})
        if uz is not None and g.get("names"):
            near = []
            for i in np.argsort(np.linalg.norm(gz - uz, axis=1)):
                n = g["names"][i]
                if n and n not in near:
                    near.append(n)
                if len(near) == 3:
                    break
            if near:
                res["nearest"][g["label"]] = near
    desc = {}
    for c in range(k):
        mask = labels == c
        m = Rz[mask].mean(axis=0)
        top = np.argsort(-np.abs(m))[:2]
        desc[c] = ", ".join(f"{pretty(keys[i])} {'выше' if m[i] > 0 else 'ниже'} среднего"
                            for i in top) + f" ({mask.sum()} игр.)"
    res["cluster_desc"] = desc
    return res


PALETTE = ["#89b4fa", "#a6e3a1", "#f9e2af", "#cba6f7", "#94e2d5", "#fab387"]


def render_map(res, title, ref_n):
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#1e1e2e")
    ax.set_facecolor("#181825")
    R2 = res["ref2"]
    for c in range(res["k"]):
        m = res["labels"] == c
        ax.scatter(R2[m, 0], R2[m, 1], s=22, alpha=0.38, color=PALETTE[c % 6], label=f"кластер {c + 1}")
    for g in res["groups"]:
        xy, lab = g["xy"], f"{g['label']} ({len(g['xy'])})"
        if g["style"] == "pro":
            ax.scatter(xy[:, 0], xy[:, 1], s=120, marker="D", color="white",
                       edgecolor="#11111b", zorder=4, label=lab)
            for (x, y), n in zip(xy, g["names"] or []):
                ax.annotate(n, (x, y), xytext=(6, 6), textcoords="offset points", color="white",
                            fontsize=8, zorder=6, bbox=dict(boxstyle="round,pad=0.15",
                                                            fc="#11111b", ec="none", alpha=0.75))
        else:
            ax.scatter(xy[:, 0], xy[:, 1], s=85, facecolors="none", edgecolors="white",
                       linewidth=1.5, zorder=3, label=lab)
    if "user2" in res:
        ax.scatter(*res["user2"], s=340, marker="*", color="#f38ba8", edgecolor="white",
                   linewidth=1.2, zorder=7, label="Вы")
    ax.axhline(0, color="#585b70", ls="--", lw=1, alpha=0.5)
    ax.axvline(0, color="#585b70", ls="--", lw=1, alpha=0.5)
    ev, (xn, xp), (yn, yp) = res["evr"], res["xaxis"], res["yaxis"]
    ax.set_xlabel(f"← {xn}      ось 1 ({ev[0]:.0%})      {xp} →", color="#cdd6f4", fontsize=9)
    ax.set_ylabel(f"← {yn}      ось 2 ({ev[1]:.0%})      {yp} →", color="#cdd6f4", fontsize=9)
    ax.set_title(f"{title}  ·  база {ref_n} игроков", color="white", fontsize=14, pad=12)
    ax.tick_params(colors="#a6adc8")
    ax.grid(True, color="#313244", ls=":", alpha=0.5)
    for s in ax.spines.values():
        s.set_color("#45475a")
    ax.legend(facecolor="#313244", edgecolor="#45475a", labelcolor="white", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def describe_user(res, user_row):
    keys, z, pr, c = res["keys"], res["user_z"], res["percentile_rank"], res["user_cluster"]
    parts = [f"Ваш кластер {c + 1}: {res['cluster_desc'][c]}"]
    parts += [f"Ближе всего по стилю ({lab}): {', '.join(n)}" for lab, n in res["nearest"].items()]
    lines = [f"• {pretty(keys[i])}: {_num(user_row.get(keys[i])):.2f} (выше, чем у {pr[i]:.0%} игроков)"
             for i in np.argsort(-np.abs(z))[:4]]
    parts.append("Самое необычное в вашей игре:\n" + "\n".join(lines))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- бот
async def _none():
    return []


async def analyze(account_id, hero_id, hero_name):
    """(png | None, подпись)."""
    pros = load_pros()
    id2pro = {v: k for k, v in pros.items()}
    ref, mine, (top, tnames), prows = await asyncio.gather(
        reference_rows(hero_id), hero_stats([account_id], hero_id), top_rows(hero_id),
        hero_stats(list(pros.values()), hero_id) if pros else _none())
    u = mine[0] if mine else None
    if u is None:
        return None, f"Нет данных по этому ID на {hero_name}. Проверь Steam ID32 и что профиль не закрыт."
    if matches_of(u) < MIN_USER:
        return None, f"На {hero_name} у вас {matches_of(u)} матч(ей) в API, нужно хотя бы {MIN_USER}, лучше 20+."
    keys = style_keys(ref)
    if len(ref) < 40 or len(keys) < 3:
        return None, "Не удалось получить базу игроков или показатели для карты. Попробуй позже."

    groups, label = [], "Топ лидерборда"
    if not top:  # запасной топ: лучший винрейт в базе
        cand = []
        for r in ref:
            w, n = _num(r.get("wins")), matches_of(r)
            if np.isfinite(w) and n >= MIN_WR:
                cand.append((w / n, r))
        top = [r for _, r in sorted(cand, key=lambda t: t[0], reverse=True)[:TOP_N]]
        tnames, label = {}, "Топ по винрейту"
    if top:
        names = [tnames.get(_aid(r) or -1, "") for r in top]
        groups.append({"label": label, "rows": top, "style": "top", "names": names if any(names) else None})
    psel = [r for r in prows if _aid(r) in id2pro and matches_of(r) >= MIN_PRO]
    if psel:
        groups.append({"label": "Про", "rows": psel, "style": "pro", "names": [id2pro[_aid(r)] for r in psel]})

    res = await asyncio.to_thread(build_map, ref, u, keys, groups)
    png = await asyncio.to_thread(render_map, res, f"Стиль на {hero_name}", len(ref))
    cap = f"{hero_name}: {matches_of(u)} матчей\n\n" + describe_user(res, u)
    if pros and not psel:
        cap += "\n\nПро из pros.json не найдены на этом герое."
    return png, cap[:1020]


_last = {}


def make_dispatcher():
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def start(m: types.Message):
        await m.answer("Пришли Steam ID32, например 291654238, и я построю карту твоего стиля на Ivy.\n"
                       "Другой герой: ID и имя, например 291654238 haze.")

    @dp.message(F.text)
    async def handle(m: types.Message):
        text = (m.text or "").strip()
        if text.startswith("/"):
            return
        parts = text.split(maxsplit=1)
        if not parts or not parts[0].isdigit():
            await m.answer("Нужен числовой Steam ID32, например 291654238.")
            return
        uid, now = (m.from_user.id if m.from_user else 0), time.time()
        if now - _last.get(uid, 0) < COOLDOWN:
            await m.answer("Подожди немного перед следующим запросом.")
            return
        _last[uid] = now
        hero = await resolve_hero(parts[1] if len(parts) > 1 else None)
        if hero is None:
            await m.answer(f"Не нашёл героя «{parts[1]}». Напиши имя как в игре или числовой ID героя.")
            return
        await m.answer("Собираю данные, это может занять до минуты...")
        try:
            png, cap = await analyze(int(parts[0]), *hero)
        except Exception:
            log.exception("analyze failed")
            await m.answer("Что-то сломалось при расчёте. Попробуй позже.")
            return
        if png is None:
            await m.answer(cap)
        else:
            await m.answer_photo(BufferedInputFile(png, filename="style.png"), caption=cap)

    return dp


async def main():
    logging.basicConfig(level=logging.INFO)
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise SystemExit("Задай переменную окружения BOT_TOKEN")
    bot = Bot(token=token)
    app = web.Application()
    app.router.add_get("/", lambda _r: web.Response(text="ok"))
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", 8080))).start()
    try:
        await make_dispatcher().start_polling(bot)
    finally:
        if _s and not _s.closed:
            await _s.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
