import os
import io
import asyncio
import logging
import numpy as np
import pandas as pd
import aiohttp
from aiohttp import web

# Фикс для работы без графического экрана
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import BufferedInputFile

BOT_TOKEN = os.getenv("BOT_TOKEN", "8642003762:AAE4MRVSE_Fj2gefeMLl0NXMjsjtYi1iS9Q")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

DEADLOCK_API_BASE = "https://api.deadlock-api.com/v1"

async def fetch_player_matches(account_id: int, limit: int = 100):
    url = f"{DEADLOCK_API_BASE}/players/{account_id}/match-history"
    params = {"limit": limit}
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return None
            return await resp.json()

def get_stat_value(match_obj: dict, keys_preference: list, keywords_fallback: list, default=0):
    """Универсальный парсер метрик: поиск по приоритетным названиям и подстрокам."""
    if not isinstance(match_obj, dict):
        return default

    containers = [match_obj]
    for sub in ["player_stats", "stats", "player", "hero_stats"]:
        if sub in match_obj and isinstance(match_obj[sub], dict):
            containers.append(match_obj[sub])

    # 1. Точная проверка по предпочтительным названиям ключей
    for c in containers:
        for k in keys_preference:
            if k in c and c[k] is not None:
                try:
                    return float(c[k])
                except (ValueError, TypeError):
                    pass

    # 2. Поиск по частичному совпадению ключевого слова в названии поля
    for c in containers:
        for key_name, val in c.items():
            if val is None:
                continue
            key_lower = str(key_name).lower()
            for kw in keywords_fallback:
                if kw.lower() in key_lower:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass

    return float(default)

def calculate_player_features(matches_data: list, target_hero_id: int = 20):
    ivy_matches = [m for m in matches_data if m.get("hero_id") == target_hero_id]
    
    if not ivy_matches:
        return None

    net_worths, total_dmgs, weapon_dmgs = [], [], []
    heals, obj_dmgs, kills, assists, deaths = [], [], [], [], []

    for m in ivy_matches:
        nw = get_stat_value(
            m,
            keys_preference=["net_worth", "networth", "souls", "total_souls", "gold"],
            keywords_fallback=["worth", "soul", "gold"]
        )
        net_worths.append(nw)

        td = get_stat_value(
            m,
            keys_preference=["player_damage", "hero_damage", "damage_dealt", "damage_to_players", "damage"],
            keywords_fallback=["damage", "dmg"]
        )
        total_dmgs.append(td)

        wd = get_stat_value(
            m,
            keys_preference=["weapon_damage", "bullet_damage", "gun_damage"],
            keywords_fallback=["weapon", "bullet", "gun"]
        )
        weapon_dmgs.append(wd)

        hl = get_stat_value(
            m,
            keys_preference=["heal_amount", "healing", "heals", "total_healing"],
            keywords_fallback=["heal"]
        )
        heals.append(hl)

        od = get_stat_value(
            m,
            keys_preference=["objective_damage", "structure_damage", "tower_damage", "boss_damage"],
            keywords_fallback=["obj", "struct", "tower"]
        )
        obj_dmgs.append(od)

        k = get_stat_value(
            m,
            keys_preference=["kills", "player_kills", "num_kills"],
            keywords_fallback=["kill"]
        )
        kills.append(k)

        a = get_stat_value(
            m,
            keys_preference=["assists", "player_assists", "num_assists"],
            keywords_fallback=["assist"]
        )
        assists.append(a)

        d = get_stat_value(
            m,
            keys_preference=["deaths", "player_deaths", "num_deaths"],
            keywords_fallback=["death"]
        )
        deaths.append(d)

    avg_total_dmg = float(np.mean(total_dmgs)) if total_dmgs else 0.0
    avg_net_worth = float(np.mean(net_worths)) if net_worths else 0.0
    avg_kills = float(np.mean(kills)) if kills else 0.0
    avg_assists = float(np.mean(assists)) if assists else 0.0
    avg_deaths = float(np.mean(deaths)) if deaths else 0.0

    denom_dmg = avg_total_dmg if avg_total_dmg > 0 else 1.0
    denom_nw = avg_net_worth if avg_net_worth > 0 else 1.0

    weapon_ratio = (float(np.mean(weapon_dmgs)) if weapon_dmgs else 0.0) / denom_dmg
    heal_ratio = (float(np.mean(heals)) if heals else 0.0) / denom_nw
    obj_ratio = (float(np.mean(obj_dmgs)) if obj_dmgs else 0.0) / (denom_dmg + 1.0)
    assist_ratio = avg_assists / (avg_kills + avg_assists + 1.0)

    kda = (avg_kills + avg_assists) / (avg_deaths if avg_deaths > 0 else 1.0)

    return {
        "games_played": len(ivy_matches),
        "style_vector": [weapon_ratio, heal_ratio, obj_ratio, assist_ratio],
        "kda": kda,
        "avg_net_worth": avg_net_worth,
        "avg_dmg": avg_total_dmg
    }

def generate_style_chart(user_stats: dict, reference_df: pd.DataFrame) -> io.BytesIO:
    all_style_vectors = np.vstack([
        reference_df[['weapon_ratio', 'heal_ratio', 'obj_ratio', 'assist_ratio']].values,
        user_stats['style_vector']
    ])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_style_vectors)

    pca = PCA(n_components=1)
    x_coords = pca.fit_transform(scaled_data).flatten()

    ref_x = x_coords[:-1]
    user_x = x_coords[-1]

    ref_y = reference_df['impact_z'].values
    user_y = (user_stats['kda'] - reference_df['kda'].mean()) / (reference_df['kda'].std() + 1e-6)

    plt.figure(figsize=(10, 6), facecolor='#1e1e2e')
    ax = plt.axes()
    ax.set_facecolor('#181825')

    plt.scatter(ref_x, ref_y, color='#89b4fa', alpha=0.5, s=40, label='База игроков (Ivy)')
    plt.scatter(user_x, user_y, color='#e64553', s=200, marker='P', label='Вы (Текущий профиль)', zorder=5)

    plt.axhline(0, color='#585b70', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(0, color='#585b70', linestyle='--', linewidth=1, alpha=0.5)

    plt.title('Ваш стиль игры на Ivy (Deadlock)', color='white', fontsize=14, pad=12)
    plt.xlabel('← Саппорт / Утилити       Стиль (Ось X)       Кэрри / Огнестрел →', color='#cdd6f4', fontsize=10)
    plt.ylabel('← Ниже среднего       Импакт (Ось Y)       Выше среднего →', color='#cdd6f4', fontsize=10)

    plt.tick_params(colors='#a6adc8')
    plt.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='white')
    plt.grid(True, color='#313244', linestyle=':', alpha=0.4)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    plt.close()
    return buf

ref_data = {
    'weapon_ratio': np.random.uniform(0.3, 0.8, 100),
    'heal_ratio': np.random.uniform(0.05, 0.5, 100),
    'obj_ratio': np.random.uniform(0.1, 0.6, 100),
    'assist_ratio': np.random.uniform(0.2, 0.7, 100),
    'kda': np.random.normal(2.5, 0.8, 100),
    'impact_z': np.random.normal(0, 1, 100)
}
reference_df = pd.DataFrame(ref_data)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "👋 Привет! Отправь мне свой **Steam ID32** (например: `291654238`), "
        "чтобы получить 2D-карту твоего стиля игры на **Ivy**."
    )

@dp.message(F.text)
async def process_text_message(message: types.Message):
    if message.text.startswith('/'):
        return

    cleaned_text = message.text.strip()

    if not cleaned_text.isdigit():
        await message.answer("⚠️ Пожалуйста, отправьте только числовой Steam ID32 (например: `291654238`).")
        return

    account_id = int(cleaned_text)
    await message.answer("🔍 Запрашиваю данные с сервера Deadlock API...")

    matches = await fetch_player_matches(account_id)

    if not matches:
        await message.answer("❌ Не удалось найти данные по этому Steam ID или сервер недоступен.")
        return

    stats = calculate_player_features(matches, target_hero_id=20)

    if not stats:
        await message.answer("⚠️ В последних матчах не найдено сыгранных игр на **Ivy**.")
        return

    await message.answer("🎨 Генерирую карту стиля...")

    img_buf = generate_style_chart(stats, reference_df)

    caption = (
        f"📊 **Анализ стиля на Ivy** ({stats['games_played']} матчей)\n\n"
        f"• **KDA**: `{stats['kda']:.2f}`\n"
        f"• **Средний фарм**: `{int(stats['avg_net_worth'])}` душ\n"
        f"• **Средний урон**: `{int(stats['avg_dmg'])}`"
    )

    photo = BufferedInputFile(img_buf.read(), filename="ivy_style.png")
    await message.answer_photo(photo=photo, caption=caption, parse_mode="Markdown")

async def handle_ping(request):
    return web.Response(text="Bot status: OK")

async def main():
    logging.basicConfig(level=logging.INFO)
    
    app = web.Application()
    app.router.add_get('/', handle_ping)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
