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

def extract_stat(obj: dict, keys: list, default=0):
    """Точное извлечение значения по списку ключей API без нечеткого поиска."""
    for key in keys:
        if key in obj and obj[key] is not None:
            try:
                return float(obj[key])
            except (ValueError, TypeError):
                pass
    return float(default)

def calculate_player_features(matches_data: list, target_hero_id: int = 20):
    ivy_matches = [m for m in matches_data if m.get("hero_id") == target_hero_id]
    
    if not ivy_matches:
        return None

    # Логируем ключи первого матча в консоль Render для точной проверки структуры API
    sample_match = ivy_matches[0]
    stats_container = sample_match.get("player_stats") if isinstance(sample_match.get("player_stats"), dict) else sample_match
    logging.info(f"Deadlock API Match Keys: {list(stats_container.keys())}")

    net_worths, total_dmgs, weapon_dmgs = [], [], []
    heals, obj_dmgs, kills, assists, deaths = [], [], [], [], []

    for m in ivy_matches:
        st = m.get("player_stats") if isinstance(m.get("player_stats"), dict) else m

        # Урон по игрокам
        td = extract_stat(st, ["player_damage", "hero_damage", "damage_dealt_to_players", "damage_dealt", "damage"])
        total_dmgs.append(td)

        # Фарм / Души
        nw = extract_stat(st, ["net_worth", "souls", "total_souls", "gold"])
        net_worths.append(nw)

        # Урон оружием
        wd = extract_stat(st, ["weapon_damage", "bullet_damage", "gun_damage"])
        weapon_dmgs.append(wd)

        # Лечение
        hl = extract_stat(st, ["healing", "heal_amount", "heals", "total_healing"])
        heals.append(hl)

        # Урон по объектам
        od = extract_stat(st, ["objective_damage", "structure_damage", "tower_damage"])
        obj_dmgs.append(od)

        # KDA
        kills.append(extract_stat(st, ["kills", "num_kills"]))
        assists.append(extract_stat(st, ["assists", "num_assists"]))
        deaths.append(extract_stat(st, ["deaths", "num_deaths"]))

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
    
    # Ограничение координат в пределах видимой сетки
    user_x = np.clip(x_coords[-1], -2.5, 2.5)

    ref_y = reference_df['impact_z'].values
    calc_y = (user_stats['kda'] - reference_df['kda'].mean()) / (reference_df['kda'].std() + 1e-6)
    user_y = np.clip(calc_y, -2.5, 2.5)

    plt.figure(figsize=(10, 6), facecolor='#1e1e2e')
    ax = plt.axes()
    ax.set_facecolor('#181825')

    plt.scatter(ref_x, ref_y, color='#89b4fa', alpha=0.5, s=40, label='База игроков (Ivy)')
    plt.scatter(user_x, user_y, color='#e64553', s=200, marker='P', label='Вы (Текущий профиль)', zorder=5)

    plt.axhline(0, color='#585b70', linestyle='--', linewidth=1, alpha=0.5)
    plt.axvline(0, color='#585b70', linestyle='--', linewidth=1, alpha=0.5)

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

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
    '
