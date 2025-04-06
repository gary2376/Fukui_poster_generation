from flask import Flask, render_template, jsonify, Response, send_file, make_response, request
from deep_translator import GoogleTranslator
from rag_processor import process_query
import io
import pandas as pd
import os
import random
import urllib.parse
import json
from PIL import Image

from openai import OpenAI
import openai

openai.api_key = "sk-OuNkHG9dk1nZBFe4B7227789C9Fe4281B2886c14F63f1542"
openai.base_url = "https://free.v36.cm/v1/"

# 快取地點翻譯：中文 ➜ 英文
location_translation_cache = {}
app = Flask(__name__)

EXCEL_PATH = r'E:\python_project\contest\poster_fukui\code\full_function\data\location_record.xlsx'
IMAGE_BASE_PATH = r'E:\python_project\contest\poster_fukui\code\full_function\data\all_location_image'
TEMP_IMAGE_PATH = r'E:\python_project\contest\poster_fukui\code\full_function\data\temp'

# 確保 temp 目錄存在
if not os.path.exists(TEMP_IMAGE_PATH):
    os.makedirs(TEMP_IMAGE_PATH)

# ✅ GPT Prompt 精練函式
def smart_prompt_refiner(locations, keywords, adjectives, season=None, max_tokens=80):
    location_text = locations[0] if len(locations) == 1 else f"{locations[0]} and {locations[1]}"
    
    system_prompt = (
        "Write a fluent image prompt (≤77 tokens) that:\n"
        "- Mentions both locations\n"
        "- Uses most or all provided keywords and adjectives as-is\n"
        "- Includes vivid landscape elements (e.g. nature, light, weather)\n"
        "- Clearly mention the given season in the scene\n"
        "Keep it realistic, visual, and poetic."
    )

    user_prompt = (
        f"Locations: {location_text}\n"
        f"Season: {season}\n"
        f"Keywords: {', '.join(keywords)}\n"
        f"Adjectives: {', '.join(adjectives)}"
    )

    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.6,
    )

    return completion.choices[0].message.content.strip()

def jsonify_utf8(data):
    return Response(json.dumps(data, ensure_ascii=False), content_type="application/json; charset=utf-8")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cities')
def get_cities():
    try:
        df = pd.read_excel(EXCEL_PATH, dtype=str)
        df['city'] = df['city'].str.strip()
        cities = df['city'].dropna().unique().tolist()
        return jsonify_utf8(cities)
    except Exception as e:
        print(f"讀取城市時發生錯誤: {e}")
        return jsonify_utf8([])

@app.route('/locations/<city>')
def get_locations(city):
    try:
        city = urllib.parse.unquote(city, encoding='utf-8')
        df = pd.read_excel(EXCEL_PATH, dtype=str)
        df['city'] = df['city'].str.strip()
        df['location'] = df['location'].str.strip()
        city_locations = df[df['city'] == city]['location'].dropna().tolist()
        return jsonify_utf8(city_locations)
    except Exception as e:
        print(f"讀取 {city} 的地點時發生錯誤: {e}")
        return jsonify_utf8([])

@app.route('/place-images/<city>/<locations>')
def place_images(city, locations):
    try:
        city = urllib.parse.unquote(city, encoding='utf-8')
        location_list = [urllib.parse.unquote(loc, encoding='utf-8') for loc in locations.split(',')]

        location_list = location_list[:2]  # ✅ 限制最多選 2 個地點

        # ✅ 定義 3x2 切割區域，分為「左側」和「右側」
        left_positions = [(0, 0), (317, 0), (634, 0)]  # 左側
        right_positions = [(0, 352), (317, 352), (634, 352)]  # 右側

        # ✅ 提高圖片出現在下方四格的機率
        weighted_positions = [
            (0, 0), (0, 352),  # 第一排（20%）
            (317, 0), (317, 352), (317, 0), (317, 352),  # 第二排（40%）
            (634, 0), (634, 352), (634, 0), (634, 352)  # 第三排（40%）
        ]

        while True:
            left_pos = random.choice([pos for pos in weighted_positions if pos in left_positions])
            right_pos = random.choice([pos for pos in weighted_positions if pos in right_positions])

            # ✅ 確保不是同一列，避免並排
            if left_pos[0] != right_pos[0]:
                break

        selected_positions = [left_pos, right_pos]

        base_image = Image.new("RGBA", (704, 952), (0, 0, 0, 0))  # ✅ 透明背景

        for location, (top, left) in zip(location_list, selected_positions):
            location_dir = os.path.join(IMAGE_BASE_PATH, city, location)

            if not os.path.exists(location_dir):
                continue

            image_files = [f for f in os.listdir(location_dir) if f.endswith(".png")]

            if image_files:
                random_image = random.choice(image_files)
                image_path = os.path.join(location_dir, random_image)
                img = Image.open(image_path).convert("RGBA")

                # ✅ 放大圖片：最大寬度為格子寬的 120%，但不能超過格子實際大小
                max_width = int(352 * 1.2)
                max_width = min(max_width, 352)  # 不能超出格子寬度

                # ✅ 根據圖片原始比例縮放
                aspect_ratio = img.height / img.width
                target_width = max_width
                target_height = int(target_width * aspect_ratio)

                # ✅ 高度也不能超過格子高
                if target_height > 317:
                    target_height = 317
                    target_width = int(target_height / aspect_ratio)

                img = img.resize((target_width, target_height), Image.LANCZOS)

                # ✅ 計算圖片在格子內的偏移（確保完全不超界）
                max_offset_x = 352 - target_width
                max_offset_y = 317 - target_height

                offset_x = random.randint(0, max_offset_x)
                offset_y = random.randint(0, max_offset_y)

                final_x = left + offset_x
                final_y = top + offset_y

                # ✅ 貼上圖片
                base_image.paste(img, (final_x, final_y), img)

        # ✅ 儲存合成圖片
        combined_image_path = os.path.join(TEMP_IMAGE_PATH, "combined.png")
        base_image.save(combined_image_path, "PNG")

        return jsonify_utf8({"image_url": "/temp-image"})
    except Exception as e:
        print(f"處理 {city} 的圖片時發生錯誤: {e}")
        return jsonify_utf8([])

@app.route('/temp-image')
def get_temp_image():
    image_path = os.path.join(TEMP_IMAGE_PATH, "combined.png")
    
    response = make_response(send_file(image_path, mimetype='image/png'))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response

@app.route('/generate-prompt', methods=['POST'])
def generate_prompt():
    try:
        data = request.json
        locations_zh = data.get('locations', [])

        translator = GoogleTranslator(source='zh-TW', target='en')
        locations_en = []
        for loc in locations_zh:
            if loc in location_translation_cache:
                translated = location_translation_cache[loc]
            else:
                translated = translator.translate(loc)
                location_translation_cache[loc] = translated
            locations_en.append(translated)

        results = [process_query(loc) for loc in locations_en]

        def unique_preserve_order(seq):
            seen = set()
            return [x for x in seq if not (x in seen or seen.add(x))]

        all_keywords = []
        all_adjectives = []
        for res in results:
            all_keywords.extend(res["keywords"])
            all_adjectives.extend(res["adjectives"])

        # 去除重複
        keywords = unique_preserve_order(all_keywords)
        adjectives = unique_preserve_order(all_adjectives)

        # ✅ 使用 GPT 精練 prompt
        season = random.choice(["early spring", "mid-summer", "late autumn", "a snowy winter afternoon"])
        final_prompt = smart_prompt_refiner(locations_en, keywords, adjectives, season)


        negative_prompt = (
            f"{', '.join(locations_en)}, extra buildings, futuristic elements, harsh lighting, "
            f"overexposed sky, pixelation, distortion, low detail, bad reflections."
        )

        return jsonify_utf8({
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "locations_en": locations_en
        })

    except Exception as e:
        print(f"生成 prompt 時發生錯誤: {e}")
        return jsonify_utf8({"error": str(e)})

from poster_generator import generate_final_poster_image

@app.route('/generate-poster', methods=['POST'])
def generate_poster():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')

        final_image = generate_final_poster_image(prompt, negative_prompt)

        output_path = os.path.join(TEMP_IMAGE_PATH, 'poster_result.png')
        final_image.save(output_path)

        return jsonify_utf8({"image_url": "/poster-result"})
    except Exception as e:
        print("生成圖片時出錯：", e)
        return jsonify_utf8({"error": str(e)})
    
@app.route('/poster-result')
def get_generated_poster():
    image_path = os.path.join(TEMP_IMAGE_PATH, 'poster_result.png')

    if not os.path.exists(image_path):
        return "Image not found", 404

    response = make_response(send_file(image_path, mimetype='image/png'))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/translate-to-english', methods=['POST'])
def translate_to_english():
    try:
        data = request.json
        original_prompt = data.get("text", "")

        # 判斷是否含有中文（簡單判斷 Unicode 區間）
        if any('\u4e00' <= ch <= '\u9fff' for ch in original_prompt):
            translator = GoogleTranslator(source='auto', target='en')
            translated = translator.translate(original_prompt)
        else:
            translated = original_prompt  # 已經是英文

        return jsonify_utf8({"translated": translated})
    except Exception as e:
        print("翻譯發生錯誤：", e)
        return jsonify_utf8({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)
