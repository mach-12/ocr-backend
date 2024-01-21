import base64
import cv2
import numpy as np
import json
import re
from fastapi import FastAPI, HTTPException, Request

import easyocr

app = FastAPI()

reader = easyocr.Reader(['en'])

def run_model(image_data):
    ocr_result = reader.readtext(image_data)
    ocr_text = [i[1] for i in ocr_result]
    ocr_text_filtered = clean_and_convert(ocr_text)

    menu_data = [
        {
            "menu_item": [i for i in ocr_text_filtered],
            "menu_section": ["" for i in ocr_text_filtered]
        }
    ]

    return menu_data

def clean_and_convert(words):
    cleaned_words = []

    for word in words:
        # Remove symbols using regular expression
        cleaned_word = re.sub(r'[^a-zA-Z]', ' ', word)
        cleaned_word = cleaned_word.strip()

        # Convert to lowercase
        cleaned_word = cleaned_word.capitalize()

        # Append to the cleaned list
        if len(word) <= 3 or len(word) > 40:
            cleaned_words.append("")
            continue

        if re.search(r'\b{}\b'.format(re.escape(word)), "rs.", flags=re.IGNORECASE):
            cleaned_words.append("")
            continue

        cleaned_words.append(cleaned_word)

    cleaned_words = [i for i in cleaned_words if len(i) > 0]
    return cleaned_words

@app.post("/menu_ocr_trigger")
async def menu_ocr_trigger(request: Request):
    img_base64 = None

    try:
        req_body = await request.json()
        img_base64 = req_body.get('img')
    except ValueError:
        pass

    if img_base64:
        img = base64.b64decode(img_base64)
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        model_result = run_model(img)
        return model_result
    else:
        raise HTTPException(status_code=400, detail="Invalid request. Pass 'img' in the request body.")
