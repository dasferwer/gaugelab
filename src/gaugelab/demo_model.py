"""Предсказуемый эмулятор модели для демонстрации и проверок без платных API."""

import asyncio
import json

from fastapi import FastAPI
from pydantic import BaseModel, Field

ANSWERS = {
    "Столица Франции?": "Париж",
    "Столица Италии?": "Рим",
    "Столица Японии?": "Токио",
    "2 + 2?": "4",
    "3 * 3?": "9",
    "5 - 2?": "3",
    "Цвет травы?": "Зелёный",
    "Протокол защищённого HTTP?": "HTTPS",
    "База данных этого проекта?": "PostgreSQL",
    "Язык этого проекта?": "Python",
}
app = FastAPI(title="GaugeLab: эмулятор модели")


class Input(BaseModel):
    model: str
    prompt: str = Field(max_length=50000)
    seed: int
    temperature: float
    max_tokens: int


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(body: Input):
    prompt = body.prompt
    if "DEMO_SLOW" in prompt:
        await asyncio.sleep(2)
    if "DEMO_DELAY" in prompt:
        await asyncio.sleep(60)
    if prompt.startswith("GAUGELAB_JUDGE_V1"):
        data = json.loads(prompt.split("\nDATA=", 1)[1])
        output = json.dumps(
            {
                "score": float(data["expected"].casefold() == data["answer"].casefold()),
                "reason": "Сравнение с эталоном в локальном эмуляторе",
            },
            ensure_ascii=False,
        )
    elif "DEMO_WRONG" in prompt:
        output = "Не знаю"
    else:
        question = next((q for q in ANSWERS if q in prompt), None)
        output = ANSWERS.get(question, "Неизвестный вопрос")
        if "DEMO_NOISY" in prompt and body.seed % 2:
            output = "Не знаю"
        if "DEMO_VERBOSE" in prompt:
            output = "Подробный ответ: " + output
    # Это условные единицы эмулятора, а не токенизация реальной языковой модели.
    return {
        "model_digest": "demo-v1",
        "text": output,
        "input_tokens": len(prompt.split()),
        "output_tokens": len(output.split()),
    }
