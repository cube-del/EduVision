import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import re

# --- НАСТРОЙКИ ---
MODEL_PATH = './qolda_model_local'  # Путь к вашей модели
MAX_SLICES = 6                      # Баланс скорости и качества
USE_4BIT = True                     # 4-битное квантование для экономии памяти
# -----------------

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (из прошлого кода) ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def clean_output(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

# --- ЗАГРУЗКА МОДЕЛИ (ГЛОБАЛЬНО) ---
print("Загрузка модели в память... Пожалуйста, подождите.")
compute_dtype = torch.float16

quantization_config = None
if USE_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

try:
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype if not USE_4BIT else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Критическая ошибка при загрузке модели: {e}")
    exit(1)

# --- ФУНКЦИЯ ИНФЕРЕНСА ---
def predict(image):
    if image is None:
        return "Пожалуйста, загрузите изображение."
    
    try:
        # image приходит уже как PIL объект благодаря type='pil' в Gradio
        input_size = 448
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=MAX_SLICES)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        
        # Переносим на GPU и приводим к нужному типу
        pixel_values = pixel_values.to(compute_dtype).cuda()

        generation_config = dict(max_new_tokens=1024, do_sample=False)
        
        # --- БЫЛО ---
        # question = '<image>\nРаспознай весь рукописный текст на этом изображении. Выведи только сам текст.'

        # --- СТАЛО (Вставляем новый промпт) ---
        prompt_text = """Ты — строгий учитель русского языка. Твоя задача — точно перепечатать текст ученика с картинки для проверки.
        
        ВАЖНО:
        1. НЕ ИСПРАВЛЯЙ ОШИБКИ. Пиши в точности то, что видишь (даже если написано "малако", пиши "малако").
        2. Сохраняй пунктуацию автора (не добавляй запятые, если их нет).
        3. Выведи ТОЛЬКО распознанный текст без своих комментариев."""
        
        question = f'<image>\n{prompt_text}'

        with torch.no_grad():
            response = model.chat(tokenizer, pixel_values, question, generation_config)
        
        return clean_output(response)
    
    except Exception as e:
        return f"Ошибка при обработке: {str(e)}"

# --- ИНТЕРФЕЙС GRADIO ---
with gr.Blocks(title="Handwriting OCR") as demo:
    gr.Markdown("# 📝 Распознавание рукописного текста (Local OCR)")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Загрузите изображение")
            submit_btn = gr.Button("Распознать", variant="primary")
        
        with gr.Column():
            # Убрали show_copy_button, добавили interactive=False
            output_text = gr.Textbox(label="Результат", lines=15, interactive=False)

    submit_btn.click(
        fn=predict,
        inputs=[input_image],
        outputs=[output_text]
    )

    gr.Markdown("Работает локально на Qwen-VL/Qolda с 4-bit оптимизацией.")

if __name__ == "__main__":
    # share=True создаст публичную ссылку, если нужно
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)