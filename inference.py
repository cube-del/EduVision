import math
import re
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# --- НАСТРОЙКИ ---
MODEL_PATH = './qolda_model_local' # Убедись, что путь правильный
IMAGE_PATH = 'test.jpeg'
# -----------------

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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def clean_output(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()

if __name__ == '__main__':
    # Проверка доступности GPU
    if not torch.cuda.is_available():
        print("ВНИМАНИЕ: CUDA (GPU) не найдена! Код упадет.")
    else:
        print(f"Используется GPU: {torch.cuda.get_device_name(0)}")

    print(f"Загрузка модели на GPU...")
    
    # 1. ЗАГРУЗКА НА CUDA
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,  # Используем bfloat16 для скорости (если карта старая, замени на float16)
        low_cpu_mem_usage=True,
        use_flash_attn=False,        # Оставляем False, чтобы не было ошибки библиотеки
        trust_remote_code=True,
        device_map="cuda"            # Явно указываем видеокарту
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)

    try:
        # 2. ПОДГОТОВКА КАРТИНКИ (ОТПРАВЛЯЕМ НА .cuda())
        pixel_values = load_image(IMAGE_PATH, max_num=12).to(torch.bfloat16).cuda()
        
        generation_config = dict(max_new_tokens=1024, do_sample=False) # do_sample=False для OCR лучше (точнее)

        # 3. ПРОМПТ
        question = '<image>\nРаспознай весь рукописный текст на этом изображении. Выведи только сам текст.'
        
        print(f"Анализирую изображение...")
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        
        # 4. ВЫВОД
        final_text = clean_output(response)
        
        print("-" * 30)
        print("РЕЗУЛЬТАТ (GPU):")
        print(final_text)
        print("-" * 30)
        
    except FileNotFoundError:
        print(f"Ошибка: Файл {IMAGE_PATH} не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")