import os
import sys
import time
import random
from PIL import Image
import numpy as np
import cv2

def get_next_number(directory, prefix=""):
    """
    Определяет следующий доступный номер для именования файлов или папок в директории.
    
    Args:
        directory: Директория для проверки
        prefix: Префикс имени файла или папки
        
    Returns:
        Следующий доступный номер
    """
    if not os.path.exists(directory):
        return 1
        
    # Ищем все папки с числовыми именами
    items = [item for item in os.listdir(directory) 
             if os.path.isdir(os.path.join(directory, item)) and item.startswith(prefix)]
    
    if not items:
        return 1
        
    # Извлекаем числа из имен папок
    numbers = []
    for item in items:
        name = item.replace(prefix, "")
        try:
            numbers.append(int(name))
        except ValueError:
            continue
    
    # Возвращаем следующий номер
    return max(numbers) + 1 if numbers else 1

def crop_overlapping_images(image_path, output_dir="results", 
                           min_crop_percentage=0.2,
                           max_crop_percentage=0.4,
                           overlap_percentage=0.05,
                           max_rotation_angle=15):
    """
    Вырезает два фрагмента изображения с пересекающейся областью, 
    поворачивает их на случайный угол и сохраняет.
    
    Args:
        image_path: Путь к исходному изображению
        output_dir: Директория для сохранения результатов
        min_crop_percentage: Минимальный размер фрагмента относительно оригинала
        max_crop_percentage: Максимальный размер фрагмента относительно оригинала
        overlap_percentage: Процент перекрытия между фрагментами (0-1)
        max_rotation_angle: Максимальный угол поворота в градусах
        
    Returns:
        Кортеж из:
        - путь к первому фрагменту
        - путь ко второму фрагменту
        - словарь с информацией для склеивания
    """
    # Создаем директорию для результатов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Получаем следующий доступный номер для именования папки
        next_number = get_next_number(output_dir)
        
        # Создаем папку с номером для текущего результата
        result_dir = os.path.join(output_dir, str(next_number))
        os.makedirs(result_dir)
        
        # Открываем изображение
        img = Image.open(image_path)
        width, height = img.size
        
        # Генерируем случайные размеры для первого фрагмента
        first_crop_percentage = random.uniform(min_crop_percentage, max_crop_percentage)
        first_crop_width = int(width * first_crop_percentage)
        first_crop_height = int(height * first_crop_percentage)
        
        # Генерируем случайные размеры для второго фрагмента
        second_crop_percentage = random.uniform(min_crop_percentage, max_crop_percentage)
        second_crop_width = int(width * second_crop_percentage)
        second_crop_height = int(height * second_crop_percentage)
        
        # Выбираем случайные координаты для первого фрагмента
        first_left = random.randint(0, width - first_crop_width)
        first_top = random.randint(0, height - first_crop_height)
        first_right = first_left + first_crop_width
        first_bottom = first_top + first_crop_height
        
        # Вычисляем максимально возможное смещение для второго фрагмента,
        # чтобы обеспечить перекрытие
        max_left_shift = int(first_crop_width * (1 - overlap_percentage))
        max_top_shift = int(first_crop_height * (1 - overlap_percentage))
        
        # Вычисляем координаты для второго фрагмента, обеспечивая перекрытие
        second_left = max(0, min(width - second_crop_width, 
                                first_left - random.randint(0, max_left_shift)))
        second_top = max(0, min(height - second_crop_height, 
                               first_top - random.randint(0, max_top_shift)))
        second_right = second_left + second_crop_width
        second_bottom = second_top + second_crop_height
        
        # Проверяем, есть ли перекрытие
        has_overlap = (second_right > first_left and second_left < first_right and
                      second_bottom > first_top and second_top < first_bottom)
        
        if not has_overlap:
            # Если перекрытия нет, корректируем координаты второго фрагмента
            second_left = max(0, min(width - second_crop_width, first_left - int(second_crop_width * overlap_percentage)))
            second_top = max(0, min(height - second_crop_height, first_top - int(second_crop_height * overlap_percentage)))
            second_right = second_left + second_crop_width
            second_bottom = second_top + second_crop_height
        
        # Вычисляем координаты области пересечения в абсолютных координатах
        overlap_left = max(first_left, second_left)
        overlap_top = max(first_top, second_top)
        overlap_right = min(first_right, second_right)
        overlap_bottom = min(first_bottom, second_bottom)
        
        # Вычисляем координаты области пересечения относительно каждого фрагмента
        first_overlap_x = overlap_left - first_left
        first_overlap_y = overlap_top - first_top
        second_overlap_x = overlap_left - second_left
        second_overlap_y = overlap_top - second_top
        
        # Вычисляем размеры области пересечения
        overlap_width = overlap_right - overlap_left
        overlap_height = overlap_bottom - overlap_top
        
        # Вырезаем фрагменты
        first_cropped_img = img.crop((first_left, first_top, first_right, first_bottom))
        second_cropped_img = img.crop((second_left, second_top, second_right, second_bottom))
        
        # Генерируем случайные углы поворота
        first_rotation = random.uniform(-max_rotation_angle, max_rotation_angle)
        second_rotation = random.uniform(-max_rotation_angle, max_rotation_angle)
        
        # Вместо простого поворота с расширением, используем аффинную трансформацию
        # Преобразуем PIL изображения в массивы numpy для работы с OpenCV
        first_cv = np.array(first_cropped_img)
        second_cv = np.array(second_cropped_img)
        
        # Конвертируем в BGR для OpenCV, если изображения RGB
        if len(first_cv.shape) == 3 and first_cv.shape[2] == 3:
            first_cv = cv2.cvtColor(first_cv, cv2.COLOR_RGB2BGR)
            second_cv = cv2.cvtColor(second_cv, cv2.COLOR_RGB2BGR)
        
        # Получаем размеры изображений
        h1, w1 = first_cv.shape[:2]
        h2, w2 = second_cv.shape[:2]
        
        # Создаем матрицы поворота
        center1 = (w1 // 2, h1 // 2)
        center2 = (w2 // 2, h2 // 2)
        
        # Создаем матрицы поворота
        M1 = cv2.getRotationMatrix2D(center1, first_rotation, 1.0)
        M2 = cv2.getRotationMatrix2D(center2, second_rotation, 1.0)
        
        # Вычисляем новые размеры после поворота
        cos1 = abs(M1[0, 0])
        sin1 = abs(M1[0, 1])
        new_w1 = int((h1 * sin1) + (w1 * cos1))
        new_h1 = int((h1 * cos1) + (w1 * sin1))
        
        cos2 = abs(M2[0, 0])
        sin2 = abs(M2[0, 1])
        new_w2 = int((h2 * sin2) + (w2 * cos2))
        new_h2 = int((h2 * cos2) + (w2 * sin2))
        
        # Корректируем матрицы поворота
        M1[0, 2] += (new_w1 / 2) - center1[0]
        M1[1, 2] += (new_h1 / 2) - center1[1]
        
        M2[0, 2] += (new_w2 / 2) - center2[0]
        M2[1, 2] += (new_h2 / 2) - center2[1]
        
        # Применяем поворот
        first_rotated = cv2.warpAffine(first_cv, M1, (new_w1, new_h1), flags=cv2.INTER_LINEAR)
        second_rotated = cv2.warpAffine(second_cv, M2, (new_w2, new_h2), flags=cv2.INTER_LINEAR)
        
        # Конвертируем обратно в RGB для PIL
        if len(first_rotated.shape) == 3 and first_rotated.shape[2] == 3:
            first_rotated = cv2.cvtColor(first_rotated, cv2.COLOR_BGR2RGB)
            second_rotated = cv2.cvtColor(second_rotated, cv2.COLOR_BGR2RGB)
        
        # Преобразуем обратно в PIL изображения
        first_rotated_pil = Image.fromarray(first_rotated)
        second_rotated_pil = Image.fromarray(second_rotated)
        
        # Формируем имена выходных файлов
        _, ext = os.path.splitext(image_path)
        first_output_path = os.path.join(result_dir, f"1{ext}")
        second_output_path = os.path.join(result_dir, f"2{ext}")
        
        # Сохраняем результаты
        first_rotated_pil.save(first_output_path)
        second_rotated_pil.save(second_output_path)
        
        # Обновляем информацию о пересечении с учетом поворота
        # Для простоты мы сохраняем исходные координаты пересечения и углы поворота,
        # а в функции склейки будем учитывать эти углы
        overlap_info = {
            'first_image': (first_overlap_x, first_overlap_y),
            'second_image': (second_overlap_x, second_overlap_y),
            'width': overlap_width,
            'height': overlap_height,
            'first_rotation': first_rotation,
            'second_rotation': second_rotation,
            'first_original_size': (w1, h1),
            'second_original_size': (w2, h2),
            'first_rotated_size': (new_w1, new_h1),
            'second_rotated_size': (new_w2, new_h2)
        }
        
        print(f"Первый фрагмент сохранен в {first_output_path}")
        print(f"Второй фрагмент сохранен в {second_output_path}")
        print(f"Размер первого фрагмента: {first_crop_width}x{first_crop_height} ({first_crop_percentage:.2f}% от оригинала)")
        print(f"Размер второго фрагмента: {second_crop_width}x{second_crop_height} ({second_crop_percentage:.2f}% от оригинала)")
        print(f"Информация о пересечении:")
        print(f"  Начало пересечения в первом изображении: {overlap_info['first_image']}")
        print(f"  Начало пересечения во втором изображении: {overlap_info['second_image']}")
        print(f"  Размер области пересечения: {overlap_info['width']}x{overlap_info['height']}")
        print(f"  Угол поворота первого изображения: {overlap_info['first_rotation']:.2f}°")
        print(f"  Угол поворота второго изображения: {overlap_info['second_rotation']:.2f}°")
        
        return first_output_path, second_output_path, overlap_info
    
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def crop_image(image_path, output_dir="results", crop_percentage=0.1):
    """
    Вырезает случайный фрагмент изображения и сохраняет его в указанную директорию.
    
    Args:
        image_path: Путь к исходному изображению
        output_dir: Директория для сохранения результата
        crop_percentage: Размер вырезаемого фрагмента относительно оригинала (0.1 = 10%)
    """
    # Создаем директорию для результатов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Открываем изображение
        img = Image.open(image_path)
        width, height = img.size
        
        # Вычисляем размеры вырезаемого фрагмента (±10% от оригинала)
        crop_width = int(width * crop_percentage)
        crop_height = int(height * crop_percentage)
        
        # Выбираем случайные координаты для вырезания
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        
        # Вырезаем фрагмент
        cropped_img = img.crop((left, top, right, bottom))
        
        # Формируем имя выходного файла
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_cropped{ext}")
        
        # Сохраняем результат
        cropped_img.save(output_path)
        print(f"Изображение успешно обработано и сохранено в {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None

def main():
    # Проверяем, передан ли путь к изображению как аргумент командной строки
    if len(sys.argv) < 2:
        print("Использование: python main.py <путь_к_изображению>")
        return
    
    image_path = sys.argv[1]
    
    # Проверяем существование файла
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден")
        return
    
    # Вырезаем и сохраняем два перекрывающихся фрагмента изображения
    first_path, second_path, overlap_info = crop_overlapping_images(image_path)
    
    if overlap_info:
        print("\nДля склеивания изображений обратно используйте следующие координаты:")
        print(f"Первое изображение: начало пересечения в ({overlap_info['first_image'][0]}, {overlap_info['first_image'][1]})")
        print(f"Второе изображение: начало пересечения в ({overlap_info['second_image'][0]}, {overlap_info['second_image'][1]})")
        print(f"Размер области пересечения: {overlap_info['width']}x{overlap_info['height']}")
        print(f"Угол поворота первого изображения: {overlap_info['first_rotation']:.2f}°")
        print(f"Угол поворота второго изображения: {overlap_info['second_rotation']:.2f}°")

if __name__ == "__main__":
    main()
