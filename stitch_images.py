import os
import sys
import cv2
import numpy as np
from PIL import Image
import math
import glob

def get_next_number(directory, prefix=""):
    """
    Определяет следующий доступный номер для именования файлов в директории.
    
    Args:
        directory: Директория для проверки
        prefix: Префикс имени файла
        
    Returns:
        Следующий доступный номер
    """
    if not os.path.exists(directory):
        return 1
        
    # Ищем все файлы с числовыми именами
    pattern = os.path.join(directory, f"{prefix}*.*")
    files = glob.glob(pattern)
    
    if not files:
        return 1
        
    # Извлекаем числа из имен файлов
    numbers = []
    for file in files:
        basename = os.path.basename(file)
        name, _ = os.path.splitext(basename)
        name = name.replace(prefix, "")
        try:
            numbers.append(int(name))
        except ValueError:
            continue
    
    # Возвращаем следующий номер
    return max(numbers) + 1 if numbers else 1

def stitch_images(first_image_path, second_image_path, overlap_info, output_dir="stitched"):
    """
    Склеивает два изображения на основе информации о пересечении.
    Использует гомографию для учета поворота и масштаба.
    
    Args:
        first_image_path: Путь к первому изображению
        second_image_path: Путь ко второму изображению
        overlap_info: Словарь с информацией о пересечении
        output_dir: Директория для сохранения результата
        
    Returns:
        Путь к склеенному изображению
    """
    # Создаем директорию для результатов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Загружаем изображения с помощью OpenCV
        first_img_cv = cv2.imread(first_image_path)
        second_img_cv = cv2.imread(second_image_path)
        
        # Конвертируем в оттенки серого для поиска ключевых точек
        first_gray = cv2.cvtColor(first_img_cv, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.cvtColor(second_img_cv, cv2.COLOR_BGR2GRAY)
        
        # Инициализируем детектор ключевых точек SIFT
        sift = cv2.SIFT_create()
        
        # Находим ключевые точки и дескрипторы
        keypoints1, descriptors1 = sift.detectAndCompute(first_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(second_gray, None)
        
        # Используем FLANN для быстрого сопоставления ключевых точек
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Применяем тест соотношения Лоу для фильтрации хороших совпадений
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Если найдено достаточно хороших совпадений, вычисляем гомографию
        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Вычисляем матрицу гомографии
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            # Получаем размеры изображений
            h1, w1 = first_img_cv.shape[:2]
            h2, w2 = second_img_cv.shape[:2]
            
            # Вычисляем размеры результирующего изображения
            pts = np.float32([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            
            # Находим минимальные и максимальные координаты
            min_x = min(np.min(dst[:, 0, 0]), 0)
            min_y = min(np.min(dst[:, 0, 1]), 0)
            max_x = max(np.max(dst[:, 0, 0]), w1)
            max_y = max(np.max(dst[:, 0, 1]), h1)
            
            # Вычисляем смещение и размеры результирующего изображения
            offset_x = -min_x
            offset_y = -min_y
            result_width = int(max_x - min_x)
            result_height = int(max_y - min_y)
            
            # Создаем матрицу трансформации с учетом смещения
            translation_matrix = np.array([
                [1, 0, offset_x],
                [0, 1, offset_y],
                [0, 0, 1]
            ])
            
            # Комбинируем матрицы трансформации
            warp_matrix = np.dot(translation_matrix, H)
            
            # Применяем трансформацию ко второму изображению
            result_img = cv2.warpPerspective(second_img_cv, warp_matrix, (result_width, result_height))
            
            # Копируем первое изображение в результат с учетом смещения
            result_img[int(offset_y):int(offset_y+h1), int(offset_x):int(offset_x+w1)] = first_img_cv
            
            # Обрезаем пустые области
            # Конвертируем в оттенки серого
            result_gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
            # Находим ненулевые пиксели
            non_zero = cv2.findNonZero(result_gray)
            if non_zero is not None:
                # Находим ограничивающий прямоугольник
                x, y, w, h = cv2.boundingRect(non_zero)
                # Обрезаем изображение
                result_img = result_img[y:y+h, x:x+w]
            
            # Получаем следующий доступный номер для именования файла
            next_number = get_next_number(output_dir)
            
            # Формируем имя выходного файла с номером
            _, ext = os.path.splitext(first_image_path)
            output_path = os.path.join(output_dir, f"{next_number}{ext}")
            
            # Сохраняем результат
            cv2.imwrite(output_path, result_img)
            
            print(f"Изображения успешно склеены и сохранены в {output_path}")
            
            return output_path
        else:
            print(f"Недостаточно совпадений - {len(good_matches)}/{MIN_MATCH_COUNT}")
            
            # Если не удалось найти достаточно совпадений, используем информацию о пересечении
            # из аргументов функции (запасной вариант)
            return fallback_stitch(first_image_path, second_image_path, overlap_info, output_dir)
    
    except Exception as e:
        print(f"Ошибка при склеивании изображений: {e}")
        import traceback
        traceback.print_exc()
        
        # В случае ошибки используем запасной метод
        return fallback_stitch(first_image_path, second_image_path, overlap_info, output_dir)

def fallback_stitch(first_image_path, second_image_path, overlap_info, output_dir):
    """
    Запасной метод склеивания, если не удалось найти ключевые точки.
    """
    try:
        # Открываем изображения с помощью PIL
        first_img = Image.open(first_image_path).convert("RGBA")
        second_img = Image.open(second_image_path).convert("RGBA")
        
        # Получаем информацию о пересечении
        first_overlap_x, first_overlap_y = overlap_info['first_image']
        second_overlap_x, second_overlap_y = overlap_info['second_image']
        
        # Вычисляем относительное положение второго изображения
        dx = first_overlap_x - second_overlap_x
        dy = first_overlap_y - second_overlap_y
        
        # Получаем размеры изображений
        first_width, first_height = first_img.size
        second_width, second_height = second_img.size
        
        # Вычисляем размеры итогового изображения
        result_width = max(first_width, dx + second_width)
        result_height = max(first_height, dy + second_height)
        
        # Создаем пустое изображение нужного размера
        result_img = Image.new('RGBA', (result_width, result_height), (0, 0, 0, 0))
        
        # Вставляем первое изображение
        result_img.paste(first_img, (0, 0), first_img)
        
        # Вставляем второе изображение
        result_img.paste(second_img, (dx, dy), second_img)
        
        # Получаем следующий доступный номер для именования файла
        next_number = get_next_number(output_dir)
        
        # Формируем имя выходного файла с номером
        _, ext = os.path.splitext(first_image_path)
        output_path = os.path.join(output_dir, f"{next_number}{ext}")
        
        # Сохраняем результат
        result_img = result_img.convert('RGB')
        result_img.save(output_path)
        
        print(f"Изображения склеены запасным методом и сохранены в {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Ошибка при использовании запасного метода склеивания: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Проверяем, переданы ли все необходимые аргументы
    if len(sys.argv) < 3:
        print("Использование: python stitch_images.py <путь_к_первому_изображению> <путь_ко_второму_изображению> [<информация_о_пересечении>]")
        print("Если информация о пересечении не указана, скрипт попытается получить её из вывода первого скрипта.")
        return
    
    first_image_path = sys.argv[1]
    second_image_path = sys.argv[2]
    
    # Проверяем существование файлов
    if not os.path.exists(first_image_path):
        print(f"Файл {first_image_path} не найден")
        return
    
    if not os.path.exists(second_image_path):
        print(f"Файл {second_image_path} не найден")
        return
    
    # Если информация о пересечении передана как аргумент
    if len(sys.argv) >= 4:
        try:
            import json
            overlap_info = json.loads(sys.argv[3])
        except:
            print("Ошибка при разборе информации о пересечении. Используйте формат JSON.")
            return
    else:
        # Запрашиваем информацию о пересечении у пользователя
        print("Введите информацию о пересечении:")
        first_x = int(input("Координата X начала пересечения в первом изображении: "))
        first_y = int(input("Координата Y начала пересечения в первом изображении: "))
        second_x = int(input("Координата X начала пересечения во втором изображении: "))
        second_y = int(input("Координата Y начала пересечения во втором изображении: "))
        width = int(input("Ширина области пересечения: "))
        height = int(input("Высота области пересечения: "))
        first_rotation = float(input("Угол поворота первого изображения (в градусах): "))
        second_rotation = float(input("Угол поворота второго изображения (в градусах): "))
        
        overlap_info = {
            'first_image': (first_x, first_y),
            'second_image': (second_x, second_y),
            'width': width,
            'height': height,
            'first_rotation': first_rotation,
            'second_rotation': second_rotation
        }
    
    # Склеиваем изображения
    stitch_images(first_image_path, second_image_path, overlap_info)

if __name__ == "__main__":
    main() 