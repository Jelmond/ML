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
        
        # Используем информацию о поворотах из overlap_info
        first_rotation = overlap_info.get('first_rotation', 0)
        second_rotation = overlap_info.get('second_rotation', 0)
        
        # Получаем размеры изображений
        h1, w1 = first_img_cv.shape[:2]
        h2, w2 = second_img_cv.shape[:2]
        
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
            
            # Создаем маску для плавного перехода
            overlap_mask = np.zeros((result_height, result_width), dtype=np.uint8)
            # Заполняем область перекрытия градиентом
            for y in range(int(offset_y), int(offset_y+h1)):
                for x in range(int(offset_x), int(offset_x+w1)):
                    # Проверяем, находится ли точка в области перекрытия
                    if 0 <= y-int(offset_y) < h1 and 0 <= x-int(offset_x) < w1:
                        if result_img[y, x].any() and first_img_cv[y-int(offset_y), x-int(offset_x)].any():
                            # Точка находится в области перекрытия
                            # Создаем градиент от центра перекрытия
                            dist_from_center = abs(x - (int(offset_x) + w1/2)) / (w1/2)
                            overlap_mask[y, x] = int((1.0 - dist_from_center) * 255)

            # Создаем копию результата для первого изображения
            first_overlay = np.zeros_like(result_img)
            first_overlay[int(offset_y):int(offset_y+h1), int(offset_x):int(offset_x+w1)] = first_img_cv

            # Смешиваем изображения с использованием маски
            result_img = blend_images(result_img, first_overlay, overlap_mask)
            
            # Обрезаем черные края
            result_img = crop_black_borders(result_img)
            
            # Выравниваем изображение (поворачиваем, чтобы оно было прямым)
            result_img = align_image(result_img)
            
            # После склейки, применяем выравнивание с учетом исходных углов поворота
            # Вычисляем общий угол поворота (разница между углами поворота фрагментов)
            rotation_diff = second_rotation - first_rotation
            
            # Выравниваем результат с учетом этой разницы
            h_result, w_result = result_img.shape[:2]
            center = (w_result // 2, h_result // 2)
            
            # Создаем матрицу поворота для компенсации разницы углов
            M = cv2.getRotationMatrix2D(center, -rotation_diff, 1.0)
            
            # Вычисляем новые размеры после поворота
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h_result * sin) + (w_result * cos))
            new_h = int((h_result * cos) + (w_result * sin))
            
            # Корректируем матрицу поворота
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Применяем поворот
            aligned_result = cv2.warpAffine(result_img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
            
            # Обрезаем черные края
            result_img = crop_black_borders(aligned_result)
            
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

def blend_images(img1, img2, mask):
    """
    Плавно смешивает два изображения с использованием маски и размытия.
    
    Args:
        img1: Первое изображение
        img2: Второе изображение
        mask: Маска для смешивания (0-255)
        
    Returns:
        Смешанное изображение
    """
    # Размываем маску для более плавного перехода
    blurred_mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
    # Нормализуем маску
    mask_normalized = blurred_mask.astype(float) / 255.0
    # Расширяем маску до 3 каналов
    mask_3channel = np.stack([mask_normalized] * 3, axis=2)
    
    # Смешиваем изображения
    blended = (img1 * (1.0 - mask_3channel) + img2 * mask_3channel).astype(np.uint8)
    
    # Находим область перехода (где маска не 0 и не 255)
    transition_mask = np.logical_and(mask > 10, mask < 245).astype(np.uint8) * 255
    
    # Если есть область перехода, применяем дополнительное размытие
    if np.any(transition_mask):
        # Расширяем область перехода
        kernel = np.ones((5, 5), np.uint8)
        transition_mask = cv2.dilate(transition_mask, kernel, iterations=2)
        
        # Размываем изображение в области перехода
        blurred_img = cv2.GaussianBlur(blended, (5, 5), 0)
        
        # Нормализуем маску перехода
        transition_mask_norm = transition_mask.astype(float) / 255.0
        transition_mask_3ch = np.stack([transition_mask_norm] * 3, axis=2)
        
        # Применяем размытие только в области перехода
        blended = (blended * (1.0 - transition_mask_3ch) + 
                  blurred_img * transition_mask_3ch).astype(np.uint8)
    
    return blended

def crop_black_borders(image):
    """
    Обрезает черные края изображения.
    
    Args:
        image: Изображение для обработки
        
    Returns:
        Обрезанное изображение
    """
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Бинаризуем изображение
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Находим самый большой контур
        max_contour = max(contours, key=cv2.contourArea)
        
        # Получаем ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # Обрезаем изображение
        return image[y:y+h, x:x+w]
    
    return image

def align_image(image):
    """
    Выравнивает изображение, чтобы оно было прямым.
    Использует улучшенный алгоритм для медицинских изображений.
    
    Args:
        image: Изображение для выравнивания
        
    Returns:
        Выровненное изображение
    """
    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применяем адаптивное пороговое значение для лучшего выделения структур
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Находим края с более низким порогом для рентгеновских снимков
    edges = cv2.Canny(binary, 30, 100, apertureSize=3)
    
    # Применяем морфологические операции для усиления линий
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Находим линии с помощью преобразования Хафа с меньшим порогом
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
    
    if lines is not None and len(lines) > 0:
        # Вычисляем углы наклона линий
        angles = []
        for line in lines:
            rho, theta = line[0]
            # Преобразуем угол в градусы
            angle_deg = theta * 180 / np.pi
            
            # Фокусируемся на почти вертикальных и почти горизонтальных линиях
            if (angle_deg < 10 or angle_deg > 170) or (80 < angle_deg < 100):
                # Нормализуем угол
                if angle_deg > 90:
                    angle_deg = angle_deg - 180
                
                angles.append(angle_deg)
        
        if angles:
            # Фильтруем выбросы
            angles = np.array(angles)
            
            # Используем кластеризацию для определения основных направлений
            from sklearn.cluster import KMeans
            
            # Если есть достаточно углов для кластеризации
            if len(angles) >= 2:
                # Преобразуем углы в 2D точки на единичной окружности для корректной кластеризации
                points = np.array([[np.cos(a * np.pi / 180), np.sin(a * np.pi / 180)] for a in angles])
                kmeans = KMeans(n_clusters=2, random_state=0).fit(points)
                
                # Получаем центры кластеров и преобразуем обратно в углы
                cluster_centers = kmeans.cluster_centers_
                cluster_angles = [np.arctan2(center[1], center[0]) * 180 / np.pi for center in cluster_centers]
                
                # Выбираем угол ближайший к горизонтали или вертикали
                rotation_angle = min(cluster_angles, key=lambda x: min(abs(x), abs(x-90), abs(x+90)))
            else:
                # Если недостаточно углов, используем медиану
                rotation_angle = np.median(angles)
            
            # Ограничиваем угол поворота до разумных пределов
            if abs(rotation_angle) > 45:
                # Если угол слишком большой, вероятно это ошибка
                rotation_angle = 0
            
            # Получаем размеры изображения
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Создаем матрицу поворота
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Вычисляем новые размеры после поворота
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Корректируем матрицу поворота
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Применяем поворот
            rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            
            # Обрезаем черные края после поворота
            return crop_black_borders(rotated)
    
    # Если не удалось определить угол, возвращаем исходное изображение
    return image

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