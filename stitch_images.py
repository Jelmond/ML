import os
import sys
from PIL import Image, ImageDraw
import math

def stitch_images(first_image_path, second_image_path, overlap_info, output_dir="stitched"):
    """
    Склеивает два изображения на основе информации о пересечении.
    
    Args:
        first_image_path: Путь к первому изображению
        second_image_path: Путь ко второму изображению
        overlap_info: Словарь с информацией о пересечении:
          {
            'first_image': (x1, y1),  # Координаты начала пересечения в первом изображении
            'second_image': (x2, y2),  # Координаты начала пересечения во втором изображении
            'width': w,               # Ширина области пересечения
            'height': h,              # Высота области пересечения
            'first_rotation': angle1, # Угол поворота первого изображения
            'second_rotation': angle2 # Угол поворота второго изображения
          }
        output_dir: Директория для сохранения результата
        
    Returns:
        Путь к склеенному изображению
    """
    # Создаем директорию для результатов, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Открываем изображения
        first_img = Image.open(first_image_path).convert("RGBA")
        second_img = Image.open(second_image_path).convert("RGBA")
        
        # Получаем информацию о пересечении
        first_overlap_x, first_overlap_y = overlap_info['first_image']
        second_overlap_x, second_overlap_y = overlap_info['second_image']
        overlap_width = overlap_info['width']
        overlap_height = overlap_info['height']
        first_rotation = overlap_info['first_rotation']
        second_rotation = overlap_info['second_rotation']
        
        # Поворачиваем изображения обратно
        first_img = first_img.rotate(-first_rotation, expand=True)
        second_img = second_img.rotate(-second_rotation, expand=True)
        
        # Получаем размеры изображений
        first_width, first_height = first_img.size
        second_width, second_height = second_img.size
        
        # Вычисляем размеры итогового изображения
        # Сначала определяем относительное положение второго изображения
        # относительно первого
        dx = first_overlap_x - second_overlap_x
        dy = first_overlap_y - second_overlap_y
        
        # Вычисляем координаты углов второго изображения относительно первого
        second_left = dx
        second_top = dy
        second_right = second_left + second_width
        second_bottom = second_top + second_height
        
        # Вычисляем размеры итогового изображения
        result_width = max(first_width, second_right)
        result_height = max(first_height, second_bottom)
        
        # Создаем пустое изображение нужного размера
        result_img = Image.new('RGBA', (result_width, result_height), (0, 0, 0, 0))
        
        # Вставляем первое изображение
        result_img.paste(first_img, (0, 0))
        
        # Вставляем второе изображение с использованием альфа-канала
        # Извлекаем альфа-канал как маску
        r, g, b, a = second_img.split()
        result_img.paste(second_img.convert("RGB"), (second_left, second_top), mask=a)
        
        # Формируем имя выходного файла
        first_filename = os.path.basename(first_image_path)
        first_name, ext = os.path.splitext(first_filename)
        output_path = os.path.join(output_dir, f"{first_name.replace('_crop1', '')}_stitched{ext}")
        
        # Сохраняем результат
        result_img = result_img.convert('RGB')  # Конвертируем в RGB перед сохранением
        result_img.save(output_path)
        
        print(f"Изображения успешно склеены и сохранены в {output_path}")
        
        return output_path
    
    except Exception as e:
        print(f"Ошибка при склеивании изображений: {e}")
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