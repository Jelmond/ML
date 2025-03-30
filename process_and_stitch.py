import os
import sys
from main import crop_overlapping_images
from stitch_images import stitch_images

def process_and_stitch(image_path, output_dir="results", stitched_dir="stitched"):
    """
    Вырезает два перекрывающихся фрагмента из изображения, а затем склеивает их обратно.
    
    Args:
        image_path: Путь к исходному изображению
        output_dir: Директория для сохранения вырезанных фрагментов
        stitched_dir: Директория для сохранения склеенного изображения
    """
    # Вырезаем и сохраняем два перекрывающихся фрагмента изображения
    first_path, second_path, overlap_info = crop_overlapping_images(image_path, output_dir)
    
    if not first_path or not second_path or not overlap_info:
        print("Ошибка при вырезании фрагментов изображения")
        return
    
    # Склеиваем фрагменты обратно
    stitched_path = stitch_images(first_path, second_path, overlap_info, stitched_dir)
    
    if stitched_path:
        print(f"\nИзображение успешно обработано и склеено обратно: {stitched_path}")
    else:
        print("Ошибка при склеивании фрагментов изображения")

def main():
    # Проверяем, передан ли путь к изображению как аргумент командной строки
    if len(sys.argv) < 2:
        print("Использование: python process_and_stitch.py <путь_к_изображению>")
        return
    
    image_path = sys.argv[1]
    
    # Проверяем существование файла
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден")
        return
    
    # Обрабатываем и склеиваем изображение
    process_and_stitch(image_path)

if __name__ == "__main__":
    main() 