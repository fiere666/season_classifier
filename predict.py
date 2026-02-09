import joblib
import sys
import os
import glob
import numpy as np

try:
    model, label_encoder = joblib.load('season_classifier.pkl')
    print("Модель загружена успешно!")
except FileNotFoundError:
    print("Файл модели не найден Сначала обучите модель.")
    print("   Запустите: python train.py")
    sys.exit(1)
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    sys.exit(1)

def show_image_menu():
    """Показывает меню выбора изображения"""
    print(" ВЫБЕРИТЕ ИЗОБРАЖЕНИЕ")
    
    images = []
    index = 1
    
    seasons = ['spring', 'summers', 'autumn', 'winter','random']
    season_names = {
        'spring': 'Весна',
        'summers': 'Лето',
        'autumn': 'Осень',
        'winter': 'Зима',
        'random': 'Случайные изображения'
    }
    
    for season in seasons:
        season_path = os.path.join("data", season)
        
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in glob.glob(os.path.join(season_path, ext)):
                images.append(img_path)
                img_name = os.path.basename(img_path)
                print(f"{index:2d}. {season_names[season]} - {img_name}")
                index += 1
    
    print(f"\n{index:2d}. Ввести свой путь к изображению")
    print(f"{index + 1:2d}. Выход")
    
    return images, index

def predict_image(image_path):
    """Предсказывает сезон для одного изображения"""
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return

    from model import predict_season
    predicted_class, probabilities = predict_season(model, label_encoder, image_path)
    
    if predicted_class is None:
        print("Не удалось проанализировать изображение")
        return

    print(f" РЕЗУЛЬТАТ: {os.path.basename(image_path)}")

    for i, season in enumerate(label_encoder.classes_):
        print(f"{season}: {probabilities[i]*100:5.1f}%")
    
    print(f"\n Предсказанный сезон: {predicted_class}")
    print(f"Уверенность: {probabilities[np.argmax(probabilities)]*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        predict_image(image_path)
    else:
        images, last_index = show_image_menu()
        
        while True:
            try:
                choice = input("\n Введите номер: ")
                
                if not choice.isdigit():
                    print(" Введите число")
                    continue
                
                choice_num = int(choice)
                
                if choice_num == last_index + 1:
                    print(" До свидания!")
                    break
                elif choice_num == last_index:
                    custom_path = input("Введите путь к изображению: ").strip()
                    if custom_path:
                        predict_image(custom_path)
                elif 1 <= choice_num <= len(images):
                    image_path = images[choice_num - 1]
                    predict_image(image_path)
                else:
                    print(f"Введите число от 1 до {last_index + 1}")
                    
            except KeyboardInterrupt:
                print("\nПрограмма завершена")
                break
            except Exception as e:
                print(f"Ошибка: {e}")