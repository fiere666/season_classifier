import joblib
from model import train_season_classifier, test_examples

def main():
    try:
        print("ОБУЧЕНИЕ МОДЕЛИ КЛАССИФИКАЦИИ ВРЕМЕН ГОДА")
        print("\nЗагрузка данных и обучение модели...")
        model, label_encoder = train_season_classifier("data")
        
        print("\nТестирование модели")
        test_examples(model, label_encoder)
        
        print("\n Сохранение модели")
        joblib.dump((model, label_encoder), 'season_classifier.pkl')
        print("Модель сохранена как 'season_classifier.pkl'")
        
        print(" ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("\nТеперь вы можете использовать модель:")
        print("1.Для предсказания: python predict.py")
        print("2.Для обучения заново: python train.py")
        
    except Exception as e:
        print(f"\n Ошибка: {e}")
        print("\nПроверьте что:")
        print("1. Папка 'data' существует и содержит подпапки spring, summers, autumn, winter")
        print("2. В каждой подпапке есть изображения (.jpg, .jpeg или .png)")
        print("3. Установлены все библиотеки: pip install scikit-learn opencv-python numpy matplotlib joblib")

if __name__ == "__main__":
    main()