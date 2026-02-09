import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import glob

def extract_features(image_path):
    """Извлекает цветовые признаки из изображения"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Ошибка чтения файла: {image_path}")
            return None
        img = cv2.resize(img, (128, 128))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mean_hue = np.mean(hsv[:,:,0])        
        mean_saturation = np.mean(hsv[:,:,1]) 
        mean_value = np.mean(hsv[:,:,2])      
        
        std_hue = np.std(hsv[:,:,0])
        std_saturation = np.std(hsv[:,:,1])
        std_value = np.std(hsv[:,:,2])
        
        hist_hue = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_hue = hist_hue / hist_hue.sum()  
        
        green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
        green_ratio = np.sum(green_mask > 0) / (128 * 128)
        
        cold_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
        cold_ratio = np.sum(cold_mask > 0) / (128 * 128)
        
        warm_mask1 = cv2.inRange(hsv, (0, 50, 50), (20, 255, 255))
        warm_mask2 = cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
        warm_ratio = (np.sum(warm_mask1 > 0) + np.sum(warm_mask2 > 0)) / (128 * 128)
        
        features = np.array([
            mean_hue, mean_saturation, mean_value,
            std_hue, std_saturation, std_value,
            green_ratio, cold_ratio, warm_ratio
        ])
        
        features = np.concatenate([features, hist_hue])
        return features
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None

def load_dataset(data_dir):
    """Загружает все изображения и извлекает признаки"""
    features = []
    labels = []
    
    seasons = {
        'spring': 'весна',
        'summers': 'лето', 
        'autumn': 'осень',
        'winter': 'зима'
    }
    
    for eng_name, rus_name in seasons.items():
        season_dir = os.path.join(data_dir, eng_name)
        image_paths = glob.glob(os.path.join(season_dir, '*.jpg')) + \
                     glob.glob(os.path.join(season_dir, '*.jpeg')) + \
                     glob.glob(os.path.join(season_dir, '*.png'))
        
        print(f"Загрузка {len(image_paths)} изображений для {rus_name}")
        
        for img_path in image_paths:
            feat = extract_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append(rus_name)
    
    if len(features) == 0:
        raise ValueError("Не загружено ни одного изображения! Проверьте пути.")
    
    return np.array(features), np.array(labels)

def train_season_classifier(data_dir="data"):
    """Основная функция обучения - возвращает обученную модель и кодировщик"""
    X, y = load_dataset(data_dir)
    
    print(f"\nЗагружено {len(X)} изображений")
    print(f"Размерность признаков: {X.shape}")
    print(f"Классы: {np.unique(y)}")
    print(f"Распределение по классам: {np.bincount(LabelEncoder().fit_transform(y))}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )
    
    model = RandomForestClassifier(
        n_estimators=100,      
        max_depth=10,          
        min_samples_split=5,   
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nТочность на обучающей выборке: {train_score:.2%}")
    print(f"Точность на тестовой выборке: {test_score:.2%}")
    
    print("\nКросс-валидация (4 фолда):")
    cv_scores = cross_val_score(model, X, y_encoded, cv=4)
    print(f"Средняя точность: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
     
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')

    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            plt.text(j, i, cm[i, j], ha='center', va='center')

    plt.xticks(range(len(le.classes_)), le.classes_)
    plt.yticks(range(len(le.classes_)), le.classes_)
    plt.title('Матрица ошибок')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    plt.figure(figsize=(10, 6))
    feature_names = ['mean_hue', 'mean_sat', 'mean_val',
                    'std_hue', 'std_sat', 'std_val',
                    'green_ratio', 'cold_ratio', 'warm_ratio'] + \
                    [f'hist_{i}' for i in range(32)]

    important_features = 15
    importances = model.feature_importances_[:important_features]
    indices = np.argsort(importances)[::-1][:important_features]
    
    plt.title(f'Важность признаков (топ-{important_features})')
    plt.bar(range(important_features), importances[indices])
    plt.xticks(range(important_features), 
               np.array(feature_names)[indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("График важности признаков сохранен как 'feature_importance.png'")
    
    return model, le

def predict_season(model, label_encoder, image_path):
    """Предсказывает сезон для одного изображения"""
    features = extract_features(image_path)
    if features is None:
        return None, None
    
    features = features.reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = label_encoder.classes_[predicted_class_idx]
    
    return predicted_class, probabilities

def test_examples(model, le):
    """Тестирует модель на нескольких примерах"""
    for season in ['весна', 'лето', 'осень', 'зима']:
        eng_season = {
            'весна': 'spring',
            'лето': 'summers', 
            'осень': 'autumn',
            'зима': 'winter'
        }[season]
        
        pattern = os.path.join("data", eng_season, "*.*")
        images = glob.glob(pattern)
        
        if images:
            test_image = images[0]
            print(f"\nТестируем: {test_image}")
            
            predicted_class, probabilities = predict_season(model, le, test_image)
            
            if predicted_class is not None:
                print(f"Результат: {predicted_class}")
                for i, season_name in enumerate(le.classes_):
                    print(f"{season_name}: {probabilities[i]*100:5.1f}%")