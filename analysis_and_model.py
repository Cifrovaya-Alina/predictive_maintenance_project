import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
from ucimlrepo import fetch_ucirepo
from io import BytesIO
import base64
# Кэшируем загрузку датасета
@st.cache_data
def load_dataset():
    try:
        dataset = fetch_ucirepo(id=601)
        data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        
        # Правильные названия столбцов из датасета
        column_mapping = {
            'Air temperature': 'Air_temperature_K',
            'Process temperature': 'Process_temperature_K',
            'Rotational speed': 'Rotational_speed_rpm',
            'Torque': 'Torque_Nm',
            'Tool wear': 'Tool_wear_min',
            'Machine failure': 'Machine_failure',
            'Type': 'Type'
        }
        
        data = data.rename(columns=column_mapping)
        return data
        
    except Exception as e:
        st.error(f"Ошибка загрузки датасета: {str(e)}")
        return None
def get_models_and_metrics(X_train, y_train, X_test, y_test, model_type="XGBoost", use_smote=True):
    """Обучает модели и возвращает метрики"""
    if model_type == "Logistic Regression":
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', max_depth=10, random_state=42)
    elif model_type == "XGBoost":
        model = XGBClassifier(n_estimators=200, scale_pos_weight=(len(y_train)/sum(y_train)), 
                             learning_rate=0.1, random_state=42)
    else:  # SVM
        model = SVC(class_weight='balanced', probability=True, random_state=42)

    if use_smote:
        model = imbpipeline([('smote', SMOTE(random_state=42)), ('model', model)])
    
    model.fit(X_train, y_train)
    
    # Получаем предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    # Рассчитываем метрики
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'fpr': None,
        'tpr': None
    }
    
    # ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    metrics.update({'fpr': fpr, 'tpr': tpr})
    
    return model, metrics
def analysis_and_model_page():
    st.title("Анализ данных и модель для предиктивного обслуживания")
    
    # Загрузка данных
    st.header("1. Загрузка данных")
    data = load_dataset()
    
    if data is None:
        return
    
    # Проверяем наличие необходимых столбцов
    required_columns = [
        'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm',
        'Torque_Nm', 'Tool_wear_min', 'Machine_failure', 'Type'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Отсутствуют необходимые столбцы: {missing_columns}")
        st.write("Доступные столбцы:", list(data.columns))
        return
    
    st.success("Датасет успешно загружен!")
    
    # Предобработка данных
    st.header("2. Предобработка данных")
    
    if st.checkbox("Показать первые 5 строк данных"):
        st.write(data.head())
    
    # Удаление ненужных столбцов
    cols_to_drop = ['UDI', 'Product_ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    
    # Кодирование категориальных признаков
    if 'Type' in data.columns:
        le = LabelEncoder()
        data['Type'] = le.fit_transform(data['Type'])
        st.write("Типы оборудования закодированы как:", dict(zip(le.classes_, le.transform(le.classes_))))
    
    # Проверка дисбаланса классов
    st.subheader("Распределение целевой переменной")
    fig, ax = plt.subplots()
    data['Machine_failure'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Количество отказов (1) и исправных состояний (0)")
    st.pyplot(fig)
    
    # Масштабирование числовых признаков
    num_cols = ['Air_temperature_K', 'Process_temperature_K', 
               'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    
    # Разделение данных
    X = data.drop(columns=['Machine_failure'])
    y = data['Machine_failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Обучение модели
    st.header("3. Обучение модели")
    
    # Настройки модели
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Выберите модель", 
            ["Logistic Regression", "Random Forest", "XGBoost", "SVM"]
        )
    with col2:
        use_smote = st.checkbox("Использовать SMOTE для балансировки классов", value=True)
    
    # Параметры моделей
    if model_type == "Logistic Regression":
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced_subsample',
            max_depth=10,
            random_state=42
        )
    elif model_type == "XGBoost":
        model = XGBClassifier(
            n_estimators=200,
            scale_pos_weight=(len(y_train)/sum(y_train)),
            learning_rate=0.1,
            random_state=42,
            tree_method='hist',  # Используем histogram-based метод
            enable_categorical=False  # Явно отключаем, так как признаки уже закодированы
        )
    else:  # SVM
        model = SVC(
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    
    # Обучение с SMOTE или без
    if use_smote:
        pipeline = imbpipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        model = pipeline
    else:
        model.fit(X_train, y_train)
    
    # Оценка модели
    st.header("4. Оценка модели")
    
    # Предсказания
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    # Метрики
    st.subheader("Основные метрики")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        st.metric("Precision (класс 1)", f"{classification_report(y_test, y_pred, output_dict=True)['1']['precision']:.2f}")
    with col2:
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.2f}")
        st.metric("Recall (класс 1)", f"{classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.2f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred), 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=ax,
        annot_kws={"size": 16}
    )
    ax.set_xlabel("Предсказанные")
    ax.set_ylabel("Фактические")
    st.pyplot(fig)
    
    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred)
    st.text(report)
    
    # ROC-кривая
    st.subheader("ROC-кривая")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{model_type} (AUC = {roc_auc_score(y_test, y_proba):.2f})")
    ax.plot([0,1], [0,1], linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)
    
    # Важность признаков (для деревьев)
    if model_type in ["Random Forest", "XGBoost"]:
        st.subheader("Важность признаков")
        
        # Получаем доступ к самой модели внутри Pipeline
        if use_smote:
            trained_model = model.named_steps['model']  # Для Pipeline с SMOTE
        else:
            trained_model = model  # Если SMOTE не используется
        
        # Проверяем поддержку feature_importances_
        if hasattr(trained_model, 'feature_importances_'):
            importances = trained_model.feature_importances_
            feat_importances = pd.Series(importances, index=X.columns)
            fig, ax = plt.subplots()
            feat_importances.sort_values().plot(kind='barh', ax=ax)
            ax.set_title("Важность признаков")
            st.pyplot(fig)
        else:
            st.warning("Эта модель не поддерживает вывод важности признаков")
    
    # Предсказания
    st.header("5. Прогнозирование отказов")
    
    with st.form("prediction_form"):
        st.write("Введите параметры оборудования:")
        col1, col2 = st.columns(2)
        
        with col1:
            type_ = st.selectbox("Тип оборудования", ["L", "M", "H"])
            air_temp = st.number_input("Температура воздуха (K)", value=300.0, min_value=290.0, max_value=320.0)
            process_temp = st.number_input("Температура процесса (K)", value=310.0, min_value=300.0, max_value=330.0)
        
        with col2:
            rotational = st.number_input("Скорость вращения (rpm)", value=1500, min_value=1000, max_value=3000)
            torque = st.number_input("Крутящий момент (Nm)", value=40.0, min_value=0.0, max_value=100.0)
            tool_wear = st.number_input("Износ инструмента (min)", value=0, min_value=0, max_value=300)
        
        submitted = st.form_submit_button("Предсказать")
        
        if submitted:
            input_data = pd.DataFrame({
                'Type': [0 if type_ == "L" else 1 if type_ == "M" else 2],
                'Air_temperature_K': [air_temp],
                'Process_temperature_K': [process_temp],
                'Rotational_speed_rpm': [rotational],
                'Torque_Nm': [torque],
                'Tool_wear_min': [tool_wear]
            })
            
            # Масштабирование
            input_data[num_cols] = scaler.transform(input_data[num_cols])
            
            # Предсказание
            try:
                pred = model.predict(input_data)[0]
                proba = float(model.predict_proba(input_data)[0,1])  # Явное преобразование в float
    
                st.subheader("Результат прогноза")
                if pred == 1:
                    st.error(f"Прогноз: ОТКАЗ оборудования (вероятность: {proba:.2%})")
                else:
                    st.success(f"Прогноз: оборудование исправно (вероятность отказа: {proba:.2%})")
    
                # Прогресс-бар
                st.progress(float(proba))
                st.write(f"Вероятность отказа: {proba:.2%}")
    
            except Exception as e:
                st.error(f"Ошибка при прогнозировании: {str(e)}")
