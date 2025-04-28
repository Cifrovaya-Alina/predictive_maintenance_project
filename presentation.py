import streamlit as st  
import reveal_slides as rs
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pandas as pd
from analysis_and_model import load_dataset, get_models_and_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

def fig_to_base64(fig):
    """Конвертирует matplotlib figure в base64"""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)  
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@st.cache_data(ttl=3600)  # Кэшируем на 1 час
def load_and_preprocess_data():
    """Загрузка и предобработка данных с кэшированием"""
    with st.spinner('Загрузка и обработка данных...'):
        data = load_dataset()
        if data is None:
            st.error("Не удалось загрузить данные")
            return None
        
        # Предобработка данных
        data = data.drop(columns=['UDI', 'Product_ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
        le = LabelEncoder()
        data['Type'] = le.fit_transform(data['Type'])
        
        # Разделение данных
        X = data.drop(columns=['Machine_failure'])
        y = data['Machine_failure']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test

@st.cache_data(ttl=3600)
def train_models(X_train, y_train, X_test, y_test):
    """Обучение моделей с кэшированием"""
    models_to_train = {
        "Logistic Regression": "Logistic Regression",
        "Random Forest": "Random Forest", 
        "XGBoost": "XGBoost",
        "SVM": "SVM"
    }
    
    roc_data = {}
    metrics = []
    model_objects = {}
    
    for name, model_type in models_to_train.items():
        with st.spinner(f'Обучение модели {name}...'):
            model, model_metrics = get_models_and_metrics(
                X_train, y_train, X_test, y_test, model_type
            )
            model_objects[name] = model
            roc_data[name] = {
                'fpr': model_metrics['fpr'],
                'tpr': model_metrics['tpr'],
                'auc': model_metrics['roc_auc']
            }
            metrics.append({
                'Model': name,
                'Accuracy': round(model_metrics['accuracy'], 3),
                'ROC-AUC': round(model_metrics['roc_auc'], 3),
                'Precision_1': round(model_metrics['classification_report']['1']['precision'], 3),
                'Recall_1': round(model_metrics['classification_report']['1']['recall'], 3),
                'F1_1': round(model_metrics['classification_report']['1']['f1-score'], 3)
            })
    
    return roc_data, metrics, model_objects

def create_roc_plot(_roc_data):
    """Создает ROC-кривые (ленивая загрузка)"""
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in _roc_data.items():
        ax.plot(data['fpr'], data['tpr'], linewidth=2, 
                label=f'{name} (AUC = {data["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC-кривые моделей', fontsize=14, pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def create_confusion_matrix(_model, X_test, y_test):
    """Создает Confusion Matrix (ленивая загрузка)"""
    y_pred = _model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                annot_kws={"size": 14}, ax=ax,
                cbar_kws={'label': 'Количество наблюдений'})
    ax.set_title('Confusion Matrix (XGBoost)', fontsize=14, pad=20)
    ax.set_xlabel('Предсказанные', fontsize=12)
    ax.set_ylabel('Фактические', fontsize=12)
    ax.set_xticklabels(['Без отказа', 'Отказ'])
    ax.set_yticklabels(['Без отказа', 'Отказ'], rotation=0)
    plt.tight_layout()
    return fig

def create_feature_importance(_model, feature_names):
    """Создает график важности признаков (ленивая загрузка)"""
    if not hasattr(_model, 'feature_importances_'):
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = _model.feature_importances_
    indices = importances.argsort()[::-1]
    
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Важность признака', fontsize=12)
    ax.set_title('Важность признаков (Random Forest)', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig

def generate_metrics_table(_metrics):
    """Генерирует HTML таблицу с метриками"""
    table = """
    <table style="width:100%; border-collapse: collapse; margin: 15px 0;">
        <thead>
            <tr style="background-color: #f2f2f2;">
                <th style="padding: 8px; text-align: left; border: 1px solid #ddd;">Модель</th>
                <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Accuracy</th>
                <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">ROC-AUC</th>
                <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Precision (1)</th>
                <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">Recall (1)</th>
                <th style="padding: 8px; text-align: center; border: 1px solid #ddd;">F1-score (1)</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for model in _metrics:
        table += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>{model['Model']}</strong></td>
                <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{model['Accuracy']}</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{model['ROC-AUC']}</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{model['Precision_1']}</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{model['Recall_1']}</td>
                <td style="padding: 8px; text-align: center; border: 1px solid #ddd;">{model['F1_1']}</td>
            </tr>
        """
    
    table += """
        </tbody>
    </table>
    """
    return table

def presentation_page():
    st.title("Презентация проекта прогнозирования отказов оборудования")
    
    # Загружаем и предобрабатываем данные (с кэшированием)
    data = load_and_preprocess_data()
    if data is None:
        return
    X_train, X_test, y_train, y_test = data
    
    # Обучаем модели (с кэшированием)
    roc_data, metrics, model_objects = train_models(X_train, y_train, X_test, y_test)
    
    # Находим лучшую модель
    best_model = max(roc_data.items(), key=lambda x: x[1]['auc'])
    
    # Генерируем таблицу с метриками
    metrics_table = generate_metrics_table(metrics)
    
    # Формируем содержание презентации
    slides_content = """
<section data-background="#ffffff">
    <h1>Прогнозирование отказов промышленного оборудования</h1>
    <p style="text-align: center; margin-top: 50px; font-size: 1.0em;">
        Проект по машинному обучению для предсказания отказов оборудования
    </p>
</section>

<section>
    <h2>О проекте</h2>
    <ul>
        <li><strong>Цель</strong>: Разработка модели для предсказания отказов оборудования</li>
        <li><strong>Датасет</strong>: AI4I 2020 Predictive Maintenance Dataset</li>
        <li><strong>Объем данных</strong>: 10,000 записей, 8 признаков</li>
        <li><strong>Особенности</strong>:
            <ul>
                <li>Сильный дисбаланс классов (3.5% отказов)</li>
                <li>Разнотипные признаки (температура, крутящий момент и др.)</li>
            </ul>
        </li>
    </ul>
</section>

<section>
    <h2>Предобработка данных</h2>
    <ul>
        <li>Удалены идентификаторы (UDI, Product_ID)</li>
        <li>Кодирование категориального признака "Type"</li>
        <li>Нормализация числовых признаков</li>
        <li>Учет дисбаланса классов при обучении моделей</li>
    </ul>
</section>

<section>
    <h2>Результаты моделей</h2>
    """ + metrics_table + """
    <p style="margin-top: 20px;">
        <strong>Лучшая модель по ROC-AUC</strong>: """ + best_model[0] + """ (""" + f"{best_model[1]['auc']:.3f}" + """)
    </p>
</section>

<section>
    <h2>ROC-кривые моделей</h2>
    <img src="data:image/png;base64,""" + fig_to_base64(create_roc_plot(roc_data)) + """" alt="ROC curves">
    <p style="text-align: center; font-size: 0.9em;">
        Кривые показывают компромисс между True Positive Rate и False Positive Rate
    </p>
</section>

<section>
    <h2>Confusion Matrix (XGBoost)</h2>
    <img src="data:image/png;base64,""" + fig_to_base64(create_confusion_matrix(model_objects['XGBoost'], X_test, y_test)) + """" alt="Confusion Matrix">
    <div style="margin-top: 15px; font-size: 0.9em;">
        <p><strong>Метрики XGBoost для класса "Отказ":</strong></p>
        <ul>
            <li>Precision: """ + str(next(m['Precision_1'] for m in metrics if m['Model'] == 'XGBoost')) + """</li>
            <li>Recall: """ + str(next(m['Recall_1'] for m in metrics if m['Model'] == 'XGBoost')) + """</li>
            <li>F1-score: """ + str(next(m['F1_1'] for m in metrics if m['Model'] == 'XGBoost')) + """</li>
        </ul>
    </div>
</section>
"""

    # Добавляем слайд с важностью признаков, если есть данные
    fi_plot = create_feature_importance(model_objects['Random Forest'], X_test.columns)
    if fi_plot is not None:
        slides_content += """
<section>
    <h2>Важность признаков (Random Forest)</h2>
    <img src="data:image/png;base64,""" + fig_to_base64(fi_plot) + """" alt="Feature Importance">
    <p style="text-align: center; font-size: 0.9em;">
        Наиболее важные признаки для прогнозирования отказов
    </p>
</section>
"""

    # Заключительные слайды
    slides_content += """
<section>
    <h2>Выводы</h2>
    <ul>
        <li>XGBoost показал наилучший баланс метрик</li>
        <li>Высокий Recall для класса "Отказ" (87%)</li>
        <li>Приемлемая точность (Precision 37%) для дисбалансированных данных</li>
        <li>Random Forest имеет сопоставимое качество с лучшей интерпретируемостью</li>
    </ul>
</section>

<section>
    <h2>Возможные улучшения</h2>
    <ul>
        <li><strong>Оптимизация порога классификации</strong>:
            <ul>
                <li>Подбор порога по бизнес-метрикам (стоимость ошибок)</li>
                <li>Использование Precision-Recall кривой</li>
            </ul>
        </li>
        <li><strong>Борьба с дисбалансом</strong>:
            <ul>
                <li>Применение SMOTE/ADASYN для генерации синтетических примеров</li>
                <li>Использование взвешенных функций потерь</li>
            </ul>
        </li>
        <li><strong>Улучшение признаков</strong>:
            <ul>
                <li>Добавление временных характеристик оборудования</li>
                <li>Создание интерактивных признаков</li>
            </ul>
        </li>
        <li><strong>Развертывание</strong>:
            <ul>
                <li>Интеграция с промышленными системами сбора данных</li>
                <li>Разработка системы мониторинга в реальном времени</li>
            </ul>
        </li>
    </ul>
</section>
"""
    
    # Настройки презентации
    with st.sidebar: 
        st.header("Настройки презентации") 
        theme = st.selectbox("Тема", [
            "black", "white", "league", "beige", 
            "sky", "night", "serif", "simple", "solarized"
        ]) 
        height = st.slider("Высота слайдов", 1000, 1500, 1200) 
        transition = st.selectbox("Переход между слайдами", [
            "slide", "fade", "convex", "concave", "zoom", "none"
        ]) 
        enable_controls = st.checkbox("Показывать элементы управления", True)
        enable_progress = st.checkbox("Показывать прогресс бар", True)
    
    # Отображение презентации
    rs.slides( 
        slides_content, 
        height=height, 
        theme=theme, 
        config={ 
            "transition": transition, 
            "controls": enable_controls,
            "progress": enable_progress,
            "slideNumber": "c/t",
            "hash": True,
            "center": False,
            "margin": 0.05,
            "width": "100%",
            "height": "100%"
        }, 
        markdown_props={
            "separator": "^\\n\\n\\n",
            "verticalSeparator": "^\\n\\n"
        },
        allow_unsafe_html=True
    )

if __name__ == "__main__":
    presentation_page()
