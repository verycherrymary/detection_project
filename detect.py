from pathlib import Path
import streamlit as st
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
from IPython.display import display
import os
from PIL import Image as ImagePIL
from roboflow import Roboflow

# ====================== главная страница ============================
# параметры главной страницы
# https://docs.streamlit.io/library/api-reference/utilities/st.set_page_config
# 1 control
st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto",
    page_title="Rodent_detection_project",
    page_icon="🧊",
)

# ----------- функции -------------------------------------

# функция для загрузки картинки с диска
# кэшируем, иначе каждый раз будет загружаться заново
@st.cache_data
def load_image(image_path):
    image = ImagePIL.open(image_path)
    # обрезка до нужного размера с сохранением пропорций
    MAX_SIZE = (600,400)
    image.thumbnail(MAX_SIZE)
    return image

# функция загрузки моделей Yolo и Roboflow
# кэшируем, иначе каждый раз будет загружаться заново
@st.cache_data
def load_model(model_path):
    # загрузка готовой предобученной модели из файла
    model=YOLO (model_path) 
    return model
# кэшируем, иначе каждый раз будет загружаться заново
@st.cache_data
def load_model_rob():
    # Загрузка готовой модели с сайта Roboflow
    rf = Roboflow(api_key="o90RvmmLcvrsH55PkfK9")
    project = rf.workspace().project("cuterat-mon9g")
    model = project.version(3).model
    return model

# ------------- загрузка картинки для страницы и модели ---------

# путь до картинки
image_path = Path.cwd() / 'photo_2024-07-13_16-17-23.jpg'
image = load_image(image_path)

# путь до модели
model_path = Path.cwd() / 'best_v8n_collab.pt'
model_yolo = load_model(model_path)
model_rob=load_model_rob()

# ---------- отрисовка текста и картинки ------------------------
# 2 control
st.write(
    """
    # Обнаружение грызунов (мышей/крыс)
    ### Загрузите картинку или видео и получите результат
    """
)

# отрисовка картинки на странице
# 3 control
st.image(image)

# Добавление элемента управления загрузкой файла
uploaded_file = st.sidebar.file_uploader("#### Выберите файл (картинку) для загрузки, иначе будет ошибка")
# Проверяем, был ли загружен файл
if uploaded_file is not None:
    # Преобразуем объект BytesIO в изображение PIL
    img = ImagePIL.open(uploaded_file)
# Создаем путь к папке для сохранения
output_dir = Path.cwd() /'img_mouse'
output_filename = 'mouse_new.jpg'
output_path = os.path.join(output_dir, output_filename)
st.write(output_path)
# Сохраняем изображение в формате JPEG
img.save(output_path, format='JPEG')
mouse_img=output_path
# визуализация детекции готовой моделью roboflow на картинке
model_rob.predict(mouse_img, confidence=40, overlap=30).save("prediction.jpg")
a=model_rob.predict(mouse_img, confidence=40, overlap=30).json()
pred=a['predictions']
first = pred [0]
st.write("### Посмотрите результат детекции ниже - rats, rodent (крысы, мыши,грызуны)):")
st.write('##### Класс:', first['class'], 'Вероятность:','{:.2f}'.format(first['confidence']))
st.image('prediction.jpg')


# Добавление элемента управления загрузкой видеофайла
uploaded_video = st.sidebar.file_uploader("#### Выберите файл (видео) для загрузки, иначе будет ошибка")
# Проверка наличия загруженного файла
if uploaded_video is not None:
    # Сохранение файла в папку
    with open('img_mouse//video_new.mp4', 'wb') as f:
        f.write(uploaded_video.getbuffer())
st.write("##### Видео с детекцией откроется в отдельном окне windows, не браузера")
st.write("##### Для повторного просмотра этого же видео нажмите Rerun")
source = "img_mouse//video_new.mp4"
# детекция грызунов по видео без сохранения, с показом в отдельном окне windows
results = model_yolo(source, save=False, show=True)
# video_file = open('runs//detect//predict4//video_mouse.avi', 'rb')
# video_bytes = video_file.read()

# st.video(yolo_detect())
