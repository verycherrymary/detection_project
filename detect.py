from pathlib import Path
# import time
import streamlit as st
# import subprocess
# import ultralytics
# ultralytics.checks()
from ultralytics import YOLO
# from IPython.display import display
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

# функция по отображению класса и вероятности для детекции картинки
def check_for_mice(image_path):
    """Проверяет наличие грызунов на изображении."""
    a = model_rob.predict(image_path, confidence=40, overlap=30).json()
    pred=a['predictions']
    if len(pred) == 0:
        st.write('##### Грызуны не обнаружены')
    else:
        first = pred [0]
        st.write('##### Класс:', first['class'], 'Вероятность:','{:.2f}'.format(first['confidence']))

# загрузка примера картинки с детекцией
path_to_images = Path.cwd() /'pred_mouse' # путь к папке с детектирован.изображениями
# Список путей к изображениям
image_paths = []
image_names = []
# Перебор всех файлов в папке с изображениями
for file in os.listdir(path_to_images):
    if file.endswith('.jpg') or file.endswith('.png'):
        image_paths.append(os.path.join(path_to_images, file))
        image_names.append(file)
# st.write(image_paths) 
# st.write(image_names)
# Создание кнопки selector
button_selector = st.sidebar.selectbox('#### Выберите изображение из примеров детекции', image_paths)
st.write("### Посмотрите пример детекции ниже - rats, rodent (крысы, мыши,грызуны)):")
st.image(button_selector)

# загрузка примера видео с детекцией
# Конвертация видео из формата avi в формат mp4
# input_video_path= Path.cwd() /'pred_video//video_new2.avi'
# st.write(input_video_path)
# # Создайте путь для сохранения нового видео в формате MP4
# output_video_path = input_video_path.with_suffix('.mp4')
# st.write(output_video_path)
# # Запустите команду ffmpeg для конвертации видео
# subprocess.run(['ffmpeg', '-i', str(input_video_path), str(output_video_path)])
path_to_video=Path.cwd() /'pred_video'
# Список путей к изображениям
video_paths = []
# Перебор всех файлов в папке с изображениями
for file in os.listdir(path_to_video):
    if file.endswith('.mp4'):
        video_paths.append(os.path.join(path_to_video, file))
# st.write(video_paths)
# Создание кнопки selector
button_selector = st.sidebar.selectbox('#### Выберите видео из примеров детекции', video_paths)
st.write("### Посмотрите пример детекции ниже - rats, rodent (крысы, мыши,грызуны)):")
# video_file = open(button_selector, 'rb')
# video_bytes = video_file.read()
st.video(button_selector)

# Добавление элемента управления загрузкой картинки
uploaded_file = st.sidebar.file_uploader("#### Выберите свое изображение для загрузки, иначе будет ошибка")
# Проверяем, был ли загружен файл
if uploaded_file is not None:
    # Преобразуем объект BytesIO в изображение PIL
    img = ImagePIL.open(uploaded_file)
# Создаем путь к папке для сохранения
output_dir = Path.cwd() /'img_mouse'
output_filename = 'mouse_new.jpg'
output_path = os.path.join(output_dir, output_filename)
# st.write(output_path)
# Сохраняем изображение в формате JPEG
img.save(output_path, format='JPEG')
mouse_img=output_path
# визуализация детекции готовой моделью roboflow на картинке
model_rob.predict(mouse_img, confidence=40, overlap=30).save("prediction.jpg")
st.write("### Посмотрите результат детекции ниже - rats, rodent (крысы, мыши,грызуны)):")
check_for_mice(mouse_img)
st.image('prediction.jpg')

# Добавление элемента управления загрузкой видеофайла
uploaded_video = st.sidebar.file_uploader("#### Выберите свое видео для загрузки, иначе будет ошибка")
# Проверка наличия загруженного файла
if uploaded_video is not None:
    # Сохранение файла в папку
    with open(Path.cwd() /'img_mouse//video_new.mp4', 'wb') as f:
        f.write(uploaded_video.getbuffer())
st.write("##### Демонстрация видео с детекцией доступно пока только в локальной версии приложения")
source =Path.cwd() / "img_mouse//video_new.mp4"
# st.write(source)
# детекция грызунов по видео без сохранения, с показом в отдельном окне windows
results = model_yolo(source, save=False, show=True)
# Укажите путь к вашему видео в формате AVI
# input_video_path= Path.cwd() / "runs//detect//predict//video_new.avi"
# st.write(input_video_path)
# # Создайте путь для сохранения нового видео в формате MP4
# output_video_path = input_video_path.with_suffix('.mp4')
# st.write(output_video_path)
# # Запустите команду ffmpeg для конвертации видео
# subprocess.run(['ffmpeg', '-i', str(input_video_path), str(output_video_path)])
# # После завершения конвертации вы можете прочитать новый файл MP4
# video_path=Path.cwd() / "runs//detect//predict//video_new.mp4"
# video_file = open(video_path, 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)


