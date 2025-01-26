import requests
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import re

LATENT_DIM = 100

# Спектральная нормализация
class SpectralNormalization(layers.Layer):
    def __init__(self, layer, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.u = self.add_weight(
            shape=(1, self.layer.kernel.shape[-1]),
            initializer="random_normal",
            trainable=False,
            name="u"
        )

    def call(self, inputs, training=None):
        kernel = self.layer.kernel
        kernel_shape = tf.shape(kernel)
        kernel_reshaped = tf.reshape(kernel, [-1, kernel_shape[-1]])
        v = tf.linalg.matvec(kernel_reshaped, tf.transpose(self.u))
        v = tf.math.l2_normalize(v)

        u = tf.linalg.matvec(kernel_reshaped, v, transpose_a=True)
        u = tf.math.l2_normalize(u)

        sigma = tf.tensordot(u, tf.linalg.matvec(kernel_reshaped, v), axes=1)
        self.layer.kernel.assign(kernel / sigma)

        self.u.assign(u)
        return self.layer(inputs, training=training)

# Остаточный блок
def residual_block(input_tensor, filters, strides=1):
    x = layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    if strides != 1 or input_tensor.shape[-1] != filters:
        input_tensor = layers.Conv2D(filters, kernel_size=1, strides=strides, padding="same")(input_tensor)

    x = layers.Add()([x, input_tensor])
    x = layers.ReLU()(x)
    return x

# Генератор
def build_generator(latent_dim):
    model = models.Sequential([
        layers.InputLayer(input_shape=(latent_dim,)),
        layers.Dense(7 * 7 * 256, activation="relu"),
        layers.Reshape((7, 7, 256)),
        residual_block(layers.Input(shape=(7, 7, 256)), 256),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(1, kernel_size=4, strides=1, padding="same", activation="tanh")
    ])
    return model

# Дискриминатор
def build_discriminator(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = SpectralNormalization(layers.Conv2D(64, kernel_size=4, strides=2, padding="same"))(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(layers.Conv2D(128, kernel_size=4, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = SpectralNormalization(layers.Conv2D(256, kernel_size=4, strides=2, padding="same"))(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    x = SpectralNormalization(layers.Dense(1, activation="sigmoid"))(x)

    return models.Model(inputs, x)

# Функция для Wikipedia API
def get_wikipedia_content(title):
    """
    Получает контент статьи Wikipedia через API.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title
    }
    response = requests.get(url, params=params)
    data = response.json()

    # Извлечение текста статьи
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "") if page.get("extract") else None

def preprocess_text(text):
    """
    Подготовка текста:
    - Удаление спецсимволов.
    - Приведение к нижнему регистру.
    """
    text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    return text.lower()

def load_wikipedia_data(titles):
    """
    Загружает и подготавливает статьи из Википедии.
    """
    all_text = ""
    for title in titles:
        print(f"Загружается статья: {title}")
        content = get_wikipedia_content(title)
        if content:
            all_text += preprocess_text(content) + " "
    return all_text

# Пример: загрузка данных из Википедии
wikipedia_titles = ["Artificial Intelligence", "Machine Learning", "Deep Learning"]
corpus = load_wikipedia_data(wikipedia_titles)
print(f"Длина корпуса: {len(corpus)} символов")

# Обучение на тексте Википедии
def text_to_training_data(corpus, seq_length=100):
    """
    Преобразует текст в тренировочные данные для GAN.
    """
    sequences = []
    next_chars = []
    for i in range(0, len(corpus) - seq_length, seq_length):
        sequences.append(corpus[i:i + seq_length])
        next_chars.append(corpus[i + seq_length])
    return np.array(sequences), np.array(next_chars)

seq_length = 100
X, y = text_to_training_data(corpus, seq_length)

# Тренировочный цикл GAN будет использовать данные `X` и обучаться на последовательностях текста.
