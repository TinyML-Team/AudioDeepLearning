import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display


# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# 데이터 세트 가져 오기
data_dir = pathlib.Path('data')


#데이터 세트에 대한 기본 통계를 확인합니다.
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
print('Commands:', commands)


#오디오 파일을 목록으로 추출하고 섞습니다.
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

#각각 80:10:10 비율을 사용하여 파일을 학습, 검증 및 테스트 세트로 분할합니다.
train_files_number=int(num_samples*0.8)
test_files_number=int(num_samples*0.1)
train_files = filenames[:train_files_number]
val_files = filenames[train_files_number: train_files_number + test_files_number]
test_files = filenames[-test_files_number:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))


# 오디오 파일 및 레이블 읽기
# 오디오 파일은 처음에는 이진 파일로 읽혀지며 숫자 텐서로 변환 할 수 있습니다.
# 오디오 파일을로드하려면 WAV 인코딩 오디오를 Tensor 및 샘플 속도로 반환하는 tf.audio.decode_wav 를 사용합니다.
# WAV 파일에는 초당 샘플 수가 설정된 시계열 데이터가 포함됩니다. 각 샘플은 특정 시간에 오디오 신호의 진폭을 나타냅니다. mini_speech_commands 의 파일과 같이 16 비트 시스템에서 값의 범위는 -32768에서 32767입니다.이 데이터 세트의 샘플 속도는 16kHz입니다. tf.audio.decode_wav 는 값을 [-1.0, 1.0] 범위로 정규화합니다.
def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

#각 WAV 파일의 레이블은 상위 디렉토리입니다.
def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

#WAV 파일의 파일 이름을 가져와 감독 교육을위한 오디오 및 레이블이 포함 된 튜플을 출력하는 방법을 정의 해 보겠습니다.
def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label


# 이제 process_path 를 적용하여 오디오 레이블 쌍을 추출하고 결과를 확인하는 훈련 세트를 빌드합니다. 나중에 유사한 절차를 사용하여 유효성 검사 및 테스트 세트를 빌드합니다.
AUTOTUNE = tf.data.experimental.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)


# 스펙트로 그램
# 파형을 스펙트로 그램으로 변환하여 시간에 따른 주파수 변화를 보여주고 2D 이미지로 표현할 수 있습니다. 이는 단시간 푸리에 변환 (STFT)을 적용하여 오디오를 시간-주파수 도메인으로 변환함으로써 수행 할 수 있습니다.
# 푸리에 변환 ( tf.signal.fft )은 신호를 구성 주파수로 변환하지만 모든 시간 정보를 잃습니다. STFT ( tf.signal.stft )는 신호를 시간 창으로 분할하고 각 창에서 푸리에 변환을 실행하여 일부 시간 정보를 보존하고 표준 컨볼 루션을 실행할 수있는 2D 텐서를 반환합니다.
# STFT는 크기와 위상을 나타내는 복소수의 배열을 생성합니다. 그러나 tf.abs 의 출력에 tf.signal.stft 적용하여 파생 될 수있는이 자습서의 크기 만 필요합니다.
# 생성 된 스펙트로 그램 "이미지"가 거의 정사각형이되도록 frame_length 및 frame_step 매개 변수를 선택합니다. STFT 매개 변수 선택에 대한 자세한 내용은 오디오 신호 처리에 대한 이 비디오 를 참조하십시오.
# 또한 파형의 길이가 같으면 스펙트로 그램 이미지로 변환 할 때 결과가 비슷한 치수를 갖게됩니다. 이것은 1 초보다 짧은 오디오 클립을 0으로 채우기 만하면됩니다.


def get_spectrogram(waveform):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform)+6592, dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

#다음으로 데이터를 탐색합니다. 파형, 스펙트로 그램 및 데이터 세트에서 한 예의 실제 오디오를 비교합니다.



def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)


for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

model = models.Sequential([
    layers.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')


sample_file = 'test.wav'

sample_ds = preprocess_dataset([str(sample_file)])

for spectrogram, label in sample_ds.batch(1):
  prediction = model(spectrogram)
  plt.bar(commands, tf.nn.softmax(prediction[0]))
  plt.title(f'Predictions for "{commands[label[0]]}"')
  plt.show()