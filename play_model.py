import tensorflow as tf
from PIL import Image
import PIL.Image

def save_model(model_path, train_path, valid_path):
    # 이미지넷에서 가져온 사전 학습한 가중치는 수정하면 안됨
    conv_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=[150, 150, 3])

    model = tf.keras.models.Sequential()
    model.add(conv_base)                                  # 기존 모델 연결
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # 전체 동결. 이미지넷에서 가져온 사전 학습한 가중치는 수정하면 안됨
    conv_base.trainable = False

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # 학습 데이터 증강
    train_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    # 검증 제너레이터는 이미지 증강하면 안됨.
    valid_img_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

    batch_size = 32
    train_generator = train_img_generator.flow_from_directory(
        train_path,
        target_size=[150, 150],
        batch_size=batch_size,
        class_mode='binary')
    valid_generator = valid_img_generator.flow_from_directory(
        valid_path,
        target_size=[150, 150],
        batch_size=batch_size,
        class_mode='binary')

    # 20회 에포크만큼 학습
    model.fit_generator(train_generator,
                        steps_per_epoch=1000 // batch_size,
                        epochs=20,
                        validation_data=valid_generator,
                        validation_steps=50)

    # hdf5 형식 : Hierarchical Data Format version 5
    model.save(model_path)

def load_model(model_path, test_path):
    model = tf.keras.models.load_model(model_path)

    generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

    test_gen = generator.flow_from_directory(test_path,
                                             target_size=[150, 150],
                                             batch_size=500,
                                             class_mode='binary')

    # 첫 번째 배치 읽기. batch_size가 500이니까 500개 가져옴
    x, y = next(test_gen)
    print('acc :', model.evaluate(x, y))

# 저장한 파일로부터 모델 변환 후 다시 저장
def convert_model(model_path, tflite_path):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
    flat_data = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(flat_data)



# model_path = 'models/cats_and_dogs_small.h5'
# save_model(model_path, 'cats_and_dogs/small/train', 'cats_and_dogs/small/valid')
# load_model(model_path, 'cats_and_dogs/small/test')
# convert_model('models/cats_and_dogs_small.h5', 'models/cats_and_dogs.tflite')

model_path = 'models/men_and_women.h5'
# save_model(model_path, 'men_and_women/small/train', 'men_and_women/small/valid')
# load_model(model_path, 'men_and_women/small/test')
convert_model('models/men_and_women.h5', 'models/men_and_women.tflite')




