import matplotlib.pyplot as plt
import tensorflow.keras as krs
import numpy
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

nw = 28
nh = 28
num_hide = 98
ep = 40
# загружаем примеры обучения mnist (рукописные цифры)
(trainx, trainy), (testx, testy) = mnist.load_data()
# нормируем от -1 до 1 изображения цифр
all_image = (trainx/255.0-0.5)*1.999

# добавляем дополнительное измерение соответствующее одной цветовой карте
all_image = numpy.expand_dims(all_image, axis=3)
# задаем входной слой экодера высота на ширину на количество карт
encoder_input = krs.layers.Input(shape=(nw,nh,1))

lay = krs.layers.Conv2D(32, (3, 3), strides = (2,2), activation='relu', padding='same')(encoder_input)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(64, (3, 3), strides = (2,2), activation='relu', padding='same')(lay)
# добавляем слой прореживания
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(128, (3, 3), strides = (2,2), activation='relu', padding='same')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(256, (3, 3), strides = (2,2), activation='relu', padding='same')(lay)
# слой который многомерный тензорный слой превращает в плоский вектор
lay = krs.layers.Flatten()(lay)
# выходной кодирующий слой
lay_out_encoder = krs.layers.Dense(num_hide, activation="linear", name='den4')(lay)
# создаем сеть энкодера
encoder = krs.Model(encoder_input, lay_out_encoder)
out_vec = encoder.predict(all_image)

all_out = krs.utils.to_categorical(trainy)
num_classes = 10
classifier_input = krs.layers.Input(shape=(num_hide,))
encoder.trainable = False
lay = krs.layers.Dense(128, activation='relu')(classifier_input)
lay = krs.layers.Dense(256, activation='relu')(lay)
lay = krs.layers.Dense(512, activation='relu')(lay)
lay = krs.layers.Dense(512, activation='relu')(lay)
lay = krs.layers.Dense(256, activation='relu')(lay)
lay = krs.layers.Dense(128, activation='relu')(classifier_input)
lay = krs.layers.Dense(10, activation='softmax')(lay)
classifier_output = krs.layers.Dense(num_classes, activation="softmax", name='den4')(lay)
classificator = krs.Model(classifier_input, classifier_output)
classificator.compile(loss='binary_crossentropy', optimizer=krs.optimizers.Adam(learning_rate = 0.0002),
                      metrics=['accuracy', krs.metrics.Precision(), krs.metrics.Recall()])

classificator.fit(x = out_vec, y = all_out, batch_size = 8000 ,epochs = ep * 3)

test_out_vec = encoder.predict(numpy.expand_dims((testx/255.0-0.5)*1.999, axis=3))
predictions = classificator.predict(test_out_vec)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve((testy == i).astype(int), predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(10, 6))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (class {i}) (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves for each class')
plt.legend(loc="lower right")
plt.show()