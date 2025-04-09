import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

train_dir = 'L07/dataset/train'
val_dir = 'L07/dataset/val'

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=16, class_mode='binary')
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir, target_size=(224, 224), batch_size=16, class_mode='binary')

base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_gen, epochs=10, validation_data=val_gen)

loss, acc = model.evaluate(val_gen)
print(f"Validation Accuracy: {acc:.4f}")

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend(); plt.title('Accuracy'); plt.show()
