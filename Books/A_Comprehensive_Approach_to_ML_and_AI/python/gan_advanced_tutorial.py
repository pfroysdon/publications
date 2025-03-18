#!/usr/bin/env python3
"""
ganTutorial_advanced.py
-----------------------
An advanced GAN example using Keras on the MNIST dataset.
The generator and discriminator are multi-layer networks.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load MNIST data and normalize to [-1,1]
(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
img_shape = x_train.shape[1:]  # (28,28)
x_train = x_train.reshape(-1, 28*28)

z_dim = 100

# Build Generator
def build_generator():
    noise = Input(shape=(z_dim,))
    x = Dense(256, activation='linear')(noise)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='linear')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation='linear')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    img = Dense(28*28, activation='tanh')(x)
    img = Reshape((28,28,1))(img)
    return Model(noise, img)

# Build Discriminator
def build_discriminator():
    img = Input(shape=(28,28,1))
    x = Reshape((28*28,))(img)
    x = Dense(512, activation='linear')(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(256, activation='linear')(x)
    x = LeakyReLU(0.2)(x)
    validity = Dense(1, activation='sigmoid')(x)
    return Model(img, validity)

optimizer = Adam(0.0002, 0.5)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
generator = build_generator()

# GAN model: generator trains to fool discriminator
z = Input(shape=(z_dim,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

epochs = 5000
batch_size = 64

valid_labels = np.ones((batch_size,1))
fake_labels = np.zeros((batch_size,1))

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    imgs = x_train[idx]
    imgs = imgs.reshape(batch_size, 28, 28, 1)
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    gen_imgs = generator.predict(noise)
    
    d_loss_real = discriminator.train_on_batch(imgs, valid_labels)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    noise = np.random.normal(0, 1, (batch_size, z_dim))
    g_loss = combined.train_on_batch(noise, valid_labels)
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%, G loss: {g_loss:.4f}")

# Generate images after training
noise = np.random.normal(0, 1, (4, z_dim))
gen_imgs = generator.predict(noise)
gen_imgs = 0.5 * gen_imgs + 0.5

plt.figure(figsize=(10,3))
plt.subplot(1,5,1)
plt.imshow(x_train[0].reshape(28,28), cmap='gray')
plt.title("Real")
plt.axis('off')
for i in range(4):
    plt.subplot(1,5,i+2)
    plt.imshow(gen_imgs[i].reshape(28,28), cmap='gray')
    plt.title(f"Gen {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.show()
