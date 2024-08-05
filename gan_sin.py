# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2024-06-28
# @Author  : white
# @FileName: gan_circle.py
# @Software: PyCharm
# **************************************
import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, binary_accuracy, save_weights
import numpy as np
import time

"""
基于gan网络的sin 正弦函数图像绘制
"""


def get_real_data(data_dim, batch_size):
    """
    获取真实数据
    正弦函数公式：sin(x)
    """
    for i in range(300):
        x = np.linspace(-4, 4, data_dim)[np.newaxis, :].repeat(batch_size, axis=0)
        yield np.sin(x)


class GAN(keras.Model):
    def __init__(self, latent_dim, data_dim):
        """
        latent_dim: 输入数据的维度个数大小
        data_dim: 输出数据维度大小
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.loss_func = keras.losses.BinaryCrossentropy(from_logits=True)
        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)

    def call(self, n, training=False, mask=None):
        return self.g.call(tf.random.normal((n, self.latent_dim)), training=training)

    def _get_generator(self):
        """
        生成器模型
        """
        model = keras.Sequential([
            keras.Input([None, self.latent_dim]),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(self.data_dim)
        ], name="generator")
        model.summary()
        return model

    def _get_discriminator(self):
        """
        判别器模型
        """
        model = keras.Sequential([
            keras.Input([None, self.data_dim]),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(1)
        ])
        model.summary()
        return model

    def train_g(self, d_label):
        with tf.GradientTape() as tape:
            g_data = self.call(len(d_label), training=True)
            pred = self.d.call(g_data, training=False)
            loss = self.loss_func(d_label, pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_data, binary_accuracy(d_label, pred)

    def train_d(self, data, label):
        with tf.GradientTape() as tape:
            pred = self.d.call(data, training=True)
            loss = self.loss_func(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(label, pred)

    def step(self, data):
        d_label = tf.ones((len(data) * 2, 1), tf.float32)
        g_loss, g_data, g_acc = self.train_g(d_label)
        d_label = tf.concat((tf.ones((len(data), 1), tf.float32), tf.zeros((len(g_data) // 2, 1), tf.float32)), axis=0)
        data = tf.concat((data, g_data[:len(g_data) // 2]), axis=0)
        d_loss, d_acc = self.train_d(data, d_label)
        return d_loss, d_acc, g_loss, g_acc


def train(gan, epoch):
    t0 = time.time()
    for ep in range(epoch):
        for t, data in enumerate(get_real_data(DATA_DIM, BATCH_SIZE)):
            d_loss, d_acc, g_loss, g_acc = gan.step(data)
            if t % 400 == 0:
                t1 = time.time()
                print(
                    "ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".
                    format(ep, t1 - t0, t, d_acc.numpy(), g_acc.numpy(), d_loss.numpy(), g_loss.numpy(), )
                )
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan, shrink=2)


if __name__ == '__main__':
    LATENT_DIM = 10
    DATA_DIM = 16
    BATCH_SIZE = 32
    EPOCH = 40

    set_soft_gpu(True)
    m = GAN(LATENT_DIM, DATA_DIM)
    train(m, EPOCH)
