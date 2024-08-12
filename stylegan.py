# [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)

import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, save_weights
from mnist_ds import get_half_batch_ds
from gan_cnn import mnist_uni_disc_cnn
import time
import numpy as np
import tensorflow.keras.initializers as initer


class AdaNorm(keras.layers.Layer):
    """
    用于将特征图拉回原来的正态分布
    """
    def __init__(self, axis=(1, 2), epsilon=1e-5):
        super().__init__()
        # NHWC
        self.axis = axis
        self.epsilon = epsilon

    def call(self, x, **kwargs):
        mean = tf.math.reduce_mean(x, axis=self.axis, keepdims=True)
        diff = x - mean
        variance = tf.reduce_mean(tf.math.square(diff), axis=self.axis, keepdims=True)
        x_norm = diff * tf.math.rsqrt(variance + self.epsilon)
        return x_norm


class AdaMod(keras.layers.Layer):
    """
    把 Mapping net 出来的 w 加工成两个 y 用来控制特征图 x 的缩放平移变化。 所有在这里我们需需要将 w 通过举证运算，转化成 y 使之可以匹配特征图的格式。
    """
    def __init__(self):
        super().__init__()
        self.y = None

    def call(self, inputs, **kwargs):
        x, w = inputs
        y = self.y(w)
        o = (y[:, 0] + 1) * x + y[:, 1]
        return o

    def build(self, input_shape):
        x_shape, w_shape = input_shape
        self.y = keras.Sequential([keras.layers.Dense(x_shape[-1] * 2,
                                                      input_shape=w_shape[1:],
                                                      name="y",
                                                      kernel_initializer=initer.RandomNormal(0, 1)),
                                   # this kernel is important
                                   keras.layers.Reshape([2, 1, 1, -1])])  # [2, h, w, c]


class AddNoise(keras.layers.Layer):
    """
    自定义模型层，添加噪声
    添加随机噪点的方式，这里其实还可以有其他方式去添加，
    我为了简化在这里传入的 inputs 的 noise，将它设定成每一层 CNN 特征图都可以用的 noise，所以每层CNN传递的noise都一样。
    但是我会对这个 noise 选取不同的长宽大小，来匹配上当前特征图的长宽。
    """

    def __init__(self):
        super().__init__()
        self.s = None
        self.x_shape = None

    def call(self, inputs, **kwargs):
        x, noise = inputs
        noise_ = noise[:, :self.x_shape[1], :self.x_shape[2], :]
        return self.s * noise_ + x

    def build(self, input_shape):
        # 在实际调用层之前调用，指挥被调用一次
        self.x_shape, _ = input_shape
        self.s = self.add_weight(name="noise_scale",
                                 shape=[1, 1, self.x_shape[-1]],  # [h, w, c]
                                 initializer=initer.random_normal(0., 1.))  # large initial noise


class Map(keras.layers.Layer):
    """
    自定义模型map对输入数据进行映射
    """

    def __init__(self, size):
        super().__init__()
        # size：Map 层的一个参数，表示 Dense 层的输出维度。
        self.size = size
        self.f = None

    def call(self, inputs, **kwargs):
        w = self.f(inputs)
        return w

    def build(self, input_shape):
        self.f = keras.Sequential([
            # input_shape：输入张量的形状。input_shape[1:] 提取除 batch size 外的形状部分，因为 Dense 层只关注每个样本的形状。
            keras.layers.Dense(self.size, input_shape=input_shape[1:]),
            # keras.layers.LeakyReLU(0.2),  # worse performance when using non-linearity in mapping
            # 但是有意思的是，如果我在 Mapping net 的全连接层加上激活函数，做一些非线性的话， 生成结果就要差很多。
            # 我不明白其中的道理，因为安理来说，非线性是 mapping 的核心概念，就是要转换空间分布的，可是为什么这个试验中，反而有坏处呢？ 有想法的同学可以在下面讨论一下。
            keras.layers.Dense(self.size),
        ])


class Style(keras.layers.Layer):
    def __init__(self, filters, upsampling=True):
        super().__init__()
        self.filters = filters
        self.upsampling = upsampling
        self.ada_mod, self.ada_norm, self.add_noise, self.up, self.conv = None, None, None, None, None

    def call(self, inputs, **kwargs):
        x, w, noise = inputs
        x = self.ada_mod((x, w))
        if self.up is not None:
            x = self.up(x)
        x = self.conv(x)
        x = self.ada_norm(x)
        x = keras.layers.LeakyReLU()(x)
        x = self.add_noise((x, noise))
        return x

    def build(self, input_shape):
        self.ada_mod = AdaMod()
        self.ada_norm = AdaNorm()
        if self.upsampling:
            # keras.layers.UpSampling2D 是 Keras 中用于上采样的层，它用于在卷积神经网络中增加特征图的空间分辨率。
            # 上采样通常用于生成模型（如 GAN）或语义分割模型中，以提高图像的分辨率。
            # UpSampling2D 的参数和功能
            # size: 上采样因子，即每个维度的放大倍数。例如，(2, 2) 表示在高度和宽度两个维度上都将特征图的尺寸放大 2 倍。
            # interpolation: 插值方法，用于在上采样过程中计算新像素的值。可以选择以下方法：
            # "nearest"：最近邻插值。
            # "bilinear"：双线性插值。
            self.up = keras.layers.UpSampling2D((2, 2), interpolation="bilinear")
        self.add_noise = AddNoise()
        self.conv = keras.layers.Conv2D(self.filters, 3, 1, "same")


class StyleGAN(keras.Model):
    """
    重新定义generator,生成图片
    """

    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.n_style = 3

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.001, beta_1=0.)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, inputs, training=None, mask=None):
        if isinstance(inputs[0], np.ndarray):
            inputs = (tf.convert_to_tensor(i) for i in inputs)
        inputs = [tf.ones((len(inputs[0]), 1)), *inputs]
        return self.g.call(inputs, training=training)

    def _get_generator(self):
        z = keras.Input((self.n_style, self.latent_dim,), name="z")
        noise_ = keras.Input((self.img_shape[0], self.img_shape[1]), name="noise")
        ones = keras.Input((1,), name="ones")

        const = keras.Sequential([
            keras.layers.Dense(7 * 7 * 128, use_bias=False, name="const"),
            # Reshape 层的作用是将输入张量重新组织成指定的形状，这对于构建和调整模型的网络架构非常有用。
            # 它可以将数据从一个形状转换为另一个形状，以便输入到后续的层中。
            keras.layers.Reshape((7, 7, 128)),
        ], name="const")(ones)

        w = Map(size=128)(z)
        # 扩展维度
        noise = tf.expand_dims(noise_, axis=-1)
        x = AddNoise()((const, noise))
        x = AdaNorm()(x)
        x = Style(64, upsampling=False)((x, w[:, 0], noise))  # 7^2
        x = Style(64)((x, w[:, 1], noise))  # 14^2
        x = Style(64)((x, w[:, 2], noise))  # 28^2
        o = keras.layers.Conv2D(self.img_shape[-1], 5, 1, "same", activation=keras.activations.tanh)(x)

        g = keras.Model([ones, z, noise_], o, name="generator")
        g.summary()
        return g

    def _get_discriminator(self):
        model = keras.Sequential([
            mnist_uni_disc_cnn(self.img_shape, use_bn=True),
            keras.layers.Dense(1)
        ], name="discriminator")
        model.summary()
        return model

    def train_d(self, img, label):
        with tf.GradientTape() as tape:
            pred = self.d.call(img, training=True)
            loss = self.loss_bool(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss

    def train_g(self, n):
        # 随机两种不同的噪声
        available_z = [tf.random.normal((n, 1, self.latent_dim)) for _ in range(2)]
        # 根据总样式随机不同的噪声
        z = tf.concat([available_z[np.random.randint(0, len(available_z))] for _ in range(self.n_style)], axis=1)

        noise = tf.random.normal((n, self.img_shape[0], self.img_shape[1]))
        inputs = (z, noise)
        with tf.GradientTape() as tape:
            g_img = self.call(inputs, training=True)
            pred = self.d.call(g_img, training=False)
            loss = self.loss_bool(tf.ones_like(pred), pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img

    def step(self, img):
        g_loss, g_img = self.train_g(len(img) * 2)
        d_label = tf.concat((tf.ones((len(img), 1), tf.float32), tf.zeros((len(g_img) // 2, 1), tf.float32)), axis=0)
        img = tf.concat((img, g_img[:len(g_img) // 2]), axis=0)
        d_loss = self.train_d(img, d_label)
        return d_loss, g_loss


def train(gan, ds, epoch):
    t0 = time.time()
    for ep in range(epoch):
        for t, (img, _) in enumerate(ds):
            d_loss, g_loss = gan.step(img)
            if t % 400 == 0:
                t1 = time.time()
                print(
                    "ep={} | time={:.1f} | t={} | d_loss={:.2f} | g_loss={:.2f}".format(
                        ep, t1 - t0, t, d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    LATENT_DIM = 100
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    d = get_half_batch_ds(BATCH_SIZE)
    m = StyleGAN(LATENT_DIM, IMG_SHAPE)
    train(m, d, EPOCH)
