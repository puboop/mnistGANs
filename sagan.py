# [Self-Attention Generative Adversarial Networks](https://arxiv.org/pdf/1805.08318.pdf)

import tensorflow as tf
from tensorflow import keras
from utils import set_soft_gpu, save_weights
from visual import save_gan, cvt_gif
from mnist_ds import get_half_batch_ds
import time

"""
自定义 tf.keras.layers.Layer 类可以让你创建具有特定功能的自定义层。
这种方法可以用于实现复杂的层操作或在模型中加入新的功能。
下面是创建自定义 Keras 层的一般步骤：

步骤概述
继承 tf.keras.layers.Layer：创建一个新的类并继承 tf.keras.layers.Layer。
实现 __init__ 方法：在这个方法中定义层的可训练参数和其他初始化操作。
实现 build 方法：在这个方法中创建层的权重。这只会在第一次调用层时执行。
实现 call 方法：定义前向传播的逻辑，即如何将输入数据通过这个层进行处理。
实现 compute_output_shape 方法（可选）：指定输出形状的计算方式（对于 Keras 2.0+ 版本，通常可以省略，因为 TensorFlow 可以自动推断）。
实现 get_config 方法（可选）：为了层的序列化和反序列化，可以实现这个方法返回层的配置。
"""


class Attention(keras.layers.Layer):
    def __init__(self, gamma=0.01, trainable=True):
        super().__init__(trainable=trainable)
        self._gamma = gamma
        self.gamma = None
        self.f = None
        self.g = None
        self.h = None
        self.v = None
        self.attention = None

    def build(self, input_shape):
        # input_shape 是输入张量的形状。input_shape[-1] 通常是输入特征图的通道数（即深度）。
        c = input_shape[-1]

        # self.block(c//8) 是一个函数（或方法），它应该返回一个由卷积层或其他操作构成的层块。
        # 这个函数在您的自定义层中用于创建多个网络组件，每个组件的通道数是 c // 8，目的是减少计算量。
        self.f = self.block(c // 8)  # reduce channel size, reduce computation
        self.g = self.block(c // 8)  # reduce channel size, reduce computation
        self.h = self.block(c // 8)  # reduce channel size, reduce computation

        # self.v 是一个卷积层，它将通道数从减少后的 c // 8 恢复到原始的 c。这个卷积层的核大小为1x1，步幅为1。
        # 1x1卷积用于调整通道数而不改变空间维度
        self.v = keras.layers.Conv2D(c, 1, 1)  # scale back to original channel size
        # GAMMA_id 是一个全局变量，用于确保每次创建新层时 gamma 权重的名称是唯一的。
        # 每创建一个新层，GAMMA_id 就会递增
        global GAMMA_id
        # self.gamma 是一个可训练的权重，通过 add_weight 方法添加到层中。
        # self.gamma 使用一个全局计数器 GAMMA_id 作为名称的一部分，以确保每个 gamma 权重的名称唯一。
        # self._gamma 是一个初始化值，这可能是一个常数或其他预设值。
        self.gamma = self.add_weight(
            "gamma{}".format(GAMMA_id), shape=None, initializer=keras.initializers.constant(self._gamma))
        GAMMA_id += 1

    @staticmethod
    def block(c):
        """
        block 方法创建了一个卷积块，通常用于以下场景：
        通道调整：1x1 卷积用于改变特征图的通道数，这在处理通道数变换时非常有用。
        特征展平：通过 Reshape 层，将空间维度展平，
            通常用于将卷积特征图的空间维度展平为序列形式，以便进行进一步处理（例如，注意力机制中的加权操作
        """
        return keras.Sequential([
            # 这是一个 1x1 的卷积层，用于将输入张量的通道数变为 c。这里的参数说明：
            # c：输出的通道数。
            # 1：卷积核的宽度。
            # 1：卷积核的高度。
            # 1x1 卷积通常用于调整通道数而不改变空间尺寸。
            keras.layers.Conv2D(c, 1, 1),  # [n, w, h, c] 1*1conv
            # Reshape 层用于重新调整张量的形状。
            # 在这里，它将输入张量从形状 [n, w, h, c]（即批量大小、高度、宽度和通道数）转换为形状 [n, w*h, c]（即批量大小、空间展平后的维度和通道数）。
            # 这个操作通常用于将特征图展平，以便后续的操作能够处理这些特征。
            keras.layers.Reshape((-1, c)),  # [n, w*h, c]
        ])

    def call(self, inputs, **kwargs):
        # f = self.f(inputs): 将输入数据通过卷积块 self.f 处理，得到形状为 [n, w*h, c//8] 的张量。这里 f 是用于计算注意力权重的特征。
        f = self.f(inputs)  # [n, w, h, c] -> [n, w*h, c//8]
        # g = self.g(inputs): 将输入数据通过卷积块 self.g 处理，得到形状为 [n, w*h, c//8] 的张量。这里 g 是用于计算注意力矩阵的特征。
        g = self.g(inputs)  # [n, w, h, c] -> [n, w*h, c//8]
        # h = self.h(inputs): 将输入数据通过卷积块 self.h 处理，得到形状为 [n, w*h, c//8] 的张量。这里 h 是用于计算上下文特征的特征。
        h = self.h(inputs)  # [n, w, h, c] -> [n, w*h, c//8]

        # s = tf.matmul(f, g, transpose_b=True): 计算 f 和 g 的点积，得到注意力矩阵 s。这里 s 的形状为 [n, w*h, w*h]，表示每个空间位置与其他位置的相关性。
        s = tf.matmul(f, g, transpose_b=True)  # [n, w*h, c//8] @ [n, c//8, w*h] = [n, w*h, w*h]
        # self.attention = tf.nn.softmax(s, axis=-1): 对注意力矩阵 s 应用 softmax，得到注意力权重 self.attention。这样可以为每个位置分配一个权重，表示其对其他位置的影响。
        self.attention = tf.nn.softmax(s, axis=-1)
        # context_wh = tf.matmul(self.attention, h): 使用注意力权重对 h 进行加权求和，得到上下文特征 context_wh。context_wh 的形状为 [n, w*h, c//8]，表示每个位置的上下文特征。
        context_wh = tf.matmul(self.attention, h)  # [n, w*h, w*h] @ [n, w*h, c//8] = [n, w*h, c//8]

        s = inputs.shape  # [n, w, h, c]
        cs = context_wh.shape  # [n, w*h, c//8]
        # context = tf.reshape(context_wh, [-1, s[1], s[2], cs[-1]]): 将 context_wh 重新调整形状为 [n, w, h, c//8]，以便与原始输入形状匹配。
        context = tf.reshape(context_wh, [-1, s[1], s[2], cs[-1]])  # [n, w, h, c//8]
        # o = self.v(self.gamma * context) + inputs: 使用卷积层 self.v 处理上下文特征 context，并与原始输入 inputs 相加，实现残差连接。self.gamma 是一个可训练的权重，用于调节上下文特征的贡献。
        o = self.v(self.gamma * context) + inputs  # residual
        return o


class SAGAN(keras.Model):
    """
    自注意力加强生成器能力,使用常用在SVM中的 hinge loss, 连续性loss.

    因为注意力的矩阵很大(w*h @ w*h), 所以训练起来比较慢, 意味着留有改动空间.
    里面的稳定W gradient的Spectral normalization（SN）写起来有点麻烦,
    我有空再考虑把这个 SN regularizer 写进来.
    """

    def __init__(self, latent_dim, img_shape, gamma):
        super().__init__()
        self.gamma = gamma
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.g = self._get_generator()
        self.d = self._get_discriminator()
        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_func = keras.losses.Hinge()  # change loss to hinge based on the paper

    def call(self, n, training=None, mask=None):
        return self.g.call(tf.random.normal((n, self.latent_dim)), training=training)

    def _get_discriminator(self):
        model = keras.Sequential([
            keras.layers.GaussianNoise(0.01, input_shape=self.img_shape),
            keras.layers.Conv2D(16, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            Attention(self.gamma),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(32, 4, strides=2, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(0.3),

            keras.layers.Flatten(),
            keras.layers.Dense(1),
        ], name="discriminator")
        model.summary()
        return model

    def _get_generator(self):
        model = keras.Sequential([
            # [n, latent] -> [n, 7 * 7 * 128] -> [n, 7, 7, 128]
            keras.layers.Dense(7 * 7 * 128, input_shape=(self.latent_dim,)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Reshape((7, 7, 128)),

            # -> [n, 14, 14, 64]
            keras.layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            Attention(self.gamma),

            # -> [n, 28, 28, 32]
            keras.layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            Attention(self.gamma),
            # -> [n, 28, 28, 1]
            keras.layers.Conv2D(1, (4, 4), padding='same', activation=keras.activations.tanh)
        ], name="generator")
        model.summary()
        return model

    def train_d(self, img, d_label):
        with tf.GradientTape() as tape:
            pred = self.d.call(img, training=True)
            loss = self.loss_func(d_label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss

    def train_g(self, d_label):
        with tf.GradientTape() as tape:
            g_img = self.call(len(d_label), training=True)
            pred = self.d.call(g_img, training=False)
            loss = self.loss_func(d_label, pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img

    def step(self, img):
        d_label = 2 * tf.ones((len(img) * 2, 1), tf.float32)  # a stronger positive label?
        g_loss, g_img = self.train_g(d_label)

        d_label = tf.concat((tf.ones((len(img), 1), tf.float32), -tf.ones((len(g_img) // 2, 1), tf.float32)), axis=0)
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
    GAMMA_id = 0
    LATENT_DIM = 100
    IMG_SHAPE = (28, 28, 1)
    BATCH_SIZE = 64
    GAMMA = 0.01
    EPOCH = 20

    set_soft_gpu(True)
    d = get_half_batch_ds(BATCH_SIZE)
    m = SAGAN(LATENT_DIM, IMG_SHAPE, GAMMA)
    train(m, d, EPOCH)
