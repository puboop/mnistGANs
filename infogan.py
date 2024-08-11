# [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/pdf/1606.03657.pdf)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from visual import save_gan, cvt_gif
from tensorflow.keras.layers import Dense, Input, BatchNormalization, LeakyReLU, Dropout
from utils import set_soft_gpu, binary_accuracy, save_weights, class_accuracy
from mnist_ds import get_half_batch_ds
from gan_cnn import mnist_uni_disc_cnn, mnist_uni_gen_cnn
import time


class InfoGAN(keras.Model):
    """
    discriminator 图片 预测 真假
    q net 图片 预测 c  (c可以理解为 虚拟类别 或 虚拟风格)
    generator z&c 生成 图片
    """

    def __init__(self, rand_dim, style_dim, label_dim, img_shape, fix_std=True, style_scale=2):
        super().__init__()
        self.rand_dim, self.style_dim, self.label_dim = rand_dim, style_dim, label_dim
        self.img_shape = img_shape
        self.fix_std = fix_std
        self.style_scale = style_scale

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_bool = keras.losses.BinaryCrossentropy(from_logits=True, reduction="none")

    def call(self, img_info, training=None, mask=None):
        img_label, img_style = img_info
        noise = tf.random.normal((len(img_label), self.rand_dim))
        if isinstance(img_label, np.ndarray):
            img_label = tf.convert_to_tensor(img_label, dtype=tf.int32)
        if isinstance(img_style, np.ndarray):
            img_style = tf.convert_to_tensor(img_style, dtype=tf.float32)
        return self.g.call([noise, img_label, img_style], training=training)

    def _get_discriminator(self):
        img = Input(shape=self.img_shape)
        s = keras.Sequential([
            mnist_uni_disc_cnn(self.img_shape),
            Dense(32),
            # 批归一化，有助于稳定训练。
            BatchNormalization(),
            # 激活函数，带有负斜率，以避免“死亡 ReLU”问题。
            LeakyReLU(),
            Dropout(0.5),
        ])
        # 变量用于确定样式维度的大小。如果 self.fix_std 为 True，style_dim 保持不变，否则扩大到原来的两倍。
        style_dim = self.style_dim if self.fix_std else self.style_dim * 2
        # q 网络用于将特征映射到样式和标签预测。首先通过全连接层 Dense(16)，然后进行批归一化和激活，最后输出样式和标签。
        q = keras.Sequential([
            Dense(16, input_shape=(32,)),
            BatchNormalization(),
            LeakyReLU(),
            Dense(style_dim + self.label_dim)
        ], name="recognition")
        # 是卷积网络的输出 s(img) 相当于 s.call(img)
        o = s(img)
        # 是二分类输出，表示图像的真假（使用 Dense(1)）。
        o_bool = Dense(1)(o)
        # 是样式和标签的预测结果。
        o_q = q(o)
        if self.fix_std:
            # 样式部分映射到 [0, 1] 区间（通过 tf.tanh 和 self.style_scale）。
            q_style = self.style_scale * tf.tanh(o_q[:, :style_dim])
        else:
            # 样式部分被分为两部分处理，一部分经过 tanh，另一部分经过 relu，然后合并。
            q_style = tf.concat(
                (
                    self.style_scale * tf.tanh(o_q[:, :style_dim // 2]),
                    tf.nn.relu(o_q[:, style_dim // 2:style_dim])
                ),
                axis=1)
        # q_label 提取了 o_q 中最后的 self.label_dim 个值，表示对标签的预测。
        q_label = o_q[:, -self.label_dim:]
        # 创建了一个 Keras 模型，其中包括图像输入和三个输出：真假判断 (o_bool)、样式 (q_style)、和标签 (q_label)。
        model = keras.Model(img, [o_bool, q_style, q_label], name="discriminator")
        model.summary()
        return model

    def _get_generator(self):
        latent_dim = self.rand_dim + self.label_dim + self.style_dim
        noise = Input(shape=(self.rand_dim,))
        style = Input(shape=(self.style_dim,))
        label = Input(shape=(), dtype=tf.int32)
        label_onehot = tf.one_hot(label, depth=self.label_dim)
        model_in = tf.concat((noise, label_onehot, style), axis=1)
        s = mnist_uni_gen_cnn((latent_dim,))
        o = s(model_in)
        model = keras.Model([noise, label, style], o, name="generator")
        model.summary()
        return model

    def loss_mutual_info(self, style, pred_style, label, pred_label):
        # 分类损失（categorical_loss）：使用 sparse_categorical_crossentropy 计算标签的分类损失。
        # label 是真实标签，pred_label 是判别器对标签的预测。
        # from_logits=True 表示 pred_label 是原始的 logits，而不是经过 softmax 的概率分布。
        categorical_loss = keras.losses.sparse_categorical_crossentropy(label, pred_label, from_logits=True)
        if self.fix_std:
            # 固定标准差：如果 self.fix_std 为 True，则 style_std 被设置为 1。style_mean 为 pred_style。
            style_mean = pred_style
            style_std = tf.ones_like(pred_style)  # fixed std
        else:
            # 可变标准差：如果 self.fix_std 为 False，则 pred_style 被分为均值和对数标准差两部分。
            # 对数标准差通过 tf.exp 转换为实际标准差，并取平方根。
            split = pred_style.shape[1] // 2
            style_mean, style_std = pred_style[:split], pred_style[split:]
            style_std = tf.sqrt(tf.exp(style_std))
        # epsilon 是样式的标准化值。通过减去均值并除以标准差（加上一个小的正值 1e-5 以避免除零错误）来计算。
        epsilon = (style - style_mean) / (style_std + 1e-5)
        # 计算样式的对数似然。
        ll_continuous = tf.reduce_sum(
            # style_std + 1e-5 防止对数计算中出现负值。
            - 0.5 * tf.math.log(2 * np.pi) - tf.math.log(style_std + 1e-5) - 0.5 * tf.square(epsilon),
            axis=1,
        )
        # 总损失是分类损失减去对数似然。
        # 分类损失鼓励生成器生成符合真实标签的样本，对数似然项鼓励生成器生成能够最大化样式信息的样本。
        loss = categorical_loss - ll_continuous
        return loss

    def train_d(self, real_fake_img, real_fake_d_label, fake_img_label, fake_style):
        with tf.GradientTape() as tape:
            pred_bool, pred_style, pred_class = self.d.call(real_fake_img, training=True)
            info_split = len(real_fake_d_label)
            real_fake_pred_bool = pred_bool[:info_split]
            loss_bool = self.loss_bool(real_fake_d_label, real_fake_pred_bool)
            fake_pred_style = pred_style[-info_split:]
            fake_pred_label = pred_class[-info_split:]
            loss_info = self.loss_mutual_info(fake_style, fake_pred_style, fake_img_label, fake_pred_label)
            loss = tf.reduce_mean(loss_bool + LAMBDA * loss_info)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss \
            , binary_accuracy(real_fake_d_label, real_fake_pred_bool) \
            , class_accuracy(fake_img_label, fake_pred_label)

    def train_g(self, random_img_label, random_img_style):
        d_label = tf.ones((len(random_img_label), 1), tf.float32)  # let d think generated images are real
        with tf.GradientTape() as tape:
            g_img = self.call([random_img_label, random_img_style], training=True)
            pred_bool, pred_style, pred_class = self.d.call(g_img, training=False)
            loss_bool = self.loss_bool(d_label, pred_bool)
            loss_info = self.loss_mutual_info(random_img_style, pred_style, random_img_label, pred_class)
            loss = tf.reduce_mean(loss_bool + LAMBDA * loss_info)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img, binary_accuracy(d_label, pred_bool)

    def step(self, real_img):
        random_img_label = tf.convert_to_tensor(np.random.randint(0, 10, len(real_img) * 2), dtype=tf.int32)
        random_img_style = tf.random.uniform((len(real_img) * 2, self.style_dim), -self.style_scale, self.style_scale)
        g_loss, g_img, g_bool_acc = self.train_g(random_img_label, random_img_style)

        real_fake_img = tf.concat((real_img, g_img), axis=0)  # 32+64
        real_fake_d_label = tf.concat(  # 32+32
            (
                tf.ones((len(real_img), 1), tf.float32)
                , tf.zeros((len(g_img) // 2, 1), tf.float32)
            )
            , axis=0
        )
        d_loss, d_bool_acc, d_class_acc = self.train_d(real_fake_img
                                                       , real_fake_d_label
                                                       , random_img_label
                                                       , random_img_style)
        return d_loss, d_bool_acc, g_loss, g_bool_acc, random_img_label, d_class_acc


def train(gan, ds):
    t0 = time.time()
    for ep in range(EPOCH):
        for t, (real_img, _) in enumerate(ds):
            d_loss, d_bool_acc, g_loss, g_bool_acc, g_img_label, d_class_acc = gan.step(real_img)
            if t % 400 == 0:
                t1 = time.time()
                print("ep={} | time={:.1f}|t={}|d_acc={:.2f}|d_classacc={:.2f}|g_acc={:.2f}|d_loss={:.2f}|g_loss={:.2f}"
                      .format(ep
                              , t1 - t0
                              , t
                              , d_bool_acc.numpy()
                              , g_bool_acc.numpy()
                              , d_class_acc.numpy()
                              , d_loss.numpy()
                              , g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    STYLE_DIM = 2
    LABEL_DIM = 10
    RAND_DIM = 8
    LAMBDA = 1
    IMG_SHAPE = (28, 28, 1)
    FIX_STD = True
    STYLE_SCALE = 1
    BATCH_SIZE = 64
    EPOCH = 40

    set_soft_gpu(True)
    d = get_half_batch_ds(BATCH_SIZE)
    m = InfoGAN(RAND_DIM, STYLE_DIM, LABEL_DIM, IMG_SHAPE, FIX_STD, STYLE_SCALE)
    train(m, d)
