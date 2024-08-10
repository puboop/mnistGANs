# [Conditional Generative Adversarial Nets](https://arxiv.org/pdf/1411.1784.pdf)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from visual import save_gan, cvt_gif
from tensorflow.keras.layers import Dense, Reshape, Input, Embedding
from utils import set_soft_gpu, binary_accuracy, save_weights
from mnist_ds import get_half_batch_ds
from gan_cnn import mnist_uni_disc_cnn, mnist_uni_gen_cnn
import time

"""
cgan为根据指定标签来生成指定类型的图片
"""


class CGAN(keras.Model):
    """
    discriminator 标签+图片 预测 真假
    generator 标签 生成 图片
    """

    def __init__(self, latent_dim, label_dim, img_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim = label_dim
        self.img_shape = img_shape

        self.g = self._get_generator()
        self.d = self._get_discriminator()

        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        self.loss_func = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, target_labels, training=None, mask=None):
        """
        1. 生成shape为（len(target_labels, self.latent_dim）的随机噪点图片
        2.
        """
        noise = tf.random.normal((len(target_labels), self.latent_dim))
        if isinstance(target_labels, np.ndarray):
            target_labels = tf.convert_to_tensor(target_labels, dtype=tf.int32)
        return self.g.call([noise, target_labels], training=training)

    def _get_discriminator(self):
        # 定义了输入图像的形状，这里的 self.img_shape 是输入图像的形状，例如 [28, 28, 1]。
        img = Input(shape=self.img_shape)
        # 定义了输入标签的形状（标量），并指定了数据类型为 tf.int32。标签用于对图像进行条件生成。
        label = Input(shape=(), dtype=tf.int32)
        """
        Input 是用于定义模型的输入张量的类。它并不执行任何计算，只是一个占位符，指定数据的形状和类型，以便将数据传递给模型的其他层。Input 是模型架构的入口点，用于连接模型的输入数据和后续层。
        特点
        定义输入形状：
            Input 用于定义输入数据的形状，例如图像的尺寸或序列的长度。
            它的形状必须与实际数据相匹配，但不包括批量大小（batch size）。
        创建张量：
            Input 创建一个张量对象，模型的其他层将使用这个张量作为输入。
        主要用途：
            定义模型的输入，作为模型架构的起点。
            与 Layer 对象连接以构建网络。
        
        Layer 是 Keras 中所有层的基类。所有神经网络层（如 Dense, Conv2D, LSTM 等）都继承自 Layer。Layer 定义了数据的变换和计算逻辑，是构建模型的核心组件。
        特点
        执行计算：
            Layer 对象在 call 方法中定义了前向传播的计算逻辑。这个方法描述了如何将输入数据转换为输出数据。
        支持训练和推理：
            Layer 支持训练过程中的权重更新和前向传播计算。在训练过程中，Layer 可以包括可训练的权重，并计算梯度进行反向传播。
        主要用途：
            执行特定的计算或变换。
            定义神经网络的各层，处理输入数据并生成输出。
        
        Input：定义模型的输入张量，不执行任何计算。它是模型的入口点。
        Layer：执行实际的计算和变换，是神经网络的组成部分。包括所有网络层，例如 Dense, Conv2D, LSTM 等
        """

        # 将标签通过嵌入层转换为 32 维的向量。假设有 10 个类别，每个类别的嵌入维度为 32。
        label_emb = Embedding(10, 32)(label)
        # 首先，将嵌入向量通过 Dense(28*28, activation=keras.activations.relu) 转换为形状为 [batch_size, 28*28] 的向量。
        # 然后，通过 Reshape((28, 28, 1)) 将该向量重塑为 28x28 的单通道图像。
        emb_img = Reshape((28, 28, 1))(Dense(28 * 28, activation=keras.activations.relu)(label_emb))
        # 将输入图像和嵌入图像在通道维度（axis=3）上连接。这样可以将图像信息和标签信息结合起来作为判别器的输入。
        concat_img = tf.concat((img, emb_img), axis=3)

        # mnist_uni_disc_cnn(input_shape=[28, 28, 2])：假设这是一个自定义的卷积神经网络模型，用于处理 [28, 28, 2] 的输入图像。
        # mnist_uni_disc_cnn 可能是一个函数，返回一个包含若干卷积层的 Sequential 模型。
        # Dense(1)：输出一个单一的值，通常用于输出判别器对图像的真伪判断。
        s = keras.Sequential([
            mnist_uni_disc_cnn(input_shape=[28, 28, 2]),
            Dense(1)
        ])
        o = s(concat_img)
        # 创建一个 Keras 模型，其中 [img, label] 是输入，o 是输出。name="discriminator" 为模型命名为 “discriminator”。
        model = keras.Model([img, label], o, name="discriminator")

        """
        keras.Sequential 是一种简单的模型类型，适用于线性堆叠的层模型，即每一层的输入是前一层的输出。适合于简单的前馈网络和一些标准的神经网络架构。
        特点
            线性堆叠：Sequential 模型按顺序逐层堆叠，适合简单的网络结构，其中每一层的输出都直接连接到下一层的输入。
            简化模型构建：非常适合创建没有复杂分支或共享层的模型。构建和管理模型的过程简单直观。
            限制：Sequential 模型不支持多输入、多输出、跳跃连接、共享层等复杂网络结构。
        
        keras.Model 是一种更灵活的模型类型，允许用户创建具有复杂结构的模型，比如多个输入或输出、层的共享、复杂的分支和连接等。
        特点
            灵活性：Model 类可以处理具有多个输入和输出的复杂模型架构，例如条件生成对抗网络（Conditional GANs）、多任务学习模型等。
            自定义层和模型：允许定义自定义层，并在 call 方法中实现复杂的前向传播逻辑。可以用在需要更复杂模型结构或特定计算的场景中。
            灵活配置：可以创建复杂的模型，例如需要实现多个输入和输出的模型、带有跳跃连接的网络等。这使得 Model 类非常适合实现复杂的神经网络架构。
        """
        model.summary()
        return model

    def _get_generator(self):
        # 定义生成器的输入 noise，它是一个形状为 (self.latent_dim,) 的向量，通常是随机噪声，用于生成样本。
        noise = Input(shape=(self.latent_dim,))
        # 定义生成器的输入 label，它是一个标量（整数），表示类别标签。数据类型为 tf.int32。
        label = Input(shape=(), dtype=tf.int32)

        # 将整数标签转换为 one-hot 编码的向量。depth=self.label_dim 表示类别的总数。
        # 例如，如果 self.label_dim 为 10，则 label_onehot 将是一个长度为 10 的向量。shape=(None, 10)
        label_onehot = tf.one_hot(label, depth=self.label_dim)
        # 将噪声向量和 one-hot 编码的标签在特征维度（axis=1）上拼接，生成模型的最终输入。这个输入包含了生成图像所需的噪声和标签信息。
        model_in = tf.concat((noise, label_onehot), axis=1)
        # mnist_uni_gen_cnn 应该是一个函数或类，它返回一个处理拼接后输入的生成器网络。
        # 该网络接收形状为 (self.latent_dim + self.label_dim,) 的输入，并生成一个输出。
        s = mnist_uni_gen_cnn((self.latent_dim + self.label_dim,))
        # 将拼接后的输入传递给生成器网络，得到生成的图像 o。
        o = s(model_in)
        # 创建一个 Keras 模型，将 noise 和 label 作为输入，o 作为输出。模型的名称为 "generator"。
        model = keras.Model([noise, label], o, name="generator")
        model.summary()
        return model

    def train_d(self, img, img_label, label):
        """
        img: 输入的图片
        img_label: 每张图片的具体类型
        label: 真实数据与生成数据
        """
        with tf.GradientTape() as tape:
            pred = self.d.call([img, img_label], training=True)
            loss = self.loss_func(label, pred)
        grads = tape.gradient(loss, self.d.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(label, pred)

    def train_g(self, random_img_label):
        # random_img_label随机图片的标签
        d_label = tf.ones((len(random_img_label), 1), tf.float32)  # let d think generated images are real
        with tf.GradientTape() as tape:
            g_img = self.call(random_img_label, training=True)
            pred = self.d.call([g_img, random_img_label], training=False)
            loss = self.loss_func(d_label, pred)
        grads = tape.gradient(loss, self.g.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_img, binary_accuracy(d_label, pred)

    def step(self, real_img, real_img_label):
        random_img_label = tf.convert_to_tensor(np.random.randint(0, 10, len(real_img) * 2), dtype=tf.int32)
        # 训练生成器
        g_loss, g_img, g_acc = self.train_g(random_img_label)

        # g_img在第一个维度上剪掉一般的数据，将其连接到真实数据上
        img = tf.concat((real_img, g_img[:len(g_img) // 2]), axis=0)
        # random_img_label随机标签，链接真实标签
        img_label = tf.concat((real_img_label, random_img_label[:len(g_img) // 2]), axis=0)
        d_label = tf.concat(
            (
                tf.ones((len(real_img_label), 1), tf.float32),
                tf.zeros((len(g_img) // 2, 1), tf.float32)
            ),
            axis=0)
        d_loss, d_acc = self.train_d(img, img_label, d_label)
        return g_img, d_loss, d_acc, g_loss, g_acc, random_img_label


def train(gan, ds):
    t0 = time.time()
    for ep in range(EPOCH):
        for t, (real_img, real_img_label) in enumerate(ds):
            g_img, d_loss, d_acc, g_loss, g_acc, g_img_label = gan.step(real_img, real_img_label)
            if t % 400 == 0:
                t1 = time.time()
                print("ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}"
                      .format(ep, t1 - t0, t, d_acc.numpy(), g_acc.numpy(), d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan)


if __name__ == "__main__":
    LATENT_DIM = 100
    IMG_SHAPE = (28, 28, 1)
    LABEL_DIM = 10
    BATCH_SIZE = 64
    EPOCH = 20

    set_soft_gpu(True)
    d = get_half_batch_ds(BATCH_SIZE)
    m = CGAN(LATENT_DIM, LABEL_DIM, IMG_SHAPE)
    train(m, d)
