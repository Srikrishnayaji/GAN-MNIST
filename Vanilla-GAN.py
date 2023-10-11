import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.img_channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.latent_dim = 100
        optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', metrics=['accuracy'])
        self.generator = self.build_generator()
        z = tf.keras.Input(shape=(self.latent_dim,))
        fake_img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(fake_img)
        self.combined = tf.keras.Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer = optimizer)
    
    def build_generator(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(256, input_dim=self.latent_dim))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(tf.keras.layers.Reshape(self.img_shape))
        noise = tf.keras.Input(shape=(self.latent_dim,))
        img = model(noise)
        return(tf.keras.Model(noise, img))
    
    def build_discriminator(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=self.img_shape))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(256))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        img = tf.keras.Input(shape=self.img_shape)
        validity = model(img)
        return(tf.keras.Model(img, validity))
    
    def train(self, epochs, batch_size = 128, sample_interval = 50):
        (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train/127.5)-1
        x_train = np.expand_dims(x_train, axis=3)
        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))
        for epoch in range(epochs):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            print("Epoch: " + str(epoch) + "disc loss: " + str(d_loss) + "gen_loss: "+ str(g_loss))
            if epoch % sample_interval == 0 or epoch == 9999:
                self.sample_images(epoch)
    
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
        

if __name__ == "__main__":
    gan = GAN()
    gan.train(epochs = 10000, batch_size = 132, sample_interval = 10000)