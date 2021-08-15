import tensorflow as tf 
import os, datetime, time
from sklearn.model_selection import train_test_split
import numpy as np

LAMBDA = 100.0
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv1D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv1DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def get_generator():
    inputs = tf.keras.layers.Input(shape=[32, 2])

    down_stack = [
    downsample(32, 4, apply_batchnorm=False),  # (batch_size, 16, 32)
    downsample(64, 4),  # (batch_size, 8, 64)
    #downsample(128, 4),  # (batch_size, 4, 128)
    #downsample(256, 4),  # (batch_size, 2, 256)
    #downsample(512, 4),  # (batch_size, 1, 512)
    ]

    up_stack = [
    #upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 1024)
    #upsample(256, 4, apply_dropout=True),  # (batch_size, 4, 512)
    #upsample(128, 4, apply_dropout=True),  # (batch_size, 8, 256)
    upsample(64, 4),  # (batch_size, 16, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv1DTranspose(3, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def get_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[32, 2], name='input_signal')
    tar = tf.keras.layers.Input(shape=[32, 3], name='target_signal')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 32, channels*2)

    down1 = downsample(32, 4, False)(x)  # (batch_size, 16, 32)
    #down2 = downsample(64, 4)(down1)  # (batch_size, 8, 64)
    #down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding1D()(down1)  # (batch_size, 15, 64)
    conv = tf.keras.layers.Conv1D(64, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 5, 64)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding1D()(leaky_relu)  # (batch_size, 7, 64)

    last = tf.keras.layers.Conv1D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 3, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)




def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss 


def get_data_pipeline(path_to_trajectories, path_to_observations, test_size=0.2):
    trajectories = np.load(path_to_trajectories).astype(np.float32)
    observations = np.load(path_to_observations).astype(np.float32)
    train_obs, test_obs, train_true, test_true  = train_test_split(observations, trajectories, test_size=test_size)
    #train_dataset = tf.data.Dataset.from_tensor_slices((train_obs, train_true))
    #test_dataset = tf.data.Dataset.from_tensor_slices((test_obs, test_true))
    return [train_obs, train_true], [test_obs, test_true]


class GAN:
    
    def __init__(self, folder, name='trial'):
        self.generator = get_generator()
        self.discriminator = get_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.checkpoint_dir = '{}/saved_models'.format(folder)
        log_dir="{}/logs".format(folder)
        self.summary_writer = tf.summary.create_file_writer(log_dir + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.name = name
    
    def train_step(self, input_signal, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_signal, training=True)
            #print('abc', gen_output.shape)
            disc_real_output = self.discriminator([input_signal, target], training=True)
            #print('def', disc_real_output.shape)
            disc_generated_output = self.discriminator([input_signal, gen_output], training=True)
            #print('ghi', disc_generated_output.shape)

            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//100)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//100)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//100)
            tf.summary.scalar('disc_loss', disc_loss, step=step//100)
        return gen_total_loss, disc_loss
        

    @tf.function
    def fit(self, train_ds, steps):
        start = time.time()
        for step in range(steps):
            input_signal, target = train_ds
            if (step) % 100 == 0:
                if step != 0:
                    print(f'Time taken for 100 steps: {time.time()-start:.2f} sec\n')
                start = time.time()
                print(f"Step: {step//100}-th hundred")
            
            gen_total_loss, disc_loss = self.train_step(input_signal, target, step)
            print('step #{}:'.format(step), end='\r')
            # Save (checkpoint) the model every 5k steps
            if (step + 1) % 100 == 0:
                self.generator.save_weights(self.checkpoint_dir + '/{}_generator'.format(self.name))
                self.discriminator.save_weights(self.checkpoint_dir + '/{}_discriminator'.format(self.name))

    def load(self, name):
        if os.path.isfile(self.checkpoint_dir + '/{}_generator'.format(name) + '.index'):
            self.generator.load_weights(self.checkpoint_dir + '/{}_generator'.format(name)).expect_partial()
        if os.path.isfile(self.checkpoint_dir + '/{}_discriminator'.format(name) + '.index'):
            self.discriminator.load_weights(self.checkpoint_dir + '/{}_discriminator'.format(name)).expect_partial()

    
    def evaluate(self, test_obs, test_true):
        pass