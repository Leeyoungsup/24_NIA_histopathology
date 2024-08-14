import math
import numpy as np
import tensorflow as tf
from keras.utils import plot_model
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.losses import *
from tqdm.notebook import tqdm
import keras.backend as K
import os, time, gc, random
import tensorflow as tf
from keras.layers import Input, Embedding, Concatenate, Dense, Reshape
from keras.models import Model
# blocks used in GAN
ndist = tf.random_normal_initializer(0, 1)
zeros = tf.zeros_initializer()
ones = tf.ones_initializer()
batchSize = 4
num_classes = 2

m = batchSize * (2000 // batchSize)
imgSize =512# size of images in pixels
zdim = imgSize # number of elements in a latent vector
p = 0.0 # probability of data augmentation
n = 4 # number of minibatches before p is changed
numImgsStep = 5e5 # number of images needed to change p from 0 -> 1 or 1 -> 0
pStep = n * batchSize / numImgsStep # how much p increases/decreases per n minibatches
eps = 1e-8 # epsilon, small number used to prevent NaN errors
pplEMA = 0.0 # exponential moving average for average PPL for PPL reg.
discLayerFilters = np.linspace(32,imgSize,int(np.log2(imgSize/4)),dtype=np.int32)
genLayerFilters = np.linspace(imgSize,32,int(np.log2(imgSize/4)),dtype=np.int32)
w1_range=int(np.log2(imgSize/2)/2)
w2_range=int(np.log2(imgSize/2)/2+(np.log2(imgSize/2)%2>=0.5))

def minibatchStd(inputs):
    inputs = tf.transpose(inputs, (0, 3, 1, 2)) # NHWC -> NCHW
    group_size = tf.minimum(4, tf.shape(inputs)[0])             # Minibatch must be divisible by (or smaller than) group_size.
    s = inputs.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(inputs, [group_size, -1, 1, s[1], s[2], s[3]])   # [GMncHW] Split minibatch into M groups of size G. Split channels into n channel groups c.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMncHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MncHW]  Calc variance over group.
    y = tf.sqrt(y + eps)                                    # [MncHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[2,3,4], keepdims=True)      # [Mn111]  Take average over fmaps and pixels.
    y = tf.reduce_mean(y, axis=[2])                         # [Mn11] Split channels into c channel groups
    y = tf.tile(y, [group_size, 1, s[2], s[3]])             # [NnHW]  Replicate over group and pixels.
    y = tf.concat([inputs, y], axis=1)                        # [NCHW]  Append as new fmap.
    y = tf.transpose(y, (0, 2, 3, 1)) # NCHW -> NHWC
    return y

class DiffUS(tf.keras.layers.Layer):
    def __init__(self):
        return super().__init__()
    
    def call(self, inputs):
        _N, H, W, C = inputs.shape.as_list()
        x = K.reshape(inputs, (-1, H, 1, W, 1, C))
        x = tf.tile(x, (1, 1, 2, 1, 2, 1))
        used = K.reshape(x, (-1, H * 2, W * 2, C))
        return used

def crop_to_fit(x):
    noise, img = x
    height = img.shape[1]
    width = img.shape[2]
    
    return noise[:, :height, :width, :]



class FCE(Dense): # fully connected equalized
    def __init__(self, units, kernel_initializer=ndist, bias_initializer=zeros, lrelu=True, *args, **kwargs):
        super().__init__(units, *args, **kwargs)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.lrelu = lrelu
        self.scale = 1

    def build(self, input_shape):
        super().build(input_shape)
        #print('fce', input_shape)
        n = input_shape[-1] # input_shape = (None, features_in) or (None, dimY, dimX, features_in)
        if self.lrelu:
            self.scale = np.sqrt((1 / 0.6) / n) # he but not really, 1 / 0.6 since lrelu(0.2) makes scales variance to 0.6 (0.2 if neg, 1 if pos, div by 2) and you want them to be 1
        else:
            self.scale = np.sqrt(1 / n)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.scale)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not tf.keras.activations.linear:
            output = self.activation(output)
        elif self.lrelu:
            output = LeakyReLU(alpha=0.2)(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'kInit': self.kernel_initializer,
            'bInit': self.bias_initializer,
            'scale': self.scale,
            'useLReLU': self.lrelu,
                      })
        return config

class CVE(Conv2D):
    def __init__(self, units, kernel_size=3, kernel_initializer=ndist, bias_initializer=zeros, padding='same', lrelu=True, *args, **kwargs):
        super().__init__(units, kernel_size, *args, **kwargs)
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.padding = padding
        self.lrelu = lrelu
        self.scale = 1

    def build(self, input_shape):
        super().build(input_shape)
        #print('cve', self.kernel.shape)
        n = np.prod(self.kernel.shape[:-1]) # self.kernel.shape = (kernel_x, kernel_y, features_in, features_out)
        if self.lrelu: # he
            self.scale = np.sqrt((1 / 0.6) / n)
        else:
            self.scale = np.sqrt(1 / n)


    def call(self, inputs):
        output = K.conv2d(inputs, self.kernel * self.scale, padding=self.padding)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not tf.keras.activations.linear:
            output = self.activation(output)
        elif self.lrelu:
            output = LeakyReLU(alpha=0.2)(output)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'kInit': self.kernel_initializer,
            'bInit': self.bias_initializer,
            'padding': self.padding,
            'scale': self.scale,
            'useLReLU': self.lrelu,
                      })
        return config

class ConvMod(Layer):
    def __init__(self, nf, x, w, kSize=3, demod=True):
        super().__init__()
        self.nf = nf
        self.kSize = kSize
        self.xShape = x.shape
        self.wShape = w.shape
        self.scale = FCE(self.xShape[-1], bias_initializer=ones, lrelu=False)
        self.conv = CVE(nf, kSize, lrelu=demod)
        self.conv(x) # create kernel without doing it in build method so h5py doesn't go sicko mode
        self.demod = demod

    def build(self, input_shape): # input_shape: [TensorShape([None, 4, 4, 256]), TensorShape([None, 256]), TensorShape([None, 4, 4, 1])]
        super().build(input_shape)

    def call(self, inputs):
        x, w = inputs

        x = tf.transpose(x, (0, 3, 1, 2)) # NHWC -> NCHW
        weight = self.conv.kernel[np.newaxis] * self.conv.scale # kkio -> 1kkio (1, kernel_size, kernel_size, input_features, output_features)

        scale = self.scale(w)
        scale = scale[:, np.newaxis, np.newaxis, :, np.newaxis] # Bs -> B, 1, 1, s, 1 (s - scaling factor)

        wp = weight * scale # 1kkio * B11s1 -> Bkk(s*i)o
        wpp = wp

        if self.demod:
            wStd = tf.math.rsqrt(tf.reduce_sum(tf.math.square(wp), axis=[1,2,3]) + 1e-8) # Bkkio -> Bo
            wpp = wp * wStd[:, np.newaxis, np.newaxis, np.newaxis, :] # [BkkIO] Scale output feature maps.

        x = tf.reshape(x, (1, -1, x.shape[2], x.shape[3])) # N, C, H, W -> 1, (N*C), H, W

        # B, k, k, i, o -> k, k, i, B, o -> k, k, i, (B*o)
        wpp = tf.reshape(tf.transpose(wpp, [1, 2, 3, 0, 4]), [wpp.shape[1], wpp.shape[2], wpp.shape[3], -1])

        x = tf.nn.conv2d(x, wpp, padding='SAME', data_format='NCHW', strides=[1, 1, 1, 1]) # grouped conv
        x = tf.reshape(x, (-1, self.nf, x.shape[2], x.shape[3])) # 1, (N*C), H, W -> N, C, H, W
        x = tf.transpose(x, (0, 2, 3, 1)) # NCHW -> NHWC
        x = K.bias_add(x, self.conv.bias)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_filters': self.nf,
            'kernel_size': self.kSize,
            'xShape': self.xShape,
            'wShape': self.wShape,
            'demodulated': self.demod
                      })
        return config
    
    
    
class timeIt:
    def __init__(self, description):
        self.start = time.time()
        self.description = description
        self.running = True
    
    def new(self, description, verbose=True):
        self.start = time.time()
        self.description = description
        
        duration = time.time() - startTime
        if verbose:
            print('{}; {:.4f} seconds to complete'.format(self.description, duration))
        
        return duration
    
    def close(self, verbose=True):
        duration = time.time() - self.start
        if verbose:
            print('{}; {:.4f} seconds to complete'.format(self.description, duration))
            
        self.start = None
        self.description = None
        self.running = False
        return duration

sess = timeIt('testing timer')
time.sleep(0.005)
_ = sess.close(verbose=True)


'''
Generator style block.
Args:
accum - accumulated output from the input/output skips
x - the non-RGB image input
w - the style (output of the mapping function with input of the latent vector)
noiseInp - normally distributed noise
filters - number of channels/feature maps the output of the style block will have
us - whether or not to upsample the images
'''
def gblock(accum, x, w, noiseInp, filters, us=True):
    if us:
        x = DiffUS()(x) # using custom upsampling function since other upsampling methods didn't provide gradients of their gradients
        accum = DiffUS()(accum)
    
    for i in range(2):
        x = ConvMod(filters, x, w)([x, w])
        noise = Lambda(crop_to_fit)([noiseInp, x]) # crop noises so it can be added with x
        noise = FCE(filters, kernel_initializer=zeros, use_bias=False, lrelu=False)(noise) #scale noises
        x = Add()([x, noise])
        x = LeakyReLU(alpha=0.2)(x)
    
    trgb = ConvMod(3, x, w, 1, demod=False)([x, w]) # toRGB 1x1 convolution
    accum = Add()([accum, trgb]) * np.sqrt(1 / 2) # the sqrt(1/2) not included in original StyleGAN2 but i didn't see why not
        
    return accum, x

# Discriminator block.
def dblock(x, filters, maxFilters=256):
    frgb = CVE(min(2 * filters, maxFilters), 1, lrelu=False, use_bias=False)(x)
    
    x = CVE(filters)(x)
    x = CVE(min(2 * filters, maxFilters))(x)
        
    frgb = AveragePooling2D()(frgb)
    x = AveragePooling2D()(x)
    x = Add()([x, frgb])
    
    return x

nBlocks = int(np.log2(imgSize / 4)) # number of upsampled style blocks

# mapper architecture
def ztow(nlayers=8):
    z = Input((zdim,))
    w = z
    if nlayers > 0:
        w = LayerNormalization()(w)
    for i in range(max(nlayers-1, 0)):
        w = FCE(zdim)(w)
    return Model(z, w, name='mapping')

# generator architecture
def genGen():
    ws = [Input((zdim,), name='w{}'.format(i)) for i in range(nBlocks+1)]
    noiseInp = Input((imgSize, imgSize, 1), name='noiseInp')

    x = Dense(1)(ws[0]); x = Lambda(lambda x: x * 0 + 1)(x)
    x = FCE(4*4*zdim, lrelu=False, use_bias=False)(x)
    x = Reshape((4, 4, zdim))(x)
    
    
    
    x = ConvMod(genLayerFilters[0], x, ws[0])([x, ws[0]])
    noise = Lambda(crop_to_fit)([noiseInp, x])
    noise = FCE(genLayerFilters[0], kernel_initializer=zeros, use_bias=False, lrelu=False)(noise)
    x = Add()([x, noise])
    x = LeakyReLU(alpha=0.2)(x)
    accum = ConvMod(3, x, ws[0], 1, demod=False)([x, ws[0]])
    
    for idx, f in enumerate(genLayerFilters):
        accum, x = gblock(accum, x, ws[idx+1], noiseInp, f)
        
    out = CVE(3, 1, lrelu=False)(accum)
    return Model([*ws, noiseInp], out, name='generator')
      
# discriminator architecture  
def genDisc():
    inp = Input((imgSize, imgSize, 3)); x = inp

    
    x = CVE(discLayerFilters[0], 1)(x)
    for fi, f in enumerate(discLayerFilters):
        x = dblock(x, f, maxFilters=discLayerFilters[-1])
    
    x = Lambda(minibatchStd)(x)
    x = CVE(discLayerFilters[-1])(x)
    x = Flatten()(x)
    x = FCE(discLayerFilters[-1])(x)
    out = FCE(1, lrelu=False)(x)

    return Model(inp, out, name='discriminator')

def rt(truePreds): # overfitting metric
    return tf.reduce_mean(tf.sign(truePreds))

def dra(obsPreds, basePreds): # observe/baseline predictions (representing fake/true data)
    meanBase = K.mean(basePreds)
    return tf.nn.sigmoid(obsPreds - meanBase)

def discLoss(truePreds, fakePreds, epsilon=eps):
    trueLoss = K.mean(tf.nn.softplus(-truePreds)) # -log(sigmoid(real_scores_out))
    fakeLoss = K.mean(tf.nn.softplus(fakePreds)) # -log(1-sigmoid(fake_scores_out))
    classLoss = trueLoss + fakeLoss
    return classLoss

def genLoss(fakePreds, epsilon=eps):
    classLoss = K.mean(tf.nn.softplus(-fakePreds))
    return classLoss

def ztow_with_labels(nlayers=8):
    z = Input((zdim,))
    label_input = Input((1,), dtype='int32')
    label_embedding = Embedding(num_classes, zdim, input_length=1)(label_input)
    label_embedding = Reshape((zdim,))(label_embedding)
    
    w = Concatenate()([z, label_embedding])
    
    if nlayers > 0:
        w = LayerNormalization()(w)
    for i in range(max(nlayers-1, 0)):
        w = FCE(zdim)(w)
    
    return Model([z, label_input], w, name='mapping_with_labels')

# Generator architecture with label embedding
def genGen_with_labels():
    ws = [Input((zdim,), name='w{}'.format(i)) for i in range(nBlocks+1)]
    noiseInp = Input((imgSize, imgSize, 1), name='noiseInp')
    label_input = Input((1,), dtype='int32')
    
    label_embedding = Embedding(num_classes, 4 * 4 * zdim, input_length=1)(label_input)
    label_embedding = Reshape((4, 4, zdim))(label_embedding)
    
    x = Dense(1)(ws[0]); x = Lambda(lambda x: x * 0 + 1)(x)
    x = FCE(4*4*zdim, lrelu=False, use_bias=False)(x)
    x = Reshape((4, 4, zdim))(x)
    
    x = Concatenate()([x, label_embedding])
    
    x = ConvMod(genLayerFilters[0], x, ws[0])([x, ws[0]])
    noise = Lambda(crop_to_fit)([noiseInp, x])
    noise = FCE(genLayerFilters[0], kernel_initializer=zeros, use_bias=False, lrelu=False)(noise)
    x = Add()([x, noise])
    x = LeakyReLU(alpha=0.2)(x)
    accum = ConvMod(3, x, ws[0], 1, demod=False)([x, ws[0]])
    
    for idx, f in enumerate(genLayerFilters):
        accum, x = gblock(accum, x, ws[idx+1], noiseInp, f)
        
    out = CVE(3, 1, lrelu=False)(accum)
    return Model([*ws, noiseInp, label_input], out, name='generator_with_labels')

# Discriminator architecture with label embedding
def genDisc_with_labels():
    inp = Input((imgSize, imgSize, 3))
    label_input = Input((1,), dtype='int32')
    
    label_embedding = Embedding(num_classes, imgSize * imgSize, input_length=1)(label_input)
    label_embedding = Reshape((imgSize, imgSize, 1))(label_embedding)
    
    x = Concatenate()([inp, label_embedding])
    
    x = CVE(discLayerFilters[0], 1)(x)
    for fi, f in enumerate(discLayerFilters):
        x = dblock(x, f, maxFilters=discLayerFilters[-1])
    
    x = Lambda(minibatchStd)(x)
    x = CVE(discLayerFilters[-1])(x)
    x = Flatten()(x)
    x = FCE(discLayerFilters[-1])(x)
    out = FCE(1, lrelu=False)(x)
    
    return Model([inp, label_input], out, name='discriminator_with_labels')