import tensorflow as tf
import tensorflow.keras.backend as K

        
# layers and sub blocks

def conv(ndims, *args, **kwargs):
    if ndims==2:
        return  tf.keras.layers.Conv2D(*args, **kwargs)
    elif ndims==3:
        return  tf.keras.layers.Conv3D(*args, **kwargs)


def depth_to_space_3D(input_tensor, block_size):
    # Get the input shape
    batch, depth, height, width, channels = input_tensor.get_shape().as_list()
    
    # Reshape the tensor
    reshaped = tf.reshape(input_tensor, (batch, depth, height, block_size, width, block_size, channels // block_size**3))
    
    # Permute the dimensions
    permuted = tf.transpose(reshaped, (0, 1, 3, 2, 5, 4, 6))
    
    # Reshape to the final output tensor
    output_tensor = tf.reshape(permuted, (batch, depth * block_size, height * block_size, width * block_size, channels // block_size**3))
    
    return output_tensor

class WeightNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, **kwargs):
        super(WeightNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.kernel = self.layer.kernel
        self.kernel_norm = tf.norm(self.kernel, axis=None, keepdims=False)
        self.kernel_normalized = self.kernel / self.kernel_norm
        self.g = self.add_weight(name='g', shape=self.kernel_norm.shape, initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        self.kernel = self.kernel_normalized * self.g

    def call(self, inputs, **kwargs):
        return self.layer(inputs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def get_config(self):
        config = super(WeightNormalization, self).get_config()
        config.update(self.layer.get_config())
        return config

class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=input_shape[-1:], initializer=tf.random_normal_initializer(1., 0.02), trainable=True)

        self.offset = self.add_weight(name='offset', shape=input_shape[-1:], initializer='zeros', trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
    
    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

class InstanceNormalization3D(tf.keras.layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization3D, self).__init__()
        self.epsilon = epsilon
 
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
        
    def get_config(self):
        config = super(InstanceNormalization3D, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config
        
        
def instanceNorm(x,ndims):
    if ndims==2:
        x = InstanceNormalization()(x)
    elif ndims==3:
        x = InstanceNormalization3D()(x)
    return x
    
def res_block_EDSR(x_in, filters, kernel, norm_type='instancenorm', apply_norm=False, ndims=2):
    x = conv(ndims, filters, kernel, padding='same')(x_in)
    x = tf.keras.layers.Activation('relu')(x)
    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            x = tf.keras.layers.BatchNormalization()(x)
        elif norm_type.lower() == 'instancenorm':
            x = instanceNorm(x,ndims)
    x = conv(ndims, filters, kernel, padding='same')(x)
    x = tf.keras.layers.Add()([x_in, x])
    return x
    
def upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=2, nameIn=''):
    def upsample_edsr(x, factor, ndims, **kwargs):
        #zx = conv(ndims, num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
#        x = conv(ndims, num_filters, 3, padding='same', **kwargs)(x)
#        x = tf.keras.layers.Activation('relu')(x)
#        if apply_norm:
#            if norm_type.lower() == 'batchnorm':
#                x = tf.keras.layers.BatchNormalization()(x)
#            elif norm_type.lower() == 'instancenorm':
#                x = instanceNorm(x,ndims)
        if ndims==2:
            x = tf.keras.layers.UpSampling2D(size=factor)(x)
            #x = SubpixelConv2D(factor)(x)
            return x
        elif ndims==3:
            x = tf.keras.layers.UpSampling3D(size=factor)(x)
            return x
        x = conv(ndims, num_filters, 3, padding='same', **kwargs)(x)
        x = tf.keras.layers.Activation('relu')(x)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif norm_type.lower() == 'instancenorm':
                x = instanceNorm(x,ndims)
    if scale == 2:
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
    elif scale == 3:
        x = upsample_edsr(x, 3, ndims=ndims, name='conv2d_1_scale_3_up'+nameIn)
    elif scale == 4:
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_2_scale_2_up'+nameIn)
    elif scale == 8:
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_2_scale_2_up'+nameIn)
        x = upsample_edsr(x, 2, ndims=ndims, name='conv2d_3_scale_2_up'+nameIn)
    return x
    
    
def upsampleEDSR1D(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=2, nameIn=''):
    def upsample_edsr(x, factor, ndims, **kwargs):
        #zx = conv(ndims, num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
#        x = conv(ndims, num_filters, 3, padding='same', **kwargs)(x)
#        x = tf.keras.layers.Activation('relu')(x)
#        if apply_norm:
#            if norm_type.lower() == 'batchnorm':
#                x = tf.keras.layers.BatchNormalization()(x)
#            elif norm_type.lower() == 'instancenorm':
#                x = instanceNorm(x,ndims)
        if ndims==2:
            x = tf.keras.layers.UpSampling2D(size=factor)(x)
            #x = SubpixelConv2D(factor)(x)
            return x
        elif ndims==3:
            x = tf.keras.layers.UpSampling3D(size=factor)(x)
            return x
        x = conv(ndims, num_filters, 3, padding='same', **kwargs)(x)
        x = tf.keras.layers.Activation('relu')(x)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif norm_type.lower() == 'instancenorm':
                x = instanceNorm(x,ndims)
    if scale == 2:
        x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
    elif scale == 3:
        x = upsample_edsr(x, (3,1), ndims=ndims, name='conv2d_1_scale_3_up'+nameIn)
    elif scale == 4:
        x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
        x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_2_scale_2_up'+nameIn)
    elif scale == 8:
        x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_1_scale_2_up'+nameIn)
        x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_2_scale_2_up'+nameIn)
        x = upsample_edsr(x, (2,1), ndims=ndims, name='conv2d_3_scale_2_up'+nameIn)
    return x
    
    
def SubpixelConv2D(scale, **kwargs):
    return  tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale), **kwargs)

def SubpixelConv2DDown(scale, **kwargs):
        return  tf.keras.layers.Lambda(lambda x: tf.nn.space_to_depth(x, scale), **kwargs)

  
def disc_block(x_in, filters, ndims):
    x = conv(ndims, filters, 4, 1, padding='same')(x_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv(ndims, filters, 4, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x
        
def downsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=2, nameIn=''):
    def downsample_edsr(x, factor, ndims, **kwargs):
        if ndims==2:
            x = SubpixelConv2DDown(factor)(x)
        elif ndims==3:
            x = tf.keras.layers.Conv3D(num_filters, 3, factor, padding='same')(x)
        x = conv(ndims, num_filters, 3, padding='same', **kwargs)(x)
        x = tf.keras.layers.Activation('relu')(x)
        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                x = tf.keras.layers.BatchNormalization()(x)
            elif norm_type.lower() == 'instancenorm':
                x = instanceNorm(x,ndims)
        return x

    if scale == 2:
        x = downsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_down'+nameIn)
    elif scale == 3:
        x = downsample_edsr(x, 3, ndims=ndims, name='conv2d_1_scale_3_down'+nameIn)
    elif scale == 4:
        x = downsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_down'+nameIn)
        x = downsample_edsr(x, 2, ndims=ndims, name='conv2d_2_scale_2_down'+nameIn)
    elif scale == 8:
        x = downsample_edsr(x, 2, ndims=ndims, name='conv2d_1_scale_2_down'+nameIn)
        x = downsample_edsr(x, 2, ndims=ndims, name='conv2d_2_scale_2_down'+nameIn)
        x = downsample_edsr(x, 2, ndims=ndims, name='conv2d_3_scale_2_down'+nameIn)
    return x
    
def downsample(filters, size, norm_type='instancenorm', apply_norm=False, ndims=2):
  """Downsamples an input.
  Conv2D => Batchnorm => LeakyRelu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer
  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      conv(ndims, filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
  result.add(tf.keras.layers.LeakyReLU())
  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      if ndims==2:
        result.add(InstanceNormalization())
      elif ndims==3:
        result.add(InstanceNormalization3D())
  return result

def edsr(scale, num_filters=64, num_res_blocks=8, ndims=2):
    if ndims==2:
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
    elif ndims==3:
        x_in = tf.keras.layers.Input(shape=(None, None, None, 1))
    x = x_in
    x = b = conv(ndims, num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block_EDSR(b, num_filters, 3, norm_type='instancenorm', apply_norm=False, ndims=ndims)
    b = conv(ndims, num_filters, 3, padding='same')(b)
    x = tf.keras.layers.Add()([x, b])

    x = upsampleEDSR(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=ndims)
    
    x = conv(ndims, num_filters, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = conv(ndims, 1, 3, padding='same')(x)
    x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
    
    return tf.keras.models.Model(x_in, x, name="EDSR-XY")
    
def edsr1D(scale, num_filters=64, num_res_blocks=8, ndims=2):
    if ndims==2:
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
    elif ndims==3:
        x_in = tf.keras.layers.Input(shape=(None, None, None, 1))
    x = x_in
    x = b = conv(ndims, num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block_EDSR(b, num_filters, 3, norm_type='instancenorm', apply_norm=False, ndims=ndims)
    b = conv(ndims, num_filters, 3, padding='same')(b)
    x = tf.keras.layers.Add()([x, b])

    x = upsampleEDSR1D(x, scale, num_filters, norm_type='instancenorm', apply_norm=False, ndims=ndims)
    
    x = conv(ndims, num_filters, 3, padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = conv(ndims, 1, 3, padding='same')(x)
    x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
    
    return tf.keras.models.Model(x_in, x, name="EDSR-Z")
    
def DiscriminatorSRGAN3Axis(args):
    #xIn = tf.keras.layers.Input(shape=[args.fine_size, args.fine_size, args.output_nc], name='Disc_Inputs')

    xIn = tf.keras.layers.Input(shape=[args.fine_size*args.scale, args.fine_size*args.scale, 1], name='Disc_Inputs')


    # shallow layers
    x = conv(2, args.ndsrf, 4, 1, padding='same')(xIn)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv(2, args.ndsrf, 4, 2, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    numDiscBlocks=3
    for i in range(numDiscBlocks):
        x = disc_block(x, args.ndsrf*(2**(i+1)), 2)
    x = tf.keras.layers.Flatten()(x)
#    x = tf.keras.layers.Dense(1024)(x)
#    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    xOut = tf.keras.layers.Dense(1, dtype='float32')(x)
    '''
    h = lrelu(conv2d(image, options.df_dim, ks=3, s=1, name='dInitConv'))
    h = lrelu(batchnormSR(conv2d(h, options.df_dim, ks=3, s=s, name='dUpConv')))
    for i in range(numDiscBlocks):
        expon=2**(i+1)
        h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=1, name=f'dBlock{i+1}Conv')))
        h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=2, name=f'dBlock{i+1}UpConv')))
    h = conv2d(h, 1, ks=3, s=1, name='d_h3_pred')
    #h = lrelu(denselayer(slim.flatten(h), 1024, name="dFC1"))
    #h = denselayer(h, 1, name="dFCout")
    return h
    
    '''  
    return tf.keras.Model(inputs=[xIn], outputs=xOut, name="DiscrimSR3Axis")

def DiscriminatorSRGAN3D(args):

    xIn = tf.keras.layers.Input(shape=[args.batch_size*args.scale, args.fine_size*args.scale, args.fine_size*args.scale, 1], name='Disc_Inputs')
    

    # shallow layers
    x = conv(3, args.ndsrf, 4, 1, padding='same')(xIn)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv(3, args.ndsrf, 4, 2, padding='same')(x)
#    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    numDiscBlocks=3
    for i in range(numDiscBlocks):
        x = disc_block(x, args.ndsrf*(2**(i+1)), 3)
    x = tf.keras.layers.Flatten()(x)
#    x = tf.keras.layers.Dense(1024)(x)
#    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    xOut = tf.keras.layers.Dense(1, dtype='float32')(x)
    '''
    h = lrelu(conv2d(image, options.df_dim, ks=3, s=1, name='dInitConv'))
    h = lrelu(batchnormSR(conv2d(h, options.df_dim, ks=3, s=s, name='dUpConv')))
    for i in range(numDiscBlocks):
        expon=2**(i+1)
        h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=1, name=f'dBlock{i+1}Conv')))
        h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=2, name=f'dBlock{i+1}UpConv')))
    h = conv2d(h, 1, ks=3, s=1, name='d_h3_pred')
    #h = lrelu(denselayer(slim.flatten(h), 1024, name="dFC1"))
    #h = denselayer(h, 1, name="dFCout")
    return h
    
    '''  
    return tf.keras.Model(inputs=[xIn], outputs=xOut, name="DiscrimSR")
    
def DiscriminatorSRGAN(args):
    #xIn = tf.keras.layers.Input(shape=[args.fine_size, args.fine_size, args.output_nc], name='Disc_Inputs')
    if args.nDims==2:
        xIn = tf.keras.layers.Input(shape=[args.fine_size, args.fine_size, 1], name='Disc_Inputs')
    elif args.nDims==3:
        xIn = tf.keras.layers.Input(shape=[args.batch_size, args.fine_size, args.fine_size, 1], name='Disc_Inputs')
    

    # shallow layers
    x = conv(args.nDims, args.ndsrf, 4, 1, padding='same')(xIn)
    x = tf.keras.layers.LeakyReLU()(x)
    x = conv(args.nDims, args.ndsrf, 4, 2, padding='same')(x)
#    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    numDiscBlocks=3
    for i in range(numDiscBlocks):
        x = disc_block(x, args.ndsrf*(2**(i+1)), args.nDims)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    xOut = tf.keras.layers.Dense(1, dtype='float32')(x)
    '''
    h = lrelu(conv2d(image, options.df_dim, ks=3, s=1, name='dInitConv'))
    h = lrelu(batchnormSR(conv2d(h, options.df_dim, ks=3, s=s, name='dUpConv')))
    for i in range(numDiscBlocks):
        expon=2**(i+1)
        h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=1, name=f'dBlock{i+1}Conv')))
        h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=2, name=f'dBlock{i+1}UpConv')))
    h = conv2d(h, 1, ks=3, s=1, name='d_h3_pred')
    #h = lrelu(denselayer(slim.flatten(h), 1024, name="dFC1"))
    #h = denselayer(h, 1, name="dFCout")
    return h
    
    '''  
    return tf.keras.Model(inputs=[xIn], outputs=xOut, name="DiscrimSR")
      
def cyclegan_generator(args):
    # the choice between concat and addition layers depends on the stochasticity of the mapping
    if args.nDims==2:
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
    elif args.nDims==3:
        x_in = tf.keras.layers.Input(shape=(None, None, None, 1))
    x = x_in
    x = xinit = conv(args.nDims, args.ngf, 3, padding='same')(x)
    x1 = x = downsampleEDSR(x, 2, args.ngf*2, norm_type='instancenorm', apply_norm=True, ndims=args.nDims, nameIn='cycleganEncode1')
    x2 = x = downsampleEDSR(x, 2, args.ngf*4, norm_type='instancenorm', apply_norm=True, ndims=args.nDims,nameIn='cycleganEncode2')
    for i in range(args.cyclegan_blocks):
        x = res_block_EDSR(x, args.ngf*4, 3, norm_type='instancenorm', apply_norm=True, ndims=args.nDims)
    x = conv(args.nDims, args.ngf*4, 3, padding='same')(x)
    #x = tf.keras.layers.Concatenate()([x2, x])
    x=x+x2
    x = upsampleEDSR(x, 2, args.ngf*2, norm_type='instancenorm', apply_norm=True, ndims=args.nDims,nameIn='cycleganDecode1')
    #x = tf.keras.layers.Concatenate()([x1, x]) # this massively increases stochasticity
    x=x+x1 # this enforces more well behaved AB mapping
    x = upsampleEDSR(x, 2, args.ngf, norm_type='instancenorm', apply_norm=True, ndims=args.nDims,nameIn='cycleganDecode2')
    x=x+xinit
    x = conv(args.nDims, args.ngf, 3, padding='same')(x)
    x = conv(args.nDims, 1, 3, padding='same')(x)
    #x = x+x_in # this makes the net work fully residually
    x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
    return tf.keras.models.Model(x_in, x, name="cycleganGenerator")

def cyclegan_generator_compact(args):
    # the choice between concat and addition layers depends on the stochasticity of the mapping
    if args.nDims==2:
        x_in = tf.keras.layers.Input(shape=(None, None, 1))
    elif args.nDims==3:
        x_in = tf.keras.layers.Input(shape=(None, None, None, 1))
    x = x_in
    x = xinit = conv(args.nDims, args.ngf, 3, padding='same')(x)
    x1 = x = downsampleEDSR(x, 2, args.ngf, norm_type='instancenorm', apply_norm=True, ndims=args.nDims, nameIn='cycleganEncode1')
    x2 = x = downsampleEDSR(x, 2, args.ngf, norm_type='instancenorm', apply_norm=True, ndims=args.nDims,nameIn='cycleganEncode2')
    for i in range(args.cyclegan_blocks):
        x = res_block_EDSR(x, args.ngf, 3, norm_type='instancenorm', apply_norm=True, ndims=args.nDims)
    x = conv(args.nDims, args.ngf, 3, padding='same')(x)
    #x = tf.keras.layers.Concatenate()([x2, x])
    x=x+x2
    x = upsampleEDSR(x, 2, args.ngf, norm_type='instancenorm', apply_norm=True, ndims=args.nDims,nameIn='cycleganDecode1')
    #x = tf.keras.layers.Concatenate()([x1, x]) # this massively increases stochasticity
    x=x+x1 # this enforces more well behaved AB mapping
    x = upsampleEDSR(x, 2, args.ngf, norm_type='instancenorm', apply_norm=True, ndims=args.nDims,nameIn='cycleganDecode2')
    x=x+xinit
    x = conv(args.nDims, args.ngf, 3, padding='same')(x)
    x = conv(args.nDims, 1, 3, padding='same')(x)
    x = x+x_in # this makes the net work fully residually
    x = tf.keras.layers.Activation('tanh', dtype='float32')(x)
    return tf.keras.models.Model(x_in, x, name="cycleganGeneratorCompact")

def discriminatorCGAN(args,norm_type='instancenorm'):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    Discriminator model
  """

  initializer = tf.random_normal_initializer(0., 0.02)
  if args.nDims==2:
    inp = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')
  elif args.nDims==3:
    inp = tf.keras.layers.Input(shape=[None, None, None, 1], name='input_image')
  x = inp

  down1 = downsample(args.ndf, 3, norm_type, False, ndims=args.nDims)(x)  # (bs, 128, 128, 64)
  down2 = downsample(args.ndf*2, 3, norm_type, True, ndims=args.nDims)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(args.ndf*4, 3, norm_type, True, ndims=args.nDims)(down2)  # (bs, 32, 32, 256)

  if args.nDims==2:
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  elif args.nDims==3:
    zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (bs, 34, 34, 256)
  conv1 = conv(args.nDims, 
      args.ngf*8, 3, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv1)
  elif norm_type.lower() == 'instancenorm':
    norm1 = instanceNorm(conv1, args.nDims)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
  if args.nDims==2:
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
  elif args.nDims==3:
    zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (bs, 33, 33, 512)


  last = conv(args.nDims, 1, 3, strides=1, kernel_initializer=initializer, dtype='float32')(zero_pad2)  # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=inp, outputs=last, name="DiscrimCycle")
      
      
      
      

######################################################## standard losses
def meanAbsoluteError(labels, predictions):
    per_example_loss = tf.reduce_mean(tf.abs(labels-predictions), axis = [1,2,3]) # could not be bothered to softcode this...
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=labels.shape[0])
    
def meanAbsoluteError1D(labels, predictions):
    per_example_loss = tf.reduce_mean(tf.abs(labels-predictions), axis = [1]) # could not be bothered to softcode this...
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=labels.shape[0])
    
   
def meanSquaredError(labels, predictions):
    per_example_MSE = tf.reduce_mean(((labels-predictions))**2, axis = [1,2,3])
    return tf.nn.compute_average_loss(per_example_MSE, global_batch_size=labels.shape[0])
   
def sigmoidCrossEntropy(labels, logits):
    per_example_sxe = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels), axis = 1)
    return tf.nn.compute_average_loss(per_example_sxe, global_batch_size=labels.shape[0])
    
    
# special gan losses
def lsganLoss(disc_real_output, disc_generated_output):
    real_loss = meanSquaredError(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = meanSquaredError(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
    return 0.5*total_disc_loss
    
def lsganLoss1D(disc_real_output, disc_generated_output):
    real_loss = meanAbsoluteError1D(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = meanAbsoluteError1D(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
    return 0.5*total_disc_loss

def rellsganLoss(disc_real_output, disc_generated_output):
    real_loss = meanSquaredError(tf.ones_like(disc_real_output), disc_real_output-tf.reduce_mean(disc_generated_output))
    generated_loss = meanSquaredError(tf.zeros_like(disc_generated_output), disc_generated_output-tf.reduce_mean(disc_real_output))
    total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
    return 0.5*total_disc_loss

def advLsganLoss(disc_generated_output):
    adversarial_loss = meanSquaredError(tf.ones_like(disc_generated_output), disc_generated_output)
    return adversarial_loss
    
def advLsganLoss1D(disc_generated_output):
    adversarial_loss = meanAbsoluteError1D(tf.ones_like(disc_generated_output), disc_generated_output)
    return adversarial_loss
    
def reladvLsganLoss(disc_real_output, disc_generated_output):
    real_loss = meanSquaredError(tf.zeros_like(disc_real_output), disc_real_output-tf.reduce_mean(disc_generated_output))
    generated_loss = meanSquaredError(tf.ones_like(disc_generated_output), disc_generated_output-tf.reduce_mean(disc_real_output))
    total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
    return 0.5*total_disc_loss
#        adversarial_loss = meanSquaredError(tf.ones_like(disc_generated_output), disc_generated_output)
#        return adversarial_loss

def scganLoss(disc_real_output, disc_generated_output):
    real_loss = sigmoidCrossEntropy(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = sigmoidCrossEntropy(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss # optimal foolery is when this equals 1, 50 real, 50 fake
    return 0.5*total_disc_loss
    
def advScganLoss(disc_generated_output):
    adversarial_loss = sigmoidCrossEntropy(tf.ones_like(disc_generated_output), disc_generated_output)
    return adversarial_loss
    
def rel_scganLoss(disc_real_output, disc_generated_output):

    real_loss = sigmoidCrossEntropy(tf.ones_like(disc_real_output), disc_real_output-tf.reduce_mean(disc_generated_output))
    generated_loss = sigmoidCrossEntropy(tf.zeros_like(disc_generated_output), disc_generated_output-tf.reduce_mean(disc_real_output))
    
    total_disc_loss = 0.5*(real_loss + generated_loss) # optimal foolery is when this equals 1, 50 real, 50 fake
    return total_disc_loss
    
def rel_advScganLoss(disc_real_output, disc_generated_output):

    real_loss = sigmoidCrossEntropy(tf.ones_like(disc_generated_output), disc_generated_output-tf.reduce_mean(disc_real_output))
    generated_loss = sigmoidCrossEntropy(tf.zeros_like(disc_real_output), disc_real_output-tf.reduce_mean(disc_generated_output))
    
    adversarial_loss =  0.5*(real_loss + generated_loss)
    return adversarial_loss

