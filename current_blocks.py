import tensorflow as tf

from tensorflow.keras.layers import *

def _get_var(_vars, i):
    if callable(_vars): return _vars(i)
    elif isinstance(_vars, list): return _vars[i]
    else: return _vars
    
def _sequential_layer_bn(model, layer_type, n, * args, 
                         bnorm = 'after', momentum=0.99, epsilon=0.001,
                         activation = None, drop_rate = 0.1, name = None, ** kwargs):
    assert isinstance(model, tf.keras.Sequential)
    assert bnorm in ('before', 'after', 'never')
    
    if bnorm == 'before':
        model.add(BatchNormalization(momentum = momentum, epsilon = epsilon))
    
    for i in range(n):
        args_i = [_get_var(a, i) for a in args]
        kwargs_i = {k : _get_var(v, i) for k, v in kwargs.items()}
        kwargs_i['name'] = '{}_{}'.format(name, i+1) if name and n > 1 else name
        if i > 0: kwargs_i.pop('input_shape', None)

        model.add(layer_type(* args_i, ** kwargs_i))
        
        if activation and i < n - 1:
            act_layer = Activation(activation)
            model.add(act_layer)
    
    if bnorm == 'after':
        model.add(BatchNormalization(momentum = momentum, epsilon = epsilon))
    
    if activation:
        act_layer = Activation(activation)
        model.add(act_layer)
    
    if drop_rate > 0.: model.add(Dropout(drop_rate))
    
    return model

def _sequential_pooling(model, dim, pooling = None, pool_size = 2, pool_strides = 2, 
                        drop_rate = 0.1):
    assert pooling in (None, 'none', 'max', 'avg', 'average', 'up', 'upsampling'), '{} is not a valid pooling type'.format(pooling)
    assert dim in ('1d', '1D', '2d', '2D')
    
    if pooling == 'max':
        if dim in ('1d', '1D'):
            model.add(MaxPooling1D(pool_size, pool_strides))
        else:
            model.add(MaxPooling2D(pool_size, pool_strides))
    elif pooling in ('avg', 'average'):
        if dim in ('1d', '1D'):
            model.add(AveragePooling1D(pool_size, pool_strides))
        else:
            model.add(AveragePooling2D(pool_size, pool_strides))
    elif pooling in ('up', 'upsampling'):
        if dim in ('1d', '1D'):
            model.add(UpSampling1D(size = pool_size))
        else:
            model.add(UpSampling2D(size = pool_size))
    
    
    if drop_rate > 0.: model.add(Dropout(drop_rate))
    
    return model
    
def _pooling(x, dim, pooling = None, pool_size = 2, pool_strides = 2, drop_rate = 0.1):
    assert pooling in (None, 'none', 'max', 'avg', 'average', 'up', 'upsampling'), '{} is not a valid pooling type'.format(pooling)
    assert dim in ('1d', '1D', '2d', '2D')
    if isinstance(x, tf.keras.Model):
        return _sequential_pooling(x, dim, pooling, pool_size = pool_size, 
                                   pool_strides = pool_strides, drop_rate = drop_rate)
    
    if pooling == 'max':
        if dim in ('1d', '1D'):
            x = MaxPooling1D(pool_size, pool_strides)(x)
        else:
            x = MaxPooling2D(pool_size, pool_strides)(x)
    elif pooling in ('avg', 'average'):
        if dim in ('1d', '1D'):
            x = AveragePooling1D(pool_size, pool_strides)(x)
        else:
            x = AveragePooling2D(pool_size, pool_strides)(x)
    elif pooling in ('up', 'upsampling'):
        if dim in ('1d', '1D'):
            x = UpSampling1D(size = pool_size)(x)
        else:
            x = UpSampling2D(size = pool_size)(x)
    
    if drop_rate > 0.: x = Dropout(drop_rate)(x)
    
    return x

def _layer_bn(inputs, layer_type, n, * args, 
              bnorm = 'after', momentum=0.99, epsilon=0.001,
              activation = None, 
              drop_rate = 0.1, 
              name = None, 
              residual = False, 
              ** kwargs):
    x = inputs
    assert bnorm in ('before', 'after', 'never')
    if isinstance(x, tf.keras.Sequential):
        return _sequential_layer_bn(x, layer_type, n, *args, bnorm = bnorm, 
                                    momentum = momentum, epsilon = epsilon,
                                    activation = activation, drop_rate = drop_rate, 
                                    name = name, ** kwargs)
    
    if bnorm == 'before':
        x = BatchNormalization(momentum = momentum, epsilon = epsilon)(x)
    
    for i in range(n):
        args_i = [_get_var(a, i) for a in args]
        kwargs_i = {k : _get_var(v, i) for k, v in kwargs.items()}
        kwargs_i['name'] = '{}_{}'.format(name, i+1) if name and n > 1 else name
        strides = kwargs_i.get('strides', 1)
        if isinstance(strides, (list, tuple)): strides = max(strides)
        if strides > 1: residual = False
        if i > 0: kwargs_i.pop('input_shape', None)
        
        x = layer_type(* args_i, ** kwargs_i)(x)
        
        if activation and i < n - 1:
            x = Activation(activation)(x)
    
    if bnorm == 'after':
        x = BatchNormalization(momentum = momentum, epsilon = epsilon)(x)
    
    if residual and tuple(x.shape) == tuple(inputs.shape):
        x = Add()([x, inputs])
    elif residual:
        print("Skip connection failed, not same shape : {} vs {}".format(inputs.shape, x.shape))
    
    if activation is not None:
        x = Activation(activation)(x)
    
    if drop_rate > 0.: x = Dropout(drop_rate)(x)
    
    return x

def DenseBN(x, units, * args, ** kwargs):
    n = len(units) if isinstance(units, (list, tuple)) else 1
    kwargs['units'] = units
    
    kwargs.setdefault('activation', 'leaky')
    
    return _layer_bn(x, Dense, n, * args, ** kwargs)
    

def Conv2DBN(x, filters, *args, drop_rate = 0.1, pooling = None, 
             pool_size = 2, pool_strides = 2, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    kwargs.setdefault('activation', 'relu')
    
    x = _layer_bn(x, Conv2D, n, * args, drop_rate = 0., ** kwargs)
    
    x = _pooling(x, '2d', pooling, pool_size = pool_size, 
                 pool_strides = pool_strides, drop_rate = drop_rate)
    
    return x

def Conv1DBN(x, filters, *args, drop_rate = 0.1, pooling = None, 
             pool_size = 2, pool_strides = 2, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    kwargs.setdefault('activation', 'relu')
    
    x = _layer_bn(x, Conv1D, n, * args, drop_rate = 0., ** kwargs)
    
    x = _pooling(x, '1d', pooling, pool_size = pool_size, 
                 pool_strides = pool_strides, drop_rate = drop_rate)
    
    return x

def SeparableConv2DBN(x, filters, *args, drop_rate = 0.1, pooling = None, 
                      pool_size = 2, pool_strides = 2, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    kwargs.setdefault('activation', 'relu')
    
    x = _layer_bn(x, SeparableConv2D, n, * args, drop_rate = 0., ** kwargs)
    
    x = _pooling(x, '2d', pooling, pool_size = pool_size, 
                 pool_strides = pool_strides, drop_rate = drop_rate)
    
    return x

def SeparableConv1DBN(x, filters, *args, drop_rate = 0.1, pooling = None, 
             pool_size = 2, pool_strides = 2, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    kwargs.setdefault('activation', 'relu')
    
    x = _layer_bn(x, SeparableConv1D, n, * args, drop_rate = 0., ** kwargs)
    
    x = _pooling(x, '1d', pooling, pool_size = pool_size, 
                 pool_strides = pool_strides, drop_rate = drop_rate)
    
    return x

def Conv2DTransposeBN(x, filters, * args, drop_rate = 0.1, pooling = None, 
             pool_size = 2, pool_strides = 2, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    kwargs.setdefault('activation', 'leaky')
    
    x = _layer_bn(x, Conv2DTranspose, n, * args, drop_rate = 0., ** kwargs)
    
    x = _pooling(x, '2d', pooling, pool_size = pool_size, 
                 pool_strides = pool_strides, drop_rate = drop_rate)
    
    return x

def Conv1DTransposeBN(x, filters, * args, drop_rate = 0.1, pooling = None, 
             pool_size = 2, pool_strides = 2, ** kwargs):
    n = len(filters) if isinstance(filters, (list, tuple)) else 1
    kwargs['filters'] = filters
    
    kwargs.setdefault('activation', 'leaky')
    
    x = _layer_bn(x, Conv1DTranspose, n, * args, drop_rate = 0., ** kwargs)
    
    x = _pooling(x, '1d', pooling, pool_size = pool_size, 
                 pool_strides = pool_strides, drop_rate = drop_rate)

    return x