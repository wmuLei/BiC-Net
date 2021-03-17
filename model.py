def CONV2D(x, filter_num, kernel_size, activation='relu', **kwargs):
    x = Conv2D(filter_num, kernel_size, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    if activation=='relu': 
        x = Activation('relu', **kwargs)(x)
    elif activation=='sigmoid': 
        x = Activation('sigmoid', **kwargs)(x)
    else:
        x = Activation('softmax', **kwargs)(x)
    return x


def BiC_Net(shape, classes=1):
    inputs = Input(shape)
    pool1 = BatchNormalization()(inputs)
    
    global conv1a, conv2a, conv3a, conv4a, conv5a
    conv1a, conv2a, conv3a, conv4a, conv5a = None, None, None, None, None
    global merg1a, merg1b, merg1c, merg1d
    merg1a, merg1b, merg1c, merg1d = None, None, None, None

    if conv1a is not None: pool1 = merge([pool1, conv1a], mode='concat', concat_axis=3); 
    if merg1d is not None: pool1 = merge([pool1, merg1d], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 32, (3, 3));    conv1 = CONV2D(conv0, 32, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1);  # 512/2
    pool1a = MaxPooling2D(pool_size=(4, 4))(conv1);  # 512/4

    if conv2a is not None: pool1 = merge([pool1, conv2a], mode='concat', concat_axis=3); 
    if merg1c is not None: pool1 = merge([pool1, merg1c], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 64, (3, 3));    conv2 = CONV2D(conv0, 64, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2);  # 512/4
    pool1b = MaxPooling2D(pool_size=(4, 4))(conv2);  # 512/8

    if conv3a is not None: pool1 = merge([pool1, conv3a], mode='concat', concat_axis=3);
    if merg1b is not None: pool1 = merge([pool1, merg1b], mode='concat', concat_axis=3);
    pool1 = merge([pool1, pool1a], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 128, (3, 3));    conv3 = CONV2D(conv0, 128, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv3);  # 512/8
    pool1c = MaxPooling2D(pool_size=(4, 4))(conv3);  # 512/16

    if conv4a is not None: pool1 = merge([pool1, conv4a], mode='concat', concat_axis=3); 
    if merg1a is not None: pool1 = merge([pool1, merg1a], mode='concat', concat_axis=3); 
    pool1 = merge([pool1, pool1b], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 256, (3, 3));    conv4 = CONV2D(conv0, 256, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv4);  # 512/16
    pool1d = MaxPooling2D(pool_size=(4, 4))(conv4);  # 512/32

    if conv5a is not None: pool1 = merge([pool1, conv5a], mode='concat', concat_axis=3); 
    pool1 = merge([pool1, pool1c], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 512, (3, 3));    conv5 = CONV2D(conv0, 512, (3, 3));
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv5);  # 512/32

    #----------------------------------------------
    pool1 = merge([pool1, pool1d], mode='concat', concat_axis=3); 
    conv0 = CONV2D(pool1, 1024, (3, 3));    conv0 = CONV2D(conv0, 1024, (3, 3));# 512/32
    #----------------------------------------------

    merg1 = UpSampling2D(size=(2, 2))(conv0);
    merg1a = UpSampling2D(size=(4, 4))(conv0);  # 512/8
    merg1 = merge([merg1, pool1c, conv5], mode='concat', concat_axis=3) # 512/16
    conv0 = CONV2D(merg1, 512, (3, 3));    conv5a = CONV2D(conv0, 512, (3, 3));
    
    merg1 = UpSampling2D(size=(2, 2))(conv5a);
    merg1b = UpSampling2D(size=(4, 4))(conv5a);  # 512/4
    merg1 = merge([merg1, merg1a, pool1b, conv4], mode='concat', concat_axis=3) # 512/8
    conv0 = CONV2D(merg1, 256, (3, 3));    conv4a = CONV2D(conv0, 256, (3, 3));

    merg1 = UpSampling2D(size=(2, 2))(conv4a);
    merg1c = UpSampling2D(size=(4, 4))(conv4a);  # 512/2
    merg1 = merge([merg1, merg1b, pool1a, conv3], mode='concat', concat_axis=3) # 512/4
    conv0 = CONV2D(merg1, 128, (3, 3));    conv3a = CONV2D(conv0, 128, (3, 3));

    merg1 = UpSampling2D(size=(2, 2))(conv3a);
    merg1d = UpSampling2D(size=(4, 4))(conv3a);  # 512/1
    merg1 = merge([merg1, merg1c, conv2], mode='concat', concat_axis=3) # 512/2
    conv0 = CONV2D(merg1, 64, (3, 3));    conv2a = CONV2D(conv0, 64, (3, 3));

    merg1 = UpSampling2D(size=(2, 2))(conv2a);
    merg1 = merge([merg1, merg1d, conv1], mode='concat', concat_axis=3) # 512/1
    conv0 = CONV2D(merg1, 32, (3, 3));    conv1a = CONV2D(conv0, 32, (3, 3));

    conv0 = CONV2D(conv1a, classes, (1, 1), activation='sigmoid')
    model = Model(input=inputs, output=conv0)
    model.summary() 
    return model
