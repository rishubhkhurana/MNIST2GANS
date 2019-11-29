###############################################################################################################################
Week3

Test accuracy of Base network: 0.815
      

    # Define the model
    model = Sequential()
    model.add(SeparableConv2D(64, 3, 3, border_mode='same', input_shape=(32, 32, 3)))#32,3
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p_conv))

    model.add(SeparableConv2D(64, 3, 3))#30,5
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p_conv))


    model.add(MaxPooling2D(pool_size=(2, 2)))#15,6


    model.add(SeparableConv2D(64, 3, 3, border_mode='same'))#15,10
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p_conv))

    model.add(SeparableConv2D(128, 3, 3))#13,14
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p_conv))


    model.add(MaxPooling2D(pool_size=(2, 2)))#6,16


    model.add(SeparableConv2D(128, 3, 3, border_mode='same'))#6,24
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p_conv))

    model.add(SeparableConv2D(128, 3, 3))#4,32
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p_conv))


    model.add(GlobalAveragePooling2D())#2,36


    model.add(Dense(256))#512
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p_lin))

    model.add(Dense(num_classes, activation='softmax'))#10


model has ~92,901 parameters


Epoch 1/50
390/390 [==============================] - 43s 109ms/step - loss: 1.3826 - acc: 0.5013 - val_loss: 1.7059 - val_acc: 0.4796
Epoch 2/50
390/390 [==============================] - 40s 102ms/step - loss: 0.9997 - acc: 0.6455 - val_loss: 1.8238 - val_acc: 0.4863
Epoch 3/50
390/390 [==============================] - 40s 102ms/step - loss: 0.8606 - acc: 0.6992 - val_loss: 1.1250 - val_acc: 0.6204
Epoch 4/50
390/390 [==============================] - 40s 102ms/step - loss: 0.7903 - acc: 0.7213 - val_loss: 1.2502 - val_acc: 0.6033
Epoch 5/50
390/390 [==============================] - 39s 101ms/step - loss: 0.7357 - acc: 0.7425 - val_loss: 0.8893 - val_acc: 0.6973
Epoch 6/50
390/390 [==============================] - 39s 100ms/step - loss: 0.6946 - acc: 0.7571 - val_loss: 0.7023 - val_acc: 0.7588
Epoch 7/50
390/390 [==============================] - 39s 100ms/step - loss: 0.6667 - acc: 0.7685 - val_loss: 0.8231 - val_acc: 0.7194
Epoch 8/50
390/390 [==============================] - 39s 100ms/step - loss: 0.6341 - acc: 0.7795 - val_loss: 0.9420 - val_acc: 0.6949
Epoch 9/50
390/390 [==============================] - 39s 100ms/step - loss: 0.6087 - acc: 0.7890 - val_loss: 0.7864 - val_acc: 0.7419
Epoch 10/50
390/390 [==============================] - 39s 100ms/step - loss: 0.5939 - acc: 0.7930 - val_loss: 1.0321 - val_acc: 0.6663
Epoch 11/50
390/390 [==============================] - 39s 100ms/step - loss: 0.5801 - acc: 0.7988 - val_loss: 0.6912 - val_acc: 0.7692
Epoch 12/50
390/390 [==============================] - 39s 101ms/step - loss: 0.5639 - acc: 0.8039 - val_loss: 0.7263 - val_acc: 0.7601
Epoch 13/50
390/390 [==============================] - 39s 100ms/step - loss: 0.5465 - acc: 0.8096 - val_loss: 0.7133 - val_acc: 0.7655
Epoch 14/50
390/390 [==============================] - 39s 100ms/step - loss: 0.5332 - acc: 0.8154 - val_loss: 0.5538 - val_acc: 0.8131
Epoch 15/50
390/390 [==============================] - 39s 101ms/step - loss: 0.5229 - acc: 0.8191 - val_loss: 0.6630 - val_acc: 0.7810
Epoch 16/50
390/390 [==============================] - 39s 100ms/step - loss: 0.5122 - acc: 0.8191 - val_loss: 0.6898 - val_acc: 0.7760
Epoch 17/50
390/390 [==============================] - 39s 101ms/step - loss: 0.5007 - acc: 0.8263 - val_loss: 0.6181 - val_acc: 0.7989
Epoch 18/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4915 - acc: 0.8264 - val_loss: 0.6281 - val_acc: 0.7838
Epoch 19/50
390/390 [==============================] - 39s 101ms/step - loss: 0.4839 - acc: 0.8321 - val_loss: 0.5592 - val_acc: 0.8168
Epoch 20/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4776 - acc: 0.8323 - val_loss: 0.5351 - val_acc: 0.8206
Epoch 21/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4668 - acc: 0.8376 - val_loss: 0.6002 - val_acc: 0.8039
Epoch 22/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4613 - acc: 0.8399 - val_loss: 0.6136 - val_acc: 0.8022
Epoch 23/50
390/390 [==============================] - 39s 101ms/step - loss: 0.4528 - acc: 0.8400 - val_loss: 0.6625 - val_acc: 0.7828
Epoch 24/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4512 - acc: 0.8415 - val_loss: 0.6523 - val_acc: 0.7870
Epoch 25/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4413 - acc: 0.8451 - val_loss: 0.6715 - val_acc: 0.7800
Epoch 26/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4330 - acc: 0.8479 - val_loss: 0.6340 - val_acc: 0.7963
Epoch 27/50
390/390 [==============================] - 39s 101ms/step - loss: 0.4284 - acc: 0.8498 - val_loss: 0.5680 - val_acc: 0.8132
Epoch 28/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4261 - acc: 0.8491 - val_loss: 0.5936 - val_acc: 0.8097
Epoch 29/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4201 - acc: 0.8507 - val_loss: 0.5450 - val_acc: 0.8194
Epoch 30/50
390/390 [==============================] - 39s 100ms/step - loss: 0.4143 - acc: 0.8546 - val_loss: 0.5687 - val_acc: 0.8169
Epoch 31/50
390/390 [==============================] - 39s 99ms/step - loss: 0.4094 - acc: 0.8552 - val_loss: 0.5360 - val_acc: 0.8229
Epoch 32/50
390/390 [==============================] - 39s 99ms/step - loss: 0.4046 - acc: 0.8571 - val_loss: 0.6825 - val_acc: 0.7926
Epoch 33/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3990 - acc: 0.8594 - val_loss: 0.5382 - val_acc: 0.8242
Epoch 34/50
390/390 [==============================] - 39s 99ms/step - loss: 0.3914 - acc: 0.8623 - val_loss: 0.6385 - val_acc: 0.7986
Epoch 35/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3916 - acc: 0.8620 - val_loss: 0.5680 - val_acc: 0.8146
Epoch 36/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3895 - acc: 0.8632 - val_loss: 0.5821 - val_acc: 0.8206
Epoch 37/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3821 - acc: 0.8658 - val_loss: 0.6054 - val_acc: 0.7982
Epoch 38/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3793 - acc: 0.8660 - val_loss: 0.5413 - val_acc: 0.8269
Epoch 39/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3753 - acc: 0.8670 - val_loss: 0.5654 - val_acc: 0.8210
Epoch 40/50
390/390 [==============================] - 39s 99ms/step - loss: 0.3747 - acc: 0.8687 - val_loss: 0.7002 - val_acc: 0.7817
Epoch 41/50
390/390 [==============================] - 39s 99ms/step - loss: 0.3663 - acc: 0.8699 - val_loss: 0.5109 - val_acc: 0.8346
Epoch 42/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3680 - acc: 0.8711 - val_loss: 0.5184 - val_acc: 0.8302
Epoch 43/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3610 - acc: 0.8726 - val_loss: 0.5720 - val_acc: 0.8171
Epoch 44/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3574 - acc: 0.8730 - val_loss: 0.5129 - val_acc: 0.8317
Epoch 45/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3585 - acc: 0.8731 - val_loss: 0.4968 - val_acc: 0.8393
Epoch 46/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3549 - acc: 0.8733 - val_loss: 0.5410 - val_acc: 0.8216
Epoch 47/50
390/390 [==============================] - 38s 99ms/step - loss: 0.3585 - acc: 0.8726 - val_loss: 0.6517 - val_acc: 0.8033
Epoch 48/50
390/390 [==============================] - 39s 99ms/step - loss: 0.3421 - acc: 0.8780 - val_loss: 0.4747 - val_acc: 0.8455
Epoch 49/50
390/390 [==============================] - 39s 100ms/step - loss: 0.3477 - acc: 0.8766 - val_loss: 0.5394 - val_acc: 0.8282
Epoch 50/50
390/390 [==============================] - 38s 99ms/step - loss: 0.3429 - acc: 0.8788 - val_loss: 0.5529 - val_acc: 0.8278




