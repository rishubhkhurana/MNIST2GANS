Week1:

score=[0.04337654851005741, 0.9923]

Definitions--

1. Convolution -- Operator similar to the one used in signal processing. In image CNNs, this is used to extract features by aggreagting information from neighboring
pixels using a kernel(feature extractor). 
2. Kernels/Filters -- Feature extractor that is used during the convolution. It is bascially learned during training of Neural network. This helps to extract featur from 
an image that can be used by later stages in NN for prediction. 
3. Epochs-- An epoch is an entire pass through the training dataset. Number of epochs determine how many passes of dataset is done during training of NN
4. 1x1 Convolution-- Convolution operation over a patch of 1x1xc pixel array, where c signify the number of channels. This operation is done to summarize the information in cchannels 
into ouput d channels for every pixels. This is generally used to reduce the number of channels at the start of new convolution block 
5. 3x3 Convolution-- basically convolution run over a patch of 3x3xc pixel array, where c signify the number of channels. This is predominantly used to extract features around the center node of 3x3 array. 
6.  Feature Maps-- Output of convolution. This is the new image produced by running filter over the input image. All the extracted features reside here. 
7. Activation Function-- Predominant source of non linearity in the network. This helps in transforming the input image into a new non linear space where useful features of the input
reside. This could be relu, signmoid, tanh etc.
8. Receptive Field -- There are two receptive fields in CNN -- local recpetive field and gloabl receptive field. Local receptive field is basically the size of  kernel. 
Local receptive field can also be seen as the patch of pixels that are used to compute the pixel for output image. Global receptive field,on the other hand, is indication of how many oirginal input image pixels have passed information
for calculation of  the current pixel at output. For example, when we apply two 3x3 filters back to back on input image, local recptive field of any pixel at the output of 2nd filter will be 3x3 but global receptive field would be 5x5.

#############################################################################################################################################################################################

Week2:


Strategy for achieving 99.4% accuracy within 20epoch with <15k parameters

Started with code8 file which had around 16K parameters. Had many observations--
1.First 1x1 convolutional happening before MaxPool2D. Doesn't really go with the lectures. Intuitively, 1x1 conv layer acts like an average over channel 
  space and max pool acts like "average" over widht and height. So, there shouldn't be a reason to prefer one over another but let's check it by swaping the layer in the next model.

2.Dropout is added just before softmax. This is not required as this will drop the output neurons for certain classes. We will remove this in future models.

3.Overall architecture seems to bank on generating 32 complex features at 5x5 RF and then keep on extracting 16 complex features over ever increasing receptive fields. 
  I guess for MNIST, we won't need to extract as many complex features. We could reduce the 16 channel conv layers or may reduce the number of channels in each of them. Let's experiment.

4.Learning rate scheduler also adds another dimension to the model. We could try to first model without the scheduler. Add scheduler only when we see a lot of fluctuations 
  in the accuracy/loss batch on batch basis
5.Large Batch size -- Batch Size tends to approximate the gradient. Although expectation of gradient is still the same for all batch sizes, variance changes a lot. 
  We will experiment with batch sizes as well

Steps taken to achieve the goal --
1. Started with swapping 1x1 onv layer and maxpool and didnt see any difference. So, used 1x1 conv layer after maxpool to stay consistent with lectures.
2. Removed dropout before softmax as it was creating nuisance to understand the achieved training accuracy and loss. The generalization gap became clearer after removing dropout
3. Reduced last conv layer channels from 16 to 10. Parameters reduced to 14.65K parameters. The accuracy achieved ~99.47% 
4. Went a little further in reducing parameters by reducing channels in 2nd last conv layer. Was able to achieve 99.4% but barely. Even could params till 9.9K params. 
5. In addition, tried tweaking with batch size. Was able to achieve 99.4% accuracy with higher batch size as it leads to a bit of overfitting. High batch size coupled with reduced 
model complexity was enough to achieve the desired accuracy but the accuracy was fluctuating a lot during training. 
6. Also tried removing learning rate scheduler but released that with fixed learning rate the accuracy kept fluctuating a lot during the end. 
   This meant during the end, there was a need to reduce the learning rate.
7. Also tried reducing dropout when I tried reducing model capacity. This helped to achieve 99.4% accuracy as it helped the lower capacity model overfit
7. Finally, submitting the one with 14.65K parameters as it had monotonically increasing accuracy. Also, didn't want to experiment with final submission.


Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 26s 432us/step - loss: 0.3969 - acc: 0.9252 - val_loss: 0.0946 - val_acc: 0.9836
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 12s 197us/step - loss: 0.1104 - acc: 0.9794 - val_loss: 0.0539 - val_acc: 0.9889
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 12s 197us/step - loss: 0.0767 - acc: 0.9840 - val_loss: 0.0451 - val_acc: 0.9905
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 12s 193us/step - loss: 0.0618 - acc: 0.9866 - val_loss: 0.0334 - val_acc: 0.9920
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 12s 193us/step - loss: 0.0514 - acc: 0.9883 - val_loss: 0.0301 - val_acc: 0.9920
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 12s 195us/step - loss: 0.0460 - acc: 0.9889 - val_loss: 0.0289 - val_acc: 0.9923
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 12s 193us/step - loss: 0.0402 - acc: 0.9906 - val_loss: 0.0221 - val_acc: 0.9940
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 12s 196us/step - loss: 0.0381 - acc: 0.9904 - val_loss: 0.0243 - val_acc: 0.9927
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 12s 197us/step - loss: 0.0358 - acc: 0.9909 - val_loss: 0.0214 - val_acc: 0.9946
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 11s 190us/step - loss: 0.0334 - acc: 0.9914 - val_loss: 0.0211 - val_acc: 0.9939
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 12s 198us/step - loss: 0.0317 - acc: 0.9914 - val_loss: 0.0251 - val_acc: 0.9930
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 12s 195us/step - loss: 0.0297 - acc: 0.9921 - val_loss: 0.0186 - val_acc: 0.9945
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 12s 193us/step - loss: 0.0289 - acc: 0.9923 - val_loss: 0.0208 - val_acc: 0.9937
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 12s 193us/step - loss: 0.0266 - acc: 0.9929 - val_loss: 0.0189 - val_acc: 0.9947
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 11s 191us/step - loss: 0.0268 - acc: 0.9925 - val_loss: 0.0188 - val_acc: 0.9941
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 12s 196us/step - loss: 0.0248 - acc: 0.9935 - val_loss: 0.0190 - val_acc: 0.9939
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 12s 195us/step - loss: 0.0242 - acc: 0.9932 - val_loss: 0.0183 - val_acc: 0.9941
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 11s 191us/step - loss: 0.0232 - acc: 0.9937 - val_loss: 0.0185 - val_acc: 0.9946
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 12s 196us/step - loss: 0.0233 - acc: 0.9936 - val_loss: 0.0192 - val_acc: 0.9943
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 12s 195us/step - loss: 0.0221 - acc: 0.9940 - val_loss: 0.0173 - val_acc: 0.9949



[0.01725905814692378, 0.9949]



########################################################################################################################################################################

Week3:

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




