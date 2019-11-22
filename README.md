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

