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
6. Finally, submitting the one with 14.65K parameters as it had monotonically increasing accuracy. Also, didn't want to experiment with final submission.