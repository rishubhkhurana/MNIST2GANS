
score=[0.04337654851005741, 0.9923]

Definitions-

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




