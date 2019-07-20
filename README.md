# FANforVOS
## Fast Attention Network for Video Object Segmentation
Semi-supervised video object segmentation (VOS) has obtained significant progress in recent years. The general purpose of VOS methods is to segment objects in video sequences provided with a single annotation in the first frame. However, many of the recent successful methods heavily fine-tuning on the object mask in the first frame, which decreases their efficiency. In this work, to address this issue, we propose a siamese encoder-decoder network with the attention mechanism requiring only one forward pass to segment the target object in a video. Specifically, the encoder part generates a low-resolution mask with smoothed boundaries, and the decoder part refines the details of initial mask and integrates lower-level features progressively. Besides, to obtain accurate segmentation results, we sequentially apply the attention module on multi-scale feature maps for refinement. Several experiments are carried out on three challenging datasets (i.e., DAVIS2016, DAVIS-2017 and SegTrackv2) to show that our method achieves competitive performance against the state-of-the-arts. 

**Our main contributions include the following:**
- The proposed model requires only one forward pass through the siamese encoder-decoder network to produce all parameters needed for the segmentation model to adapt to the specific object instance.
- We design an attention module to guide the network to focus on the target object in the current frame which helps to improve accuracy.
- Extensive experiments are conducted on three datasets, namely, DAVIS-2016, DAVIS-2017, and SegTrackv2, to demonstrate that the proposed method achieves favorable performance compared to the state-of-the-arts.		
 
## Network Architecture
We construct the model as a Siamese encoder-decoder structure which can efficiently handle four inputs and produce a segmentation mask. 
The network consists of two encoders with shared parameters, a global convolution block, and a decoder. The designed network is fully convolutional, which can handle arbitrary input size and generate a sharp output mask. Given a reference frame with the ground-truth mask, the goal of our method is to automatically segment the target object through the entire video sequence. The key idea of our method is to exploit the annotated reference frame and the current frame with the previous mask estimation to a deep network. The network detects the target object by matching the appearance at the reference frame and the current frame. Meanwhile,  the previous mask is tracked by referencing the previous target mask in the current frame. The architecture of our network is shown as follows,
<img src="https://github.com/djzgroup/RSforWordTranslation/blob/master/network.jpg" width="800">

## Acknowledgment
This work was supported in part by the National Natural Science Foundation of China under Grant 61702350 and Grant 61802355.
