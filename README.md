# FEA-T
This repository contains implementations of FEA-T for our paper "Abandon locality: Frame-wise Embedding Aided Transformer for Automatic Modulation Recognition".

We will upload the complete code as soon as possible after sorting and commenting.

# Abstract

Automatic modulation recognition (AMR) has been considered as an efficient technique for non-cooperative communication and intelligent communication. In this work, we propose a modified transformer-based method for AMR, called frame-wise embedding aided transformer (FEA-T), aiming to extract the global correlation feature of the signal to obtain a higher classification accuracy as well as a lower time cost. To enhance the global modeling capability of the transformer, we design a frame-wise embedding module (FEM) to aggregate more samples into a token in the embedding stage to generate a more efficient token sequence. We also present the optimal frame length by analyzing the representation ability of each transformer layer for a better trade-off between the speed and the performance. Moreover, we design a novel dual-branch gate linear unit (DB-GLU) scheme for the feed-forward network of the transformer to reduce the model size and enhance the performance. Experimental results on RadioML2018.01A datasets demonstrate that the proposed method outperforms state-of-the-art works in terms of recognition accuracy and running speed. 
