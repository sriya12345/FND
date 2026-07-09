# Fake News Detector
Objective:
To detect whether a given image and its caption convey the same message or if there's a misalignment.
CLIP embeddings are used here. CLIP and an MLP classifier is used on top of it.

What is CLIP? 
CLIP - Contrastive Language Image Pretraining.
It is a dual-encoder neural network. It works on the concept of contrastive learning where it brings similar objects together and dissimilar ones farther apart in the latent space. 

It has a image encoder (ResNet50, ResNet101 etc) and a text encoder (Transformer encoder) which take in images and texts respectively and produce embeddings. Then the similarity between these embeddings is measured and they are placed accordingly in the embedding space.

Why CLIP?

A typical CNN categorizes images into fixed labels it was trained on. It requires a lot of manually captioned data and takes a lot of training time. It doesn't understand Natural Language. It also cannot classify inputs into unknown labels.

But CLIP is not a classifier like CNN. It learns the concept of an object. It understands Natural Language. It maps image and text embeddings into the same embedding space, keep the similar ones closer and dissimilar ones farther apart. It enables Zero Shotclassification.

Zero Shot Classification means that the model can handle class labels with zero examples. 
The class labels here are usually text prompts. It can categorize images into unknown labels.

In training, it converts image and text into embeddings and computes the cosine similarity and then gets the contrastive loss. After training, the weights are frozen. 
In inference, it takes the inputs and computes the similarities between them.

CLIP can be used for zero shot classification, Image retrieval and Text retrieval.

Dataset used: NewsCLIPpings (public, MIT)
