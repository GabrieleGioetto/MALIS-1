# MALIS-Project

Project for the MALIS course at EURECOM.

In the paper [Vessel-CAPTCHA](https://arxiv.org/abs/2101.09321) it is proposed a new architecture for classifying 2D patch images in order to know where a vessel is located
inside a brain’s image and for extracting vessels combining human-annotations, a segmentation network and a classification
network. We worked on a similar scenario, but instead of brain images we used eyes images from the DRIVE dataset. In
both scenarios there is the problem that labeling vessels at a patch level is time expensive and very often human’s expertise is
required. For this reason, our work is focused on reducing the burden of the annotation process through Active Learning 
