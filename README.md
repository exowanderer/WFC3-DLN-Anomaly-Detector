# WFC3-DLN-Anomaly-Detector
Use Deep Learning and Public Hubble/WFC3 data to identify 'anomalies' in Space observations (for HST and JWST)

The most updated version of our deep learning, Hubble/WFC3 Image Anomaly Detector can be found as a [Colab Notebook](https://colab.research.google.com/drive/1P8W9fdWG99i8h9cbs5jwCLwFxAQsWFGw?usp=sharing).

This notebook downloads the Hubble Ultra Deep Field, and trains a convolutional autoencoder with it; then reconstructs subsections of the HUDF and compares them side-by-side with the original subframe for quality control.

Next we will inject anomalies into the images and determine the auto encoders sensitivity to these anomalies.
