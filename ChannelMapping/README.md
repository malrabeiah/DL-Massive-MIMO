# Feedfroward Network for Channel Mapping
The available codes show the effect of increasing the training set size on the performance of the neural network, which is the second figure of our paper titled "Deep Learning for TDD and FDD Massive MIMO:
Mapping Channels in Space and Frequency", submitted for publication to Asilomar 2019. They also provide the foundation to reproduce the other results and more...

![Figure6](https://github.com/malrabeiah/DL-Massive-MIMO/blob/master/ChannelMapping/FDDTDDFigure6V2.png)
# Requirements
1) MATLAB Deep Learning toolbox
2) Dataset should be generated using the [DeepMIMO dataset](http://www.deepmimo.net). The settings are mentioned in the experiemental-results section of the paper above and are listed down here:

| Parameter | Value |
| -------- | ------ |
| Name of scenario | I_1p2.4 and I_1p2.5 |
| Number of BSs    |             64                |   
| Active users     |  Row 1 to 502                 |
| Number of BS antennas in (x, y, x)  | (1,1,1)    |
| System BW | 0.02 GHz                             |
| Number of OFDM sub-carriers | 64                 |
| OFDM sampling factor | 1                         |
| OFDM limit | 16                                  |
# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
