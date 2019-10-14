# Feedfroward Network for Channel Mapping
The available codes show the effect of increasing the training set size on the performance of the neural network, which is the second figure in the results section of our paper ["Deep Learning for TDD and FDD Massive MIMO: Mapping Channels in Space and Frequency"](https://arxiv.org/abs/1905.03761), submitted for publication to Asilomar 2019. They also provide the foundation to reproduce the other results and more...

![Figure6](https://github.com/malrabeiah/DL-Massive-MIMO/blob/master/ChannelMapping/FDDTDDFigure6V2.png)
# Requirements
1) MATLAB Deep Learning toolbox
2) NVIDIA GPU with the corresponding CUDA toolkit (code could be run on a CPU, but it'll be very slow).
3) Two datasets could be generated using the [DeepMIMO dataset](https://github.com/DeepMIMO/DeepMIMO-codes). The settings are mentioned in the experiemental-results section of the paper above and are listed down here:

| Parameter | Value |
| -------- | ------ |
| Name of scenario | I1_2p4 and I1_2p5             |
| Active BSs    |            1 to 64               |   
| Active users     |  Row 1 to 502                 |
| Number of BS antennas in (x, y, x)  | (1,1,1)    |
| System BW | 0.02 GHz                             |
| Number of OFDM sub-carriers | 64                 |
| OFDM sampling factor | 1                         |
| OFDM limit | 16                                  |
| Number of Paths | 1 or 5                         |

# Reproducing The Figure:
1) Generate a dataset for scenario I1_2p4 using the settings in the table above--number of paths should be 1.
2) Organize the data into a MATLAB structure named "rawData" with the following fields: channel and userLoc. "channel" is a 3D array with dimensions: # of antennas X # of sub-carriers X # of users while "userLoc" is a 2D array with dimensions: 3 X # of users.
3) Save the data structure into a .mat file.
4) In the file main, set the option: options.rawDataFile1 to point to the .mat file.
5) Run main.m

# Citation
If you use the above codes or some of them, please cite the following work:
```
@ARTICLE{2019arXiv190503761A,
       author = {{Alrabeiah}, Muhammad and {Alkhateeb}, Ahmed},
        title = "{Deep Learning for TDD and FDD Massive MIMO: Mapping Channels in Space and Frequency}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Information Theory, Electrical Engineering and Systems Science - Signal Processing},
         year = "2019",
        month = "May",
          eid = {arXiv:1905.03761},
        pages = {arXiv:1905.03761},
archivePrefix = {arXiv},
       eprint = {1905.03761},
 primaryClass = {cs.IT},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190503761A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```



# License
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
