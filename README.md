# Coordinate-based neural representations for Computational Adaptive optics (CoCoA) in widefield microscopy
This is a public repository for the self-supervised machine learning method, termed <u>Co</u>ordinate-based neural representations for <u>Co</u>mputational <u>A</u>daptive optics (CoCoA). CoCoA is designed to jointly estimate wavefront aberration and structures based on widefield fluorescence microscopy. The corresponding arXiv paper can be found <a href="https://arxiv.org/abs/2307.03812">here</a>. Please note that this repository is continuously being updated.

## **Abstract**
Widefield microscopy is widely used for non-invasive imaging of biological structures at subcellular resolution. When applied to complex specimen, its image quality is degraded by sample-induced optical aberration. Adaptive optics can correct wavefront distortion and restore diffraction-limited resolution but require wavefront sensing and corrective devices, increasing system complexity and cost. Here, we describe a self-supervised machine learning algorithm, CoCoA, that performs joint wavefront estimation and three-dimensional structural information extraction from a single input 3D image stack without the need for external training dataset. We implemented CoCoA for widefield imaging of mouse brain tissues and validated its performance with direct-wavefront-sensing-based adaptive optics. Importantly, we systematically explored and quantitatively characterized the limiting factors of CoCoA's performance. Using CoCoA, we demonstrated the first in vivo widefield mouse brain imaging using machine-learning-based adaptive optics. Incorporating coordinate-based neural representations and a forward physics model, the self-supervised scheme of CoCoA should be applicable to microscopy modalities in general.

## System Requirements
### Python Dependencies
    python-3.6.0
    numpy-1.19.2
    torch-1.8.0
    matplotlib-3.3.2
    scipy-1.5.2
    scikit-image-0.17.2
    scikit-learn-0.23.2
    h5py-2.10.0

## Installation Guide
Provided that all package dependencies are satisfied, no additional installation is necessary. The file `/main/wf_cocoa_demo.py` is self-contained and allows for the estimation of both structure and aberration from a provided 3D input image stack.

## Code
In the '/main/' folder, you will find the code written in PyTorch. The associated datasets can be found in the '/source/beads/' folder, which contains three distinct sets:

1. Reference image stack: 0-Mode678_rms_0nm_20230319_145228
2. Image stack with external aberration on top of the reference: 3A-Mode678_rms_75nm_20230319_145837
3. Ground truth, which represents externally provided aberration data.

The code demonstrates the aberration estimation from the provided the aberrated 3D image stack. It then evaluates the estimation accuracy by comparing it with the ground truth. The anticipated result is an aberration estimate that closely aligns with the ground truth aberration. Additionally, users can display the reconstructed structure via the _out_x_m_ variable. On average, the processing time for each input image stack is approximately 2 minutes using a machine equipped with a Tesla V100 GPU and Intel Xeon Gold 6248 CPU. For more detail in computation time, please check  Tables 1 and 2 in Supplementary Material.

In order to process other datasets in the '/source/' folder, e.g. fixed mouse brain slice (Figure 2) and mouse brain in vivo (Figure 5), the same code can be used but with different sets of hyperparameters (please refer to Table 1 in Supplementary Material for more detail).

Due to repository storage limitations (less than 1GB in Github), datasets for remaining figures in the main article and supplementary material are not available here. Please email the corresponding authors (Iksung Kang and Qinrong Zhang) for access.


## Citation
If you find the paper useful in your research, please consider citing the paper:

    @article{kang2023coordinate,
      title={Coordinate-based neural representations for computational adaptive optics in widefield microscopy},
      author={Kang, Iksung and Zhang, Qinrong and Yu, Stella X and Ji, Na},
      journal={arXiv preprint arXiv:2307.03812},
      year={2023}
    }

