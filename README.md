# Coordinate-based neural representations for computational adaptive optics in widefield microscopy
This is the public repository for CoCoA, a self-supervised computational adaptive optics method for widefield microscopy. CoCoA (Coordinate-based neural representations for Computational Adaptive optics) jointly estimates the wavefront aberration and sample structure based on a single image stack acquired from widefield fluorescence microscopy.

The paper describing CoCoA is published in _Nature Machine Intelligence_ and can be found <a href="https://www.nature.com/articles/s42256-024-00853-3">here</a>.

Please note that this repository is continuously updated. If you encounter any issues downloading the source files, you can download a ZIP archive of the repository <a href="https://drive.google.com/file/d/18mbbeQRcXFfIs9I-bkSXHfVo_RfeTHEA/view?usp=sharing">here</a> (available on Google Drive).

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
As long as all package dependencies are met, no additional installation is required. The file `/main/wf_cocoa_demo.ipynb` is self-contained and enables the estimation of both structure and aberration from a given 3D input image stack.

## Code
The `/main/` folder contains code written in PyTorch, while the associated datasets are located in the `/source/beads/` folder. This folder includes three distinct datasets:

1. Reference image stack: 0-Mode678_rms_0nm_20230319_145228
2. Aberrated image stack (with external aberration): 3B-Mode678_rms_75nm_20230319_145914
3. Ground truth: External aberration applied to DM (deformable mirrors).

The code estimates wavefront aberrations from the provided aberrated 3D image stack and evaluates its accuracy by comparing the result with the ground truth. The expected outcome is an aberration estimate closely matching the ground truth. Additionally, users can visualize the reconstructed structure using the _out_x_m_ variable.

On average, processing each image stack takes approximately 2 minutes on a machine equipped with a Tesla V100 GPU and an Intel Xeon Gold 6248 CPU (or 1.5 minutes with an RTX 4090 GPU and an Intel i9-13900K CPU.) For a more detailed breakdown of computation times, refer to Tables 1 and 2 in the Supplementary Material.

To process other datasets in the `/source/` folder, such as the fixed mouse brain slice (Figure 2) and mouse brain in vivo (Figure 5), the same code can be used with different hyperparameter settings (see Table 1 in the Supplementary Material for details).

Due to GitHub’s repository storage limitation (< 1GB), datasets for additional figures in the main article and supplementary material are not hosted here. For access, please contact the corresponding authors, Iksung Kang and Qinrong Zhang.

## Citation
If you find the paper useful in your research, please consider citing the paper:
    
    @article{kang_coordinate-based_2024,
    	title = {Coordinate-based neural representations for computational adaptive optics in widefield microscopy},
    	volume = {6},
    	issn = {2522-5839},
    	url = {https://doi.org/10.1038/s42256-024-00853-3},
    	doi = {10.1038/s42256-024-00853-3},
    	abstract = {Widefield microscopy is widely used for non-invasive imaging of biological structures at subcellular resolution. When applied to a complex specimen, its image quality is degraded by sample-induced optical aberration. Adaptive optics can correct wavefront distortion and restore diffraction-limited resolution but require wavefront sensing and corrective devices, increasing system complexity and cost. Here we describe a self-supervised machine learning algorithm, CoCoA, that performs joint wavefront estimation and three-dimensional structural information extraction from a single-input three-dimensional image stack without the need for external training datasets. We implemented CoCoA for widefield imaging of mouse brain tissues and validated its performance with direct-wavefront-sensing-based adaptive optics. Importantly, we systematically explored and quantitatively characterized the limiting factors of CoCoA’s performance. Using CoCoA, we demonstrated in vivo widefield mouse brain imaging using machine learning-based adaptive optics. Incorporating coordinate-based neural representations and a forward physics model, the self-supervised scheme of CoCoA should be applicable to microscopy modalities in general.},
    	number = {6},
    	journal = {Nature Machine Intelligence},
    	author = {Kang, Iksung and Zhang, Qinrong and Yu, Stella X. and Ji, Na},
    	month = jun,
    	year = {2024},
    	pages = {714--725},
    }
