# DREAM-Net
A Pytorch-Version for DREAM-Net. Some details are different from the original paper.

1. Install torch-randon (https://github.com/matteo-ronchetti/torch-radon)
2. Generate simulated sparse-view CT data: gen_ld_data.py
3. Generate training data: make_proj_img_list.py
4. Train and Validation: train_dreamnet.py
5. Test: test_dreamnet.py

Please cite the following references:
1. TorchRadon: Fast Differentiable Routines for Computed Tomography
2. DREAM-Net: Deep Residual Error iterAtive Minimization Network for Sparse-View CT Reconstruction
