<br />
<p align="center">

  <h1 align="center">🧥LYCB: Leave Your Clothes Behind</h1>
  <h4 align="center"><a href="https://github.com/datacrisis">Keifer Lee</a>, <a href="https://github.com/iamshubhamgupto">Shubham Gupta</a>, <a href="">Karan Sharma</a></h4>
  
  <h5 align="center">&emsp; <a href="https://iamshubhamgupto.github.io/LYCB/assets/pdf/report.pdf"> Report </a></h5>
</p>


## Abstract
![Pipeline of Leave Your Clothes Behind (LYCB)](./assets/pipeline.png)
<p align="center">
  <b>Pipeline of Leave Your Clothes Behind (LYCB)</b>
</p>

## Results & Sample Weights
Below is an illustration of the input / output at each stage of the pipeline with custom data.
![Sample output](https://github.com/IamShubhamGupto/LYCB/blob/main/assets/merged_animation.gif)
<p align="center">
<b>Sample data at each stage. From left to right - Monocular input sequence, SAM extracted mask, NeRF2Mesh reconstructed mesh and test-fit with cloth simulation in Blender</b>
</p>

Sample data used in illustration above and the corresponding trained implicit model and reconstructed mesh can be found [here](https://drive.google.com/file/d/1nKHaewiDw_M1wOnBDXXRA_i0nXSzp8LR/view?usp=share_link)



## Getting Started

### Installation
For LYCB, `NeRF2Mesh`'s dependencies and `Segment-Anything` are key pre-requisites. Optionally, if you would like to use your own data (without prepared camera parameters), `COLMAP` will be required as well.

```python
#Installing key requirements

#NeRF2Mesh
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch 
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install git+https://github.com/facebookresearch/pytorch3d.git

#SAM
pip install git+https://github.com/facebookresearch/segment-anything.git
```

For COLMAP installation, check out https://colmap.github.io/install.html


### Running LYCB
1. If you are starting from a raw video (e.g. MP4) file or set of images without known camera parameters, then you will have to run `COLMAP` first to generate said parameters. </br>
**Note**: Dataset will need to be in the format of [nerf-synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) or [MIP-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)'s dataset. Also please do run COLMAP on the raw images (e.g. not masked clothing images) to ensure proper feature matching with COLMAP.
```bash
#Example for video
python scripts/colmap2nerf.py --video ./path/to/video.mp4 --run_colmap --video_fps 10 --colmap_matcher sequential

#Example for images
python scripts/colmap2nerf.py --images ./path/to/images/ --run_colmap --colmap_matcher exhaustive
```

2. Assuming that you the corresponding dataset (e.g. images and `transforms.json`) prepared, next you will have to run `SAM` to extract the clothing.
  - Make sure to keep the sequence of input-output images in the same order to ensure that the information within `transforms.json` corresponds to the right frame index.
  - Additionally, SAM by default is able to generate up to 3 masks per input with its multi-mask option, which will be very handy in isolating the clothing of interests only. By default the extraction script will select the highest scoring mask amongst the 3 generated mask, but you may have to fiddle around with the script to ensure that only the clothing item of interest is extracted consistently across all frames; could consider adding support to automatically do said matching in the futuer.
  - Lastly, the extraction script assumes that the target (e.g. clothing of interest) is consistently held at the frame center consistently across all frames; feel free to fiddle around the positive and negative keypoints within the extraction script to change the expected target focal point as required.
```python
#Extract target clothing with SAM; checkout extraction.py for more args / options
python extraction.py --data /path/to/images \
                      --output /output/path \
                      --enable_neg_keypoints
```

3. Once the target clothing has been extracted and verified (e.g. extractions are consistent and without occlusion), proceed repack the data according to the chosen dataset format; for e.g.
```
#E.g. nerf-synthetic format

- dataset_root
  |_ images/
    |_ img_0001.png
    |_ img_0002.png
    |_ ...
  |_ transforms.json
  
```

4. Generate neural-implicit representation and subsequently reconstruct mesh from queried points. For more information, advanced options and tip, checkout the original [NeRF2Mesh](https://github.com/ashawkey/nerf2mesh) implementation here. Note that you will probably have to fine-tune the parameters here according to your scene / object.
```python

#Stage 0 | Fit radiance field, perform volumetric rendering and extract coarse mesh
python nerf2mesh.py path/to/dataset --workspace testrun --lambda_offsets 1 --scale 0.33 --bound 3 --stage 0 --lambda_tv 1e-8 --lambda_normal 1e-1 --texture_size 2048 --ssaa 1 #Enforce coarser texture_size and limit SSAA for headless rendering

#Stage 1 | Fine-tune coarse model, generate mesh, rasterize, and clean
python nerf2mesh.py path/to/dataset --workspace testrun --lambda_offsets 1 --scale 0.33 --bound 3 --stage 1 --lambda_normal 1e-1 --texture_size 2048 --ssaa 1

```

5. Clean up generated mesh with any method / software of your choice as required - e.g. [MeshLab](https://www.meshlab.net). Now you have the final 3D mesh of the piece of clothing you so desired!

6. [Optional] Run clothing simulation to perform virtual try-on on desired target with the software of your choice; note that you will have to provide your own target try-on body and rig it (if applicable) accordingly. For our demo, we used [Blender](https://www.blender.org) for the clothing simulation and a simple non-descriptive male dummy model as the try-on target.

7. All done!


## Acknowledgements & References
The project is built on top of ashawkey's PyTorch implementation of NeRF2Mesh [here](https://github.com/ashawkey/nerf2mesh).

```
#NeRF2Mesh
@article{tang2022nerf2mesh,
  title={Delicate Textured Mesh Recovery from NeRF via Adaptive Surface Refinement},
  author={Tang, Jiaxiang and Zhou, Hang and Chen, Xiaokang and Hu, Tianshu and Ding, Errui and Wang, Jingdong and Zeng, Gang},
  journal={arXiv preprint arXiv:2303.02091},
  year={2022}
}

#ashawkey's PyTorch implementation
@misc{Ashawkey2023,
  author = {Ashawkey},
  title = {nerf2mesh},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ashawkey/nerf2mesh}}
}

#COLMAP
@inproceedings{schoenberger2016sfm,
    author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
    title={Structure-from-Motion Revisited},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2016},
}
@inproceedings{schoenberger2016mvs,
    author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
    title={Pixelwise View Selection for Unstructured Multi-View Stereo},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2016},
}

#Segment-Anything (SAM)
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}

```

## LICENSE
This project is [MIT](LICENSE) Licensed
