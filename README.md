## Gaussian Process Implicit Surface (GPIS) Implementation for [Touch-GS: Visual-Tactile Supervised 3D Gaussian Splatting](https://armlabstanford.github.io/touch-gs)

## :classical_building:	Dependencies
```shell
git clone https://github.com/armlabstanford/GPIS.git
cd GPIS

# create conda environment
conda create -n GPIS python=3.8
conda activate GPIS
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

```

## :floppy_disk: Datasets
The dataset used in our [paper](https://arxiv.org/abs/2403.09875) is available at [Google Drive](https://drive.google.com/file/d/1nQfewPWBt19At1Nep6K-1aIHrw7V1IYI/view?usp=sharing). This dataset includes the combined point cloud and normals from [DenseTact](https://ieeexplore.ieee.org/document/10161150) for real data. For synthetic data we include the raw depth images rendered from blender. Download and extract the data to the `data` folder. If you don't have one, make it with `mkdir data`.
```
├── data
│   ├── real_bunny
│   │   ├── ... 
│   ├── syn_bunny
│   │   ├── ... 
```

## :rocket:	 Usage

### Real Data
```shell 
python real_scene.py {PARAM_FILE}
```

### Syn Data
```shell 
python syn_scene.py {PARAM_FILE}
```

### For example
```shell 
python real_scene.py data/real_bunny/params_bunny.json
```

Data is outputted to `output/{depth, var}` and will include both GPIS estimated depths and uncertainties. 

## :satellite: Citation
```
@article{swann2024touchgs,
  author    = {Aiden Swann and Matthew Strong and Won Kyung Do and Gadiel Sznaier Camps and Mac Schwager and Monroe Kennedy III},
  title     = {Touch-GS: Visual-Tactile Supervised 3D Gaussian Splatting},
  journal   = {arXiv},
  year      = {2024},
}
```
