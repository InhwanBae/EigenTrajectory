## EigenTrajectory: Low-Rank Descriptors for Multi-Modal Trajectory Forecasting
This repository contains the code for the EigenTrajectory(𝔼𝕋) space applied to the traditional Euclidean-based trajectory predictors.

**[EigenTrajectory: Low-Rank Descriptors for Multi-Modal Trajectory Forecasting](https://inhwanbae.github.io/publication/eigentrajectory/)**
<br>
<a href="https://InhwanBae.github.io/">Inhwan Bae</a>,
<a href="https://www.cs.cmu.edu/~./jeanoh/">Jean Oh</a> and
<a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ">Hae-Gon Jeon</a>
<br>Accepted to 
<a href="https://iccv2023.thecvf.com/">ICCV 2023</a>

<div align='center'>
  <img src="img/eigentrajectory-model.svg" width=70%>
  <br>A common pipeline of trajectory prediction models and the proposed EigenTrajectory.
</div>


## 🌌 EigenTrajectory(𝔼𝕋) Space 🌌
* A novel trajectory descriptor based on Singular Value Decomposition (SVD), provides an alternative to traditional methods.
* It employs a low-rank approximation to reduce the complexity and creates a compact space to represent pedestrian movements.
* A new anchor-based refinement method to effectively encompass all potential futures.
* It can significantly improve existing standard trajectory predictors by simply replacing the Euclidean space.


## Model Training
### Setup
**Environment**
<br>All models were trained and tested on Ubuntu 18.04 with Python 3.7 and PyTorch 1.12.1 with CUDA 11.1.

**Dataset**
<br>Preprocessed [ETH](https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz) and [UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) datasets are included in this repository, under `./datasets/`. 
The train/validation/test splits are the same as those fond in [Social-GAN](https://github.com/agrimgupta92/sgan).

You can also download the dataset by running the following script.
```bash
./scripts/download_datasets.sh
```

**Baseline models**
<br>This repository supports 10 baseline models: 
[**AgentFormer**](https://arxiv.org/abs/2103.14023), 
[**DMRGCN**](https://ojs.aaai.org/index.php/AAAI/article/view/16174), 
[**GPGraph-SGCN**](https://arxiv.org/abs/2207.09953), 
[**GPGraph-STGCNN**](https://arxiv.org/abs/2207.09953), 
[**Graph-TERN**](https://ojs.aaai.org/index.php/AAAI/article/view/25759), 
[**Implicit**](https://arxiv.org/abs/2203.03057), 
[**LBEBM**](https://arxiv.org/abs/2104.03086), 
[**PECNet**](https://arxiv.org/abs/2004.02025), 
[**SGCN**](https://arxiv.org/abs/2104.01528) and 
[**Social-STGCNN**](https://arxiv.org/abs/2002.11927). 
We have included model source codes from their official GitHub in the `./baselines/` folder. 

If you want to add your own baseline model, simply paste the model code into the baseline folder and add a few lines of [initialization constructor](https://github.com/InhwanBae/EigenTrajectory/blob/main/baselines/pecnet/__init__.py) and [bridge](https://github.com/InhwanBae/EigenTrajectory/blob/main/baselines/pecnet/bridge.py) code.

### Train EigenTrajectory
To train our EigenTrajectory on the ETH and UCY datasets at once, we provide a bash script `train.sh` for a simplified execution.
```bash
./scripts/train.sh
```
We provide additional arguments for experiments: 
```bash
./scripts/train.sh -t <experiment_tag> -b <baseline_model> -c <config_file_path> -p <config_file_prefix> -d <space_seperated_dataset_string> -i <space_seperated_gpu_id_string>

# Supported baselines: agentformer, dmrgcn, gpgraphsgcn, gpgraphstgcnn, graphtern, implicit, lbebm, pecnet, sgcn, stgcnn
# Supported datasets: eth, hotel, univ, zara1, zara2

# Examples
./scripts/train.sh -b sgcn -d "hotel" -i "1"
./scripts/train.sh -b agentformer -t EigenTrajectory -d "zara2" -i "2"
./scripts/train.sh -b pecnet -c ./config/ -p eigentrajectory -d "eth hotel univ zara1 zara2" -i "0 0 0 0 0"
```
If you want to train the model with custom hyper-parameters, use `trainval.py` instead of the script file.
```bash
python trainval.py --cfg <config_file_path> --tag <experiment_tag> --gpu_id <gpu_id> 
```


## Model Evaluation
### Pretrained Models
We provide pretrained models in the [**release section**](https://github.com/InhwanBae/EigenTrajectory/releases/tag/v1.0). 
You can download all pretrained models at once by running the script. This will download the 10 EigenTrajectory models.
```bash
./scripts/download_pretrained_models.sh
```

### Evaluate EigenTrajectory
To evaluate our EigenTrajectory at once, we provide a bash script `test.sh` for a simplified execution.
```bash
./scripts/test.sh -t <experiment_tag> -b <baseline_model> -c <config_file_path> -p <config_file_prefix> -d <space_seperated_dataset_string> -i <space_seperated_gpu_id_string>

# Examples
./scripts/test.sh -b sgcn -d "hotel" -i "1"
./scripts/test.sh -b agentformer -t EigenTrajectory -d "zara2" -i "2"
./scripts/test.sh -b pecnet -c ./config/ -p eigentrajectory -d "eth hotel univ zara1 zara2" -i "0 0 0 0 0"
```

If you want to evaluate the model individually, you can use `trainval.py` with custom hyper-parameters. 
```bash
python trainval.py --test --cfg <config_file_path> --tag <experiment_tag> --gpu_id <gpu_id> 
```


## 📖 Citation
If you find this code useful for your research, please cite our papers :)

[**`DMRGCN (AAAI'21)`**](https://github.com/InhwanBae/DMRGCN) **|** 
[**`NPSN (CVPR'22)`**](https://github.com/InhwanBae/NPSN) **|** 
[**`GP-Graph (ECCV'22)`**](https://github.com/InhwanBae/GPGraph) **|** 
[**`Graph-TERN (AAAI'23)`**](https://github.com/InhwanBae/GraphTERN) **|** 
[**`EigenTrajectory (ICCV'23)`**](https://github.com/InhwanBae/EigenTrajectory)

```bibtex
@article{bae2023eigentrajectory,
  title={EigenTrajectory: Low-Rank Descriptors for Multi-Modal Trajectory Forecasting},
  author={Bae, Inhwan and Oh, Jean and Jeon, Hae-Gon},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
<details>
  <summary>More Information (Click to expand)</summary>

```bibtex
@article{bae2021dmrgcn,
  title={Disentangled Multi-Relational Graph Convolutional Network for Pedestrian Trajectory Prediction},
  author={Bae, Inhwan and Jeon, Hae-Gon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}

@inproceedings{bae2022npsn,
  title={Non-Probability Sampling Network for Stochastic Human Trajectory Prediction},
  author={Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@inproceedings{bae2022gpgraph,
  title={Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction},
  author={Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}

@article{bae2023graphtern,
  title={A Set of Control Points Conditioned Pedestrian Trajectory Prediction},
  author={Bae, Inhwan and Jeon, Hae-Gon},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}
```
</details>

<br>
