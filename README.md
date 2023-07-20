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
