# Few-shot Fault Diagnosis 
The detail of FSM3 can be find in the following paper.
[Metric-based meta-learning model for few-shot fault diagnosis under multiple limited data conditions](https://doi.org/10.1016/j.ymssp.2020.107510)

The datasets used in this paper can be download by the link [Bearing and Gearbox Datasets for Few-shot Learning Fault Diagnosis](https://data.mendeley.com/datasets/2znkfwhzs7/1)

Generally, anyone could download original CWRU dataset easily. So, I will not provide the link.

The original detial descriptation and information of the PHM2009 Competition can be found the original link is as follow:
https://phmsociety.org/data-analysis-competition/

If the original data link is unavailable, one can download the original data in the following link:
https://github.com/mingzhangPHD/Supplementary-Dataset/tree/main/PHM2009%20Gearbox/PHM_Society_2009_Competition_Expanded_txt

From the paper, one can see there are four scenarios, the first two are for bearing fault and the latter two are for gearbox fault, the detailed information can be found in the paper. So, the codes in this page are split into four files, one can download them and the dataset to understand the proposed algorithm. These codes are implemented by Pytorch.

Now, I will explain the files in the files. 

- "main.py" is the main program, you can run this text by:
```
python run_main_TAN.py
```
- "models.py" is the designed model function used in main program;
- "options.py" has the hyper-parameters and other information like "data path" etc., one can change the parameter here.
- "backbone.py" defined the FeatureExtractor mapping by neural network. 


# Citation

If you use this code and datasets for your research, please consider citing:

```
@article{WANG2021107510,
title = "Metric-based meta-learning model for few-shot fault diagnosis under multiple limited data conditions",
journal = "Mechanical Systems and Signal Processing",
volume = "155",
pages = "107510",
year = "2021",
issn = "0888-3270",
doi = "https://doi.org/10.1016/j.ymssp.2020.107510",
url = "http://www.sciencedirect.com/science/article/pii/S0888327020308967",
author = "Duo Wang and Ming Zhang and Yuchun Xu and Weining Lu and Jun Yang and Tao Zhang",
keywords = "Metric-based meta-learning, Few-shot learning, Feature space, Fault diagnosis, Limited data conditions",
abstract = "The real-world large industry has gradually become a data-rich environment with the development of information and sensor technology, making the technology of data-driven fault diagnosis acquire a thriving development and application. The success of these advanced methods depends on the assumption that enough labeled samples for each fault type are available. However, in some practical situations, it is extremely difficult to collect enough data, e.g., when the sudden catastrophic failure happens, only a few samples can be acquired before the system shuts down. This phenomenon leads to the few-shot fault diagnosis aiming at distinguishing the failure attribution accurately under very limited data conditions. In this paper, we propose a new approach, called Feature Space Metric-based Meta-learning Model (FSM3), to overcome the challenge of the few-shot fault diagnosis under multiple limited data conditions. Our method is a mixture of general supervised learning and episodic metric meta-learning, which will exploit both the attribute information from individual samples and the similarity information from sample groups. The experiment results demonstrate that our method outperforms a series of baseline methods on the 1-shot and 5-shot learning tasks of bearing and gearbox fault diagnosis across various limited data conditions. The time complexity and implementation difficulty have been analyzed to show that our method has relatively high feasibility. The feature embedding is visualized by t-SNE to investigate the effectiveness of our proposed model."
}
```






