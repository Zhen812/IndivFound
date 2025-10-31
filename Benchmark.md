# IndivFound - Benchmark data and model checkpoints
- We provide data for model pre-training and fine-tuning and model checkpoints to facilitate model comparison. 
- Please replace the fields that need to be updated in config.yaml, such as data_dir, anno_path, found_model_ckpts, cls_model_ckpts, and regress_model_ckpts, with local paths.
- All required dependicies are listed in env.txt.

### 1. Data for pre-training of EEG-specific encoder
| Dataset | Official Download Link |
| :---: | :---: |
| AMIGOS | https://eecs.qmul.ac.uk/mmv/datasets/amigos/download.html |
| EEG Motor Movement/Imagery Dataset | https://physionet.org/content/eegmmidb/1.0.0/ |
| DEAP | https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html |
| DREAMER | https://zenodo.org/records/546113 |
| SEED | https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/ |
| SEED-IV | https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/ |
| SEED-V | https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/ |
| SEED-VIG | https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/ |


### 2. Data for pre-training of ECG-specific encoder
| Dataset | Official Download Link |
| :---: | :---: |
| CPSC2018 | https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/ |
| CPSC2018-Extra | https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/ |
| Georgia | https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/ |
| DREAMER | https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/ |
| INCART | https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/ |
| PTB-XL | https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/ |

### 3. Data for pre-training of IndivFound
| Dataset | Official Download Link |
| :---: | :---: |
| AMIGOS | https://eecs.qmul.ac.uk/mmv/datasets/amigos/download.html |
| BioVid-Part A & B | https://www.nit.ovgu.de/BioVid.html |
| DREAMER | https://zenodo.org/records/546113 |
| SEED | https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/ |
| SEED-VIG | https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/ |

### 4. Model checkpoints of pre-training.
The model checkpoint of pre-training can be downloaded [here](https://pan.baidu.com/s/1JgypPYNiz4PoC8N43nkoNw?pwd=jw87).


### 5. Pre-processed data and model checkpoints of downstream tasks
| Level | Task Category | Task | Dataset(link) | Checkpoint Link |
| :---: | :---: | :---: | :---: | :---: |
| Low-level | Internal Validation | Age Grouping | [Autonomic-Aging Dataset](https://pan.baidu.com/s/10cq6ej_WXxjZTg1I5kJJ4Q?pwd=he9f) | to be updated |
|  |  | BMI Grouping | Autonomic-Aging Dataset | to be updated |
| |  | Gender Decoding | Autonomic-Aging Dataset | to be updated |
|  | External Validation | Age Grouping | [ECG-ID Dataset](https://pan.baidu.com/s/1r5rW2_oxFRAKFkl1caf9Hg?pwd=tswu) | to be updated |
|  |  | Gender Decoding | ECG-ID Dataset | to be updated |
|  |  | Gender Decoding | Autonomic-Aging Dataset | to be updated |
|  | Cross Sub-Group Eval | Age Grouping | Autonomic-Aging Dataset | to be updated |
|  |  | BMI Grouping | Autonomic-Aging Dataset | to be updated |
|  |  | Gender Decoding | Autonomic-Aging Dataset | to be updated |
| Mid-level | Internal Validation | Myocardial Infarction Detect | [PTB Dataset](https://pan.baidu.com/s/1l6OXnTIqN3fHQkPniq9CvA?pwd=cjdh) | to be updated |
|  |  | Pain & Emotion Recognition | [BioVid-Part D Dataset](https://www.nit.ovgu.de/BioVid.html) | to be updated |
| |  | Sleep Staging | Sleep-Cassette Dataset (to be updated) | to be updated |
| |  | Stress Detection | [WESAD Dataset](https://pan.baidu.com/s/1FiZuR0ax943BgsiVYtsP5Q?pwd=2vx2) | to be updated |
|  | External Validation | Sleep Staging | Sleep-Telemetry Dataset (to be updated) | to be updated |
|  | Cross Sub-Group Eval |  Myocardial Infarction Detect | PTB Dataset | to be updated |
|  |  | Pain & Emotion Recognition | BioVid-Part D Dataset | to be updated |
| |  | Sleep Staging | Sleep-Cassette Dataset | to be updated |
| High-level | Internal Validation | Imagery Motor Classification | [HGD Dataset](https://pan.baidu.com/s/1cDXZCF8_ggCMlZhlFBLhfA?pwd=wjri) | to be updated |
|  |  | Surgery Performance Evaluation | [NIBIB-RPCCC-FLS Dataset](https://pan.baidu.com/s/1QCiViKKx4LzVVzCYteINzg?pwd=p6am) | to be updated |
|  | Cross Sub-Group Eval |  Surgery Performance Evaluation | NIBIB-RPCCC-FLS Dataset | to be updated |
| Realistic Scenario | Internal Validation | Fatigue Grading | FDZHD Dataset (to be updated) | to be updated |
|  | Cross Sub-Group Eval |  Fatigue Grading | FDZHD Dataset | to be updated |

Official dataset links:

- Autonomic-Aging Dataset: https://physionet.org/content/autonomic-aging-cardiovascular/1.0.0/
- ECG-ID Dataset: https://physionet.org/content/ecgiddb/1.0.0/
- PTB Dataset: https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/
- BioVid-Part D Dataset: https://www.nit.ovgu.de/BioVid.html
- Sleep-Cassette & Sleep-Telemetry Dataset: https://physionet.org/content/sleep-edfx/1.0.0/
- WESAD Dataset: https://www.kaggle.com/datasets/mohamedasem318/wesad-full-dataset
- HGD Dataset: https://braindecode.org/stable/generated/braindecode.datasets.HGD.html
- NIBIB-RPCCC-FLS Dataset: https://physionet.org/content/eeg-eye-gaze-for-fls-tasks/1.0.0/
