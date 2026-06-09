# Revisiting F-measure Optimization in Multi-Label Classification: A Sampling-based Approach

## Environment Setup

```bash
conda create -n mlc-f-spl python=3.10
conda activate mlc-f-spl
pip install -r requirements.txt
# Install PyTorch from https://pytorch.org/
```

## Data Preparation
We conducted experiments using six datasets, including four standard multi-label classification datasets from the [Mulan library](https://mulan.sourceforge.net/datasets-mlc.html) and two modern image datasets, VOC2007 and COCO2014.

### Mulan data
The Mulan datasets were downloaded and preprocessed using the [scikit-multilearn](http://scikit.ml/index.html) package.

```bash
# cd src/aux_scripts
python prepare_skml_data.py
```

### Image data
The two image datasets should be downloaded from their respective sources:
- [PASCAL VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- [MS COCO2014](https://cocodataset.org/#download)

After downloading, extract features and perform data splitting using the following command:
```bash
# cd src/aux_scripts
python prepare_image_data.py --dataset_name DATASET_NAME --data_root DATA_ROOT
```

## Run experiments
To run experiments for a single dataset, use the following command:
```bash
# cd src
python main.py --config configs/ar/yeast.json
```
Alternatively, to run experiments for all datasets, execute the provided script:
```bash
# cd exp
bash run_ar.sh
```
