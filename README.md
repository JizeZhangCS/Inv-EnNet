# Inv-EnNet

## Requirements

```
conda create -n InvEnNet

conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

conda install tqdm tensorboard

python train.py -c ./config/default.json
```

## Usage

### Dataset Preparing

Download datasets (DICM, LIME, MEF, NPE, VV) and move them into `../datasets/`

### Testing process

1. Download the [pretrained model](https://drive.google.com/file/d/12cSA50_A5OMKONWL0McQYu2DNrwt4RfG/view?usp=sharing) from Google Drive and move it into `./saved/models/InvEnNet_default/`
2. Then run the following command:
    ```
    python test.py --resume ./saved/models/InvEnNet_default/checkpoint-epoch400.pth
    ```
3. Check `./saved/test` for enhanced images

### Training process

Run
```
python train.py -c ./config/default.json
```
