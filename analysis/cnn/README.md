# CNN

CNN code

To run on other machines, you will need to prepare the additional derived data and models.

Assuming you have the data for the project in `PROJECT_ROOT`,

1. Move the additional ROI output into `PROJECT_ROOT/results/`, along with the label files provided by the organizers.

2. Modify the cfg file at `cfgs/vanilla.json` so that the correct paths are set.

3. train using: `python train.py --cfg=cfgs/vanilla.json --gpu=0`

4. predict validation using: (faster)

`python predict.py --cfg=cfgs/vanilla.json --gpu=0 --guids=D:\\fmri\\ABCD\\data\\jeff_val_guids.txt --mode=val`

OR  (CURRENTLY NOT FUNCTIONING)

`python export_model.py --cfg=cfgs/vanilla.json --export_path=D:\\fmri\\ABCD\\weights`

change the weights in cfg, then run

`python deploy.py --cfg=cfgs/vanilla.json --gpu=0 --guids=D:\\fmri\\ABCD\\data\\jeff_val_guids.txt --mode=val`


5. evaluate predictions using

`python evaluate.py --predictions=D:\fmri\ABCD\outputs\1553474337-val.csv`

`python evaluate.py --predictions=D:\\fmri\\ABCD\\outputs\\model.ckpt-12000-val.csv`


6. predict testing using

`python predict.py --cfg=cfgs/vanilla.json --gpu=0 --guids=D:\\fmri\\ABCD\\data\\jeff_testing_guids.txt --mode=testing`
