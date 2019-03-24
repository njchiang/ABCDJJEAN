# CNN

CNN code

To run on other machines, you might need to set up the volume labels and ground truth directory properly...

modify the cfg file at `cfgs/vanilla.json`

train using:
`python train.py --cfg=cfgs/vanilla.json --gpu=0`

predict validation using: (faster)
`python predict.py --cfg=cfgs/vanilla.json --gpu=0 --guid=D:\\fmri\\ABCD\\data\\jeff_val_guids.txt --mode=val`
OR
`python export_model.py --cfg=cfgs/vanilla.json --export_path=D:\\fmri\\ABCD\\weights`
change the weights in cfg, then run
`python deploy.py --cfg=cfgs/vanilla.json --gpu=0 --guid=D:\\fmri\\ABCD\\data\\jeff_val_guids.txt --mode=val`

evaluate predictions using
`python evaluate.py --predictions=D:\\fmri\\ABCD\\outputs\\1553400163-val`
`python evaluate.py --predictions=D:\\fmri\\ABCD\\outputs\\model.ckpt-10000-val`

predict testing using
`python predict.py --cfg=cfgs/vanilla.json --gpu=0 --guid=D:\\fmri\\ABCD\\data\\jeff_testing_guids.txt --mode=testing`
