python main.py --base configs/celebahq-ldm-vq-4_multistage224-256-192-128.yaml -t --gpus 0,1,2,
python scripts/sample_diffusion.py -r DIR_TO_CKPT -l DIR_TO_RESULT_FOLDER -e 0 -c 20 --batch_size 48
fidelity --gpu 0 --fid --samples-find-deep --input1 data/celebahq/ --input2 DIR_TO_RESULT_FOLDER