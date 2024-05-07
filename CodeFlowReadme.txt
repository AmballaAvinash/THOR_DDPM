
core
    -Main.py:  run python core/Main.py --config_path ./projects/thor/configs/brain/thor.yaml
               setup experiments using the confif file

    -trainer.py: Base class for Traning


data 
    - ATLAS: stoke dataset with with and without masks


dl_utils: utils for data loading


model_zoo:
    - ddpm.py: Forward pass (Inference)
        from net_utils.simplex_noise import generate_noise
        from net_utils.nets.diffusion_unet import DiffusionModelUNet
        from net_utils.schedulers.ddpm import DDPMScheduler
        from net_utils.schedulers.ddim import DDIMScheduler 


        Anomaly mask (core THOR idea) is implemented in this file

    - vgg.py: vgg19 network



net_utils: 
    - nets: unet
    - schdulersL DDPM, DDIM
    - simplex_noise.py: generate simplex noise


optim:
    - losses: different losses
    - metrics: precision metric etc




projects/thor:
    - configs: 
        - brain: Main config file 
    - DDPMTrainer: derived class from core.trainer
    - DDIMTrainer: derived class from core.trainer
    - DownstreamEvaluatorAtlas: derived class from core.DownstreamEvaluator

transforms:
    -preprocessing 



