from tensorboard.plugins.hparams import api as hp


HP_LR = hp.HParam("lr", hp.Discrete([3e-4]))
# HP_LR = hp.HParam("lr", hp.Discrete([7e-4]))
HP_CLIP = hp.HParam("clip", hp.Discrete([0.3]))
HP_CLIP_VALUE = hp.HParam("clip_value", hp.Discrete([0.0]))
HP_ENTROPY_COEF = hp.HParam("entropy_coef", hp.Discrete([1e-5]))
HP_GRADIENT_NORM = hp.HParam("gradient_norm", hp.Discrete([10.0]))
