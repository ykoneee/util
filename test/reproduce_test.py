from lightning_module import LModel

pretrained_model = LModel.load_from_checkpoint(
    "lightning_logs/version_15/checkpoints/v_acc=0.29-epoch=195.ckpt"
)

pretrained_model.freeze()
pretrained_model.eval()
