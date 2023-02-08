from train import train
from predictHardPanop import predict

if __name__ == '__main__':
    models = train()
    print("################ PREDICT ################")
    vr = ["/Volumes/DropSave/Tese/trainedModels/hard_panop_feat_vgg_architecture/hard_panop_feat_vgg_architecture.h5"]
    predict(vr)
