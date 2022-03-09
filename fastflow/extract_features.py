
from feature_extractor import FeatureExtractor

def create_feature_dataset(model, layers, dataset, out_name):
    encoder = FeatureExtractor(model, layers)
    encoder.eval()
    