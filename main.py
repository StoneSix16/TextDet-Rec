import importlib 
import argparse
Detection = importlib.import_module(".test","CRAFT-pytorch")
Recognition = importlib.import_module(".test","HandwrittingRecognition")

parser = argparse.ArgumentParser(description='HWCR arguments')
# args for CRAFT
Detection.args.trained_model = './CRAFT-pytorch/craft_mlt_25k.pth'
# args for efficient
Recognition.args.index_root = './HandwrittingRecognition/'
Recognition.args.model_path = './HandwrittingRecognition/efficientnet_20.pth'

args = parser.parse_args()

print(Detection.args)
print(Recognition.args)