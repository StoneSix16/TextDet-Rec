import importlib 
import argparse
import os

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ExifTags
from skimage import io


import sys
sys.path.append('./HandwrittingRecognition/')
sys.path.append('./CRAFT-pytorch/')

RecModule = importlib.import_module(".Rec","HandwrittingRecognition")
DetModule = importlib.import_module(".Det","CRAFT-pytorch")

parser = argparse.ArgumentParser(description='HWCR arguments')
parser.add_argument('--image_path', dest='image_path', default='./asserts/jiangnan.jpg', type=str, help='image for Dec&Rec')
parser.add_argument('--font_path', dest='font_path', default='./fonts/DENG.ttf', type=str, help='font for Dec&Rec')
parser.add_argument('--result_root', dest='result_root', default='./results/', type=str, help='results root')
parser.add_argument('--show_time', dest='show_time', default=True, type=bool, help='show time or not')
parser.add_argument('--debug',dest='debug',default=False,type=bool, help='print heatmap when in debug mode')
args = parser.parse_args()
print(args)
DetModel = './CRAFT-pytorch/craft_mlt_25k.pth'
RecModel = './HandwrittingRecognition/efficientnet_20.pth'
RecIndex = './HandwrittingRecognition/label2cha.json'

# init

Det = DetModule.Detection(model_path=DetModel, show_time=args.show_time)
Rec = RecModule.Recognition(model_path=RecModel,index_path=RecIndex)

# load image
image = Image.open(args.image_path) # RGB order
try:
    for orientation in ExifTags.TAGS.keys() : 
        if ExifTags.TAGS[orientation]=='Orientation' : break 
    exif=dict(image._getexif().items())
    if   exif[orientation] == 3 : 
        image=image.rotate(180, expand = True)
    elif exif[orientation] == 6 : 
        image=image.rotate(270, expand = True)
    elif exif[orientation] == 8 : 
        image=image.rotate(90, expand = True)
except:
    pass
image = image.convert('RGB')

board = Image.new(mode='RGB',size=image.size, color='white')
board.paste(image,(0,0))
image_array = np.asarray(image)

boxes = Det.detect(image_array)
bounds = [(box[0][0],box[0][1],box[2][0],box[2][1]) for box in boxes]

text_regions = [image.crop(rec) for rec in bounds]
predictions = Rec.recognise(text_regions)

draw = ImageDraw.Draw(board)
font = ImageFont.truetype(font=args.font_path, size=20)
for i in range(len(boxes)):
    width = int(boxes[i][2][0] - boxes[i][0][0])
    draw.rectangle(bounds[i],fill=None,outline=(0,0,255),width=1)  
    draw.text(xy=tuple(boxes[i][0]),text=predictions[i],fill=(0,0,255), 
              font=(ImageFont.truetype(font=args.font_path, size=width)))

filename = str.split(args.image_path,'/')[-1]
board.save(os.path.join(args.result_root,filename).replace('\\','/'))