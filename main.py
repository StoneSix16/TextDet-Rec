import importlib 
import argparse
import os

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from skimage import io

import sys
sys.path.append('./HandwrittingRecognition/')
sys.path.append('./CRAFT-pytorch/')

RecModule = importlib.import_module(".Rec","HandwrittingRecognition")
DetModule = importlib.import_module(".Det","CRAFT-pytorch")

parser = argparse.ArgumentParser(description='HWCR arguments')
parser.add_argument('--image_path', dest='image_path', default='./asserts/B8H.png', type=str, help='image for Dec&Rec')
parser.add_argument('--font_path', dest='font_path', default='./fonts/DENG.ttf', type=str, help='font for Dec&Rec')
parser.add_argument('--result_root', dest='result_root', default='./results/', type=str, help='results root')
parser.add_argument('--debug',dest='debug',default=True,type=bool, help='print heatmap when in debug mode')
args = parser.parse_args()
print(args)
DetModel = './CRAFT-pytorch/craft_mlt_25k.pth'
RecModel = './HandwrittingRecognition/efficientnet_20.pth'
RecIndex = './HandwrittingRecognition/label2cha.json'

# init

Det = DetModule.Detection(model_path=DetModel, show_time=args.debug)
Rec = RecModule.Recognition(model_path=RecModel,index_path=RecIndex)

# load image
image = Image.open(args.image_path) # RGB order
image = ImageOps.exif_transpose(image)
image = image.convert('RGB')

dirname = str.split(args.image_path,'/')[-1].split('.')[0]+'/'
if not os.path.exists(os.path.join(args.result_root,dirname)):
    os.makedirs(os.path.join(args.result_root,dirname))

board = Image.new(mode='RGB',size=image.size, color='white')
image_array = np.asarray(image)

boxes = Det.detect(image_array)
bounds = [(box[0][0],box[0][1],box[2][0],box[2][1]) for box in boxes]

text_regions = [image.crop(rec) for rec in bounds]
processed_imgs,predictions = Rec.recognise(text_regions)

draw_on_image = ImageDraw.Draw(image)
draw_on_board = ImageDraw.Draw(board)
font = ImageFont.truetype(font=args.font_path, size=20)

for i in range(len(boxes)):
    width = int(boxes[i][2][0] - boxes[i][0][0])
    draw_on_image.rectangle(bounds[i],fill=None,outline=(0,0,255),width=1)  
    draw_on_board.text(xy=tuple(boxes[i][0]),text=predictions[i][0],fill=(0,0,255), 
              font=(ImageFont.truetype(font=args.font_path, size=width)))
    if args.debug:
        region_dir = os.path.join(args.result_root,dirname,f'{i}/')
        if not os.path.exists(region_dir):
            os.makedirs(region_dir)
        text_regions[i].save(os.path.join(region_dir,"raw.png").replace('\\','/'))
        processed_imgs[i].save(os.path.join(region_dir,"processed.png").replace('\\','/'))
        with open(os.path.join(region_dir,"prediction.txt"),'w',encoding='utf-8') as f:
            f.write(str(predictions[i]))

w,h = image.size
output = Image.new(mode='RGB',size=(w*2,h), color='white')
output.paste(image,(0,0))
output.paste(board,(w,0))

output.save(os.path.join(args.result_root,dirname,"output.png").replace('\\','/'))