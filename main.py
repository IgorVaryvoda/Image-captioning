import os
import time
import sys
import argparse
import replicate
from dotenv import load_dotenv

load_dotenv()

start = time.time()
#parse folder arg
parser = argparse.ArgumentParser(description='BLIP vs BLIP2 tester.')
parser.add_argument('--i', type=str, default='img/original/', help='target folder path')
parser.add_argument('--model', default="blip2", help='Select model, blip2 or blip')

args = parser.parse_args()
if not args.i.endswith('/'):
    args.i = args.i + '/'
images = os.listdir(args.i)

if (args.model == 'blip2'):
    model = replicate.models.get("salesforce/blip-2")
    version = model.versions.get("4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")
    # https://replicate.com/salesforce/blip-2/versions/4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608#input
    for image in images:
        try:
            inputs = {
                'image': open(f"{args.i}{image}", "rb"),
                'caption': True,
                'question': False,
                'use_nucleus_sampling': False,
                'temperature': 1,
            }
            output = version.predict(**inputs)
            with open('results/blip2.txt', 'a') as the_file:
                the_file.write(f"{image}, {output}\n")
        except Exception as e: print(e)
        except:
            with open('results/blip2.txt', 'a') as the_file:
                the_file.write(f"{image}, error\n")
elif (args.model == 'blip'):
    model = replicate.models.get("salesforce/blip")
    version = model.versions.get("2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746")
    for image in images:
        try:
            inputs = {
                # Input image to query or caption
                'image': open(f"{args.i}{image}", "rb"),
                'task': "image_captioning",
            }
            output = version.predict(**inputs)
            with open('results/blip.txt', 'a') as f:
                f.write(f"{image}, {output}\n")
        except Exception as e: print(e)
        except:
            with open('results/blip.txt', 'a') as f:
                f.write(f"{image}, error \n")
end = time.time()
exec_time= end - start
print(f"execution time: {exec_time}")