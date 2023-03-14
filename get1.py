import replicate
import os
from dotenv import load_dotenv
load_dotenv()
import time
start = time.time()

model = replicate.models.get("salesforce/blip")
version = model.versions.get("2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746")

images = os.listdir('img/500px')

# https://replicate.com/salesforce/blip-2/versions/4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608#input
for image in images:
    try:
        inputs = {
            # Input image to query or caption
            'image': open(f"img/500px/{image}", "rb"),
            'task': "image_captioning",
        }

    # https://replicate.com/salesforce/blip-2/versions/4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608#output-schema
        output = version.predict(**inputs)
        with open('results_blip1.txt', 'a') as the_file:
            the_file.write(f"File:{image}, caption: {output}\n")
    except:
        with open('results_blip1.txt', 'a') as the_file:
            the_file.write(f"File:{image}, error\n")
end = time.time()
print(end - start)