import replicate
import os
from dotenv import load_dotenv
load_dotenv()
import time

start = time.time()
model = replicate.models.get("salesforce/blip-2")
version = model.versions.get("4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")
images = os.listdir('img/300px')

# https://replicate.com/salesforce/blip-2/versions/4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608#input
for image in images:
    try:
        inputs = {
            # Input image to query or caption
            'image': open(f"img/300px/{image}", "rb"),

            # Select if you want to generate image captions instead of asking
            # questions
            'caption': True,

            # Question to ask about this image. Leave blank for captioning
            'question': False,

            # Optional - previous questions and answers to be used as context for
            # answering current question
            # 'context': ...,

            # Toggles the model using nucleus sampling to generate responses
            'use_nucleus_sampling': False,

            # Temperature for use with nucleus sampling
            # Range: 0.5 to 1
            'temperature': 1,
        }

        # https://replicate.com/salesforce/blip-2/versions/4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608#output-schema
        output = version.predict(**inputs)
        with open('results.txt', 'a') as the_file:
            the_file.write(f"{image}, {output}\n")
    except:
        with open('results.txt', 'a') as the_file:
            the_file.write(f"{image}, error\n")
end = time.time()
print(end - start)