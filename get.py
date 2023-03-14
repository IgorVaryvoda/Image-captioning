import replicate
import os
from dotenv import load_dotenv
load_dotenv()
import time

start = time.time()
model = replicate.models.get("salesforce/blip-2")
version = model.versions.get("4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608")
images = os.listdir('img')

# https://replicate.com/salesforce/blip-2/versions/4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608#input
for image in images:
    inputs = {
        # Input image to query or caption
         'image': open(f"img/{image}", "rb"),

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
        the_file.write(f"File:{image}, caption: {output}\n")

end = time.time()
print(end - start)