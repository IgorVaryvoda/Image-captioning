from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

# Your subscription key and endpoint
key = "1ba69e7a4448420f8e074c448e96acd3"
endpoint = "https://auto-alt-tags.cognitiveservices.azure.com/"
region = "westeurope"
#Create Azure client
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))


images = os.listdir('img')

# https://replicate.com/salesforce/blip-2/versions/4b32258c42e9efd4288bb9910bc532a69727f9acd26aa08e175713a0a857a608#input
for image in images:
      # Create a Computer Vision client
     description_results = computervision_client.describe_image(open(f"img/{image}", "rb"))
      # Get the captions (descriptions) from the response, with confidence level
     if (len(description_results.captions) == 0):
        with open('results_azure.txt', 'a') as the_file:
            the_file.write(f"File:{image}, No caption generated")
     else:
        for caption in description_results.captions:
          print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))
          description = '{"description": "'+ caption.text + '"}'
          with open('results_azure.txt', 'a') as the_file:
                  the_file.write(f"File:{image}, {description}")