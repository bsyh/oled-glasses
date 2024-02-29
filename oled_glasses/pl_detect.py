#!/home/hushouyue/miniconda3/envs/glasses/bin/python

import requests
from PIL import Image
from io import BytesIO

from transformers import ViTFeatureExtractor, ViTForImageClassification

# Get example image from official fairface repo + read it in as an image
# r = requests.get('https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg?raw=true')
# r = requests.get('https://storage.googleapis.com/kagglesdsdata/datasets/2910971/6815559/Tom_Cruise_by_Gage_Skidmore_2.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240229%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240229T031128Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=84e025d6c6492577a069b2224037cbf263a22dbfd445f97c400dabff2dd4484a23695657cb12051ceb8c99685c5acde453f414c2d50f879c9060cfff1102a5d90f7696bc61a28ecdfca1047b85a3717a481ce9b756f04928c0504ef2ea4d623ce875521db40331435c498cc7ce99acc52cf9abc501dc10ced2be734e9c77fe62870a7e89e0e96d9db84e03e3f6534526f037cdae7ce4c26568bd46fef9e44269ab0f84cd863cc158d7e74eb3e7aa808546e57aa0f8a8e4b685660fef9ab604800418f18024df816c4e6d2950297bd4a36f16d9a0d82d61bde8ca36ff1078ffff79292d6e352741e4cf27eb6702e920e559af5bf729e0f1fe52df99a4176363e3')
r = requests.get("https://encrypted-tbn2.gstatic.com/licensed-image?q=tbn:ANd9GcSfcB96rkysCCHgQJd2l_RzFnat8AkW8MYEum8DTLCU5n9p-eSvRsRlrpk1K_6JgdofrpTZ__fJa_4Vkyo")


im = Image.open(BytesIO(r.content))

# Init model, transforms
model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

# Transform our image and pass it through the model
inputs = transforms(im, return_tensors='pt')
output = model(**inputs)

# Predicted Class probabilities
proba = output.logits.softmax(1)

# Predicted Classes
preds = proba.argmax(1)

for i in range(proba.size()[1]):
    print(model.config.id2label[i],':',proba[0,i].item())
print("most likely",model.config.id2label[preds.item()])
