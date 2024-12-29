from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import tensorflow as tf
import cv2
model = tf.keras.models.load_model('kidney_stone_model.h5', compile=False)
IMG_SIZE = 224
def index(request):
    return render(request, 'index.html')
def preprocess_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return np.expand_dims(image, axis=0)
@csrf_exempt
def predict(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'No file provided'}, status=400)
        image = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
        result = 'Stone detected' if prediction > 0.5 else 'Normal'
        return JsonResponse({'prediction': result})
    return JsonResponse({'error': 'Invalid request method'}, status=400)