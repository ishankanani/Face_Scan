from django.shortcuts import render
from image_upload.models import UploadedImage, ImageEmbedding
from deepface import DeepFace
import numpy as np
import base64
import cv2
from PIL import Image
from io import BytesIO

# --- Normalize vector for cosine distance ---
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

# --- Crop the first detected face from an image using OpenCV ---
def crop_face(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            return img[y:y+h, x:x+w]
    except Exception as e:
        print("Face crop error:", e)

    return img  # fallback: use full image if face not found

# --- Extract image (from base64 or file) from POST request ---
def extract_image_from_request(request):
    try:
        if request.method == 'POST':
            # From base64 (e.g., webcam)
            image_data = request.POST.get('image_data')
            if image_data:
                format, imgstr = image_data.split(';base64,')
                img_np = np.frombuffer(base64.b64decode(imgstr), dtype=np.uint8)
                return cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            # From file upload
            elif 'upload_image' in request.FILES:
                file = request.FILES['upload_image']
                img_pil = Image.open(file).convert('RGB')
                img_np = np.array(img_pil)
                return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print("Image extraction error:", e)
    return None

# --- Render scan input form ---
def scan_page(request):
    return render(request, 'face_Scan/scan_page.html')

# --- Compare input face with clustered embeddings ---
def process_scan(request):
    matches = []

    if request.method == 'POST':
        img = extract_image_from_request(request)

        if img is not None:
            try:
                # Step 1: Crop face from input
                face_img = crop_face(img)

                # Step 2: Get normalized input embedding
                embedding_data = DeepFace.represent(
                    img_path=face_img,
                    model_name='Facenet',
                    enforce_detection=False
                )
                input_embedding = normalize_vector(np.array(embedding_data[0]['embedding']))

                # Step 3: Find closest match & its cluster
                closest_cluster = None
                min_dist = float('inf')

                for obj in ImageEmbedding.objects.select_related('image'):
                    db_embedding = normalize_vector(np.array(obj.embedding))
                    dist = np.linalg.norm(input_embedding - db_embedding)
                    if dist < min_dist:
                        min_dist = dist
                        closest_cluster = obj.cluster_id

                # Step 4: Filter by that cluster (if available)
                if closest_cluster is not None:
                    candidates = ImageEmbedding.objects.filter(cluster_id=closest_cluster).select_related('image')
                else:
                    candidates = ImageEmbedding.objects.all().select_related('image')

                # Step 5: Final matching within selected candidates
                for obj in candidates:
                    try:
                        db_embedding = normalize_vector(np.array(obj.embedding))
                        distance = np.linalg.norm(input_embedding - db_embedding)
                        if distance < 0.5:  # Tune this threshold as needed
                            matches.append(obj.image)
                    except Exception as match_err:
                        print("Matching error:", match_err)

            except Exception as e:
                print("Embedding process error:", e)

    return render(request, 'face_Scan/scan_results.html', {'matches': matches})
    return render(request, 'image_upload/upload.html', {'images': page_obj})

