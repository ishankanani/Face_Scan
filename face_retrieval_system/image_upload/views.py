from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.template.loader import render_to_string
from .models import UploadedImage, ImageEmbedding
from deepface import DeepFace
import requests
import cloudinary.uploader
import uuid
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
from django.core.paginator import Paginator
from PIL import Image
from io import BytesIO
from django.db import transaction
from django.views.decorators.csrf import csrf_exempt
from .face_engine import get_facenet_model
import logging

logger = logging.getLogger(__name__)

# --- Compress image before uploading ---
def compress_image(image_file):
    try:
        img = Image.open(image_file)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=60)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print("Image compression error:", e)
        return image_file  # fallback

# --- Detect and extract multiple faces as numpy arrays ---
def extract_faces(img_np):
    faces = []
    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detected = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in detected:
            face_img = img_np[y:y+h, x:x+w]
            faces.append(face_img)
    except Exception as e:
        print("Face detection error:", e)

    if not faces:
        faces.append(img_np)  # fallback: use whole image if no face detected

    return faces

# --- Generate normalized embedding from a face image ---
def extract_normalized_embedding(face_np):
    try:
        model = get_facenet_model()
        result = DeepFace.represent(
            img_path=face_np,
            model_name='Facenet',
            model=model,
            enforce_detection=False
        )
        embedding = np.array(result[0]['embedding'])
        return (embedding / np.linalg.norm(embedding)).tolist()
    except Exception:
        logger.exception("Embedding extraction failed.")
        return None
# --- Upload multiple images, extract multiple embeddings ---
def upload_image(request):
    if request.method == 'POST' and request.FILES.getlist('images'):
        try:
            with transaction.atomic():
                for img_file in request.FILES.getlist('images'):
                    compressed = compress_image(img_file)
                    result = cloudinary.uploader.upload(
                        compressed,
                        folder="photos",
                        public_id=str(uuid.uuid4())
                    )

                    uploaded = UploadedImage.objects.create(
                        image=result['secure_url'],
                        public_id=result['public_id']
                    )
                    uploaded.refresh_from_db()

                    response = requests.get(result['secure_url'])
                    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    img_np = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if img_np is None:
                        raise ValueError("Image decoding failed")

                    faces = extract_faces(img_np)
                    for face_np in faces:
                        embedding = extract_normalized_embedding(face_np)
                        if embedding:
                            ImageEmbedding.objects.create(image=uploaded, embedding=embedding)

        except Exception as e:
            print("Upload or embedding error:", e)

        return redirect('upload-image')

    all_images = UploadedImage.objects.all().order_by('-uploaded_at')
    paginator = Paginator(all_images, 50)
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        html = render_to_string('image_upload/image_list_partial.html', {'images': page_obj})
        return JsonResponse({'html': html})

    return render(request, 'image_upload/upload.html', {'images': page_obj})

# --- Upload single image via AJAX with multiple embeddings ---
@csrf_exempt
def upload_single_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            compressed = compress_image(request.FILES['image'])
            result = cloudinary.uploader.upload(
                compressed,
                folder="photos",
                public_id=str(uuid.uuid4())
            )

            uploaded = UploadedImage.objects.create(
                image=result['secure_url'],
                public_id=result['public_id']
            )

            response = requests.get(result['secure_url'])
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img_np = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img_np is None:
                raise ValueError("Image decoding failed")

            faces = extract_faces(img_np)
            for face_np in faces:
                embedding = extract_normalized_embedding(face_np)
                if embedding:
                    ImageEmbedding.objects.create(image=uploaded, embedding=embedding)

            image_html = render_to_string('image_upload/image_list_item.html', {'image': uploaded})
            return JsonResponse({'success': True, 'image_html': image_html})

        except Exception as e:
            print("AJAX Upload error:", e)
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid request'})

# --- Delete one image and all its embeddings ---
def delete_image(request, image_id):
    if request.method == 'POST':
        try:
            img_obj = UploadedImage.objects.get(id=image_id)
            cloudinary.uploader.destroy(img_obj.public_id)
            ImageEmbedding.objects.filter(image_id=image_id).delete()
            img_obj.delete()
        except Exception as e:
            print("Delete error:", e)
    return redirect('upload-image')

# --- Delete all images and embeddings ---
def delete_all_images(request):
    if request.method == 'POST':
        images = UploadedImage.objects.all()
        public_ids = [img.public_id for img in images]

        for pid in public_ids:
            try:
                cloudinary.uploader.destroy(pid)
            except Exception as e:
                print("Cloudinary delete error:", e)

        with transaction.atomic():
            ImageEmbedding.objects.all().delete()
            UploadedImage.objects.all().delete()

    return redirect('upload-image')

# --- Search matching images from uploaded face ---
@csrf_exempt
def process_scan(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            compressed = compress_image(request.FILES['image'])
            img = Image.open(compressed)
            img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            query_faces = extract_faces(img_np)

            query_embeddings = []
            for face_np in query_faces:
                embedding = extract_normalized_embedding(face_np)
                if embedding:
                    query_embeddings.append(np.array(embedding))

            if not query_embeddings:
                return JsonResponse({'success': False, 'error': 'No face detected in query image'})

            db_embeddings = ImageEmbedding.objects.select_related('image').all()

            threshold = 0.2
            matched_images = set()

            for query_emb in query_embeddings:
                for emb_obj in db_embeddings:
                    db_emb = np.array(emb_obj.embedding)
                    dist = np.linalg.norm(query_emb - db_emb)
                    if dist < threshold:
                        matched_images.add(emb_obj.image)

            matched_images = list(matched_images)
            paginator = Paginator(matched_images, 50)
            page_number = request.GET.get('page', 1)
            page_obj = paginator.get_page(page_number)

            html = render_to_string('image_upload/image_list_partial.html', {'images': page_obj})
            return JsonResponse({'success': True, 'html': html})

        except Exception as e:
            print("Scan error:", e)
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid request'})

def cluster_embeddings(eps=0.4, min_samples=2):
    embeddings = list(ImageEmbedding.objects.all())

    if not embeddings:
        return 0

    vectors = [np.array(e.embedding) for e in embeddings]
    X = np.vstack(vectors)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(X)

    # Save cluster labels
    for i, emb in enumerate(embeddings):
        emb.cluster_id = int(clustering.labels_[i]) if clustering.labels_[i] != -1 else None
        emb.save()

    return len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)


# views.py
from django.shortcuts import render
from image_upload.models import ImageEmbedding

def view_clusters(request):
    clusters = {}
    for embedding in ImageEmbedding.objects.select_related('image').all():
        cid = embedding.cluster_id
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(embedding.image)

    return render(request, 'image_upload/clusters.html', {'clusters': clusters})
