from django.shortcuts import render
from django.http import HttpResponse
from django.http.response import StreamingHttpResponse
import numpy as np
import cv2
import mediapipe as mp
from django.views.decorators import gzip
from signtext.camera import *
from signtext import config
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from django.http import HttpResponseNotAllowed
# from tf.keras.models import load_model
mp_holistic = mp.solutions.holistic
from django.http import HttpResponseBadRequest, JsonResponse
from django.contrib.sessions.backends.db import SessionStore
from PIL import Image
from django.conf import settings
import requests
from . import text_2_sign, search_video, audio, sign_2_text
from django.core.paginator import Paginator
from pathlib import Path   
from django.views.decorators.csrf import csrf_exempt
import tempfile
import time

s = SessionStore()

module_dir = os.path.dirname(__file__)   #get current directory
model_path = os.path.join(module_dir, 'static/signmodel.h5')
labels_path = os.path.join(module_dir, 'static/labels.npy')
sequence_path = os.path.join(module_dir, 'static/max_len.npy')

data_dir = os.path.join(module_dir, 'static/data/')
gifs = os.path.join(module_dir, 'static/data/gifs/')

model = load_model(model_path)
labels = np.load(labels_path)
sequence_length = np.load(sequence_path)

sequence_length = int(sequence_length)

s['sequence'] = config.sentence
s['predictions'] = config.predictions
s['new_pred'] = config.new_pred
s['sentence'] = config.sentence
s['threshold'] = config.threshold
s['second_image'] = config.second_image
s['feedback'] = config.feedback

s['mp_holistic'] = mp.solutions.holistic
s['holistic'] = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Create your views here.

def is_ajax(request):
    return request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'

def update_session(request):
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if s['new_pred']:
        if is_ajax:
            if request.method == 'GET':

                # response = requests.post("https://hnmensah-ghanaian-language-translator.hf.space/api/predict", json={
                #     "data": [
                #         "English",
                #         "Asante",
                #         s['new_pred'],
                # ]}).json()

                # data = str(s['new_pred']) + " - " + str(response["data"][0])
                data = str(s['new_pred'])
                
                # new_pred = data
                print(data)
                return JsonResponse({'context': data, })
            return JsonResponse({'status': 'Invalid request'}, status=400)
        else:
            return HttpResponseBadRequest('Invalid request')


def result_display_landmarks(image):

    
    with  mp_holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])



def get_landmarks_predictions(request, frame):
    results = s['holistic'] .process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    landmarks =  np.concatenate([pose, face, lh, rh])
    
    
    if results.left_hand_landmarks or results.right_hand_landmarks: #check if hand shows in frame
        # print(True)
        s['feedback'] = "Sign detected"
        s['sequence'].append(landmarks) #if True, add to sequence
    
    else: # If no hands shown, check if it's end of sign or no sign at all
        if not len(s['sequence']):  # if no sign from hands
            s['feedback'] = "Please place your hand in the camera frame to begin detection"            # continue to next frame
            
         
        else:                   # if there exists other signs, means end of sign, make predictions
            s['feedback'] = "Predicting sign..."

            needed__width = sequence_length - len(s['sequence'])
            s['sequence'] = np.pad(s['sequence'], [(0, needed__width), (0,0) ], mode='constant')
            res = model.predict(np.expand_dims(s['sequence'], axis=0))
            label = labels[np.argmax(res)]
            s['predictions'].append(label)
            s['sequence'] = [] # empty sequence for next round of signs

            if len(s['new_pred']) >= 10: # prevent words overflow from screen
                new_pred = s['predictions'][::-1][:10][::-1]   # if words overflow, pick the last 5
            else:
                new_pred = s['predictions']          #else show words
            s['new_pred'] = new_pred[-1]
            # s['new_pred'] = s['predictions'][::-1][1]
            # print(s.get('new_pred', "Heloo"))
            # "Detecting sign"

                    

    
def gen(request, camera):
    while True:
        frame = camera.get_frame()
        image = camera.get_clean_frame()

        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        get_landmarks_predictions(request, image) #-- change this 
        request.session.modified = True


@gzip.gzip_page
def livefeed(request):
    try:
        cam = VideoCamera()
        request.session.modified = True
        return StreamingHttpResponse(gen(request, cam), content_type="multipart/x-mixed-replace;boundary=frame")
    
    except:  # This is bad!
        pass


# Global control variables
is_recording = False
recording_thread = None

def start_recording(request):
    global is_recording, recording_thread
    if not is_recording:
        is_recording = True
        recording_thread = threading.Thread(target=record_video)
        recording_thread.start()
        return JsonResponse({"status": "recording started"})
    else:
        return JsonResponse({"status": "already recording"})

def stop_recording(request):
    global is_recording, recording_thread
    if is_recording:
        is_recording = False
        recording_thread.join()
        # After recording, predict translation
        translation = sign_2_text.predict_translation_from_video("realtime_test_video.mp4")
        return JsonResponse({"status": "recording stopped", "translation": translation})
    else:
        return JsonResponse({"status": "not recording"})

def record_video():
    global is_recording
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    out = cv2.VideoWriter("realtime_test_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 480))

    start_time = time.time()
    while is_recording and (time.time() - start_time) < 11:  # Max 11s
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    out.release()
    cap.release()


def signtext(request):
    if request.method == "POST":
        video_file = request.FILES.get("video")
        if video_file:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
                for chunk in video_file.chunks():
                    tmpfile.write(chunk)
                tmpfile_path = tmpfile.name

            translation_text = sign_2_text.predict_translation_from_video(tmpfile_path)

            os.remove(tmpfile_path)

            # Return JSON response with translation
            return JsonResponse({"translation": translation_text})

        # If no video uploaded, return error JSON
        return JsonResponse({"error": "No video uploaded"}, status=400)

    # For GET request render the page normally
    return render(request, 'signtext/signtext.html')


def practice(request):
    # request.session["new_pred"] = ' '.join(config.new_pred)
    request.session.modified = True

    context = {
        "sentence" : ' '.join(list(s['new_pred'])),
        "feedback": s['feedback'],  
        's': ' '.join(s['new_pred'])
    }
    return render(request, 'signtext/practice.html', context=context)

def display_landmarks(image):
    mp_drawing = mp.solutions.drawing_utils 
    mp_drawing_styles = mp.solutions.drawing_styles

    with  mp_holistic.Holistic(static_image_mode=True) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            final_results = results.pose_world_landmarks.landmark
            
            mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
            

@csrf_exempt
def audioToSign(request):
    transcribed_text = ""
    video_ids = []
    video_urls = []
    gif_url = []
    mode = None

    if request.method == "POST":
        audio_file = request.FILES.get("audio")
        if audio_file:
            # Save uploaded audio to a temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                for chunk in audio_file.chunks():
                    tmpfile.write(chunk)
                tmpfile_path = tmpfile.name

            # Process audio through your transcription + video retrieval pipeline
            transcribed_text, video_ids = audio.process_audio_file(tmpfile_path)

            # Clean up temp file
            os.remove(tmpfile_path)

            # Now pass transcribed_text into the video retrieval pipeline
            media_root_path = Path(settings.MEDIA_ROOT)
            sentence_dataset_path = media_root_path / "merged"
            word_dataset_path = media_root_path / "words_output1"
            gif_paths = media_root_path / "words_output"

            print(f"directory for sentence videos: {sentence_dataset_path}")
            
            if transcribed_text.strip() != "":
                response = text_2_sign.retrieve_video(transcribed_text)
                print(f"Response: {response}")

                if response is not None:
                    mode = response.get('mode')
                    videos = response.get('videos', [])

                    if mode == "sentence":
                        label = videos[0]
                        print(f'label sentences hitting: {label}')
                        video_abs_path = search_video.search_video_by_label(sentence_dataset_path, label)
                        gif_abs_paths = search_video.search_word_gifs_by_labels(gif_paths, [label])

                        if video_abs_path:
                            video_file_name = os.path.basename(video_abs_path)
                            video_urls.append(settings.MEDIA_URL + "merged/" + video_file_name)
                        else:
                            video_urls.append(None)

                        for abs_path in gif_abs_paths:
                            if abs_path:
                                gif_file_name = os.path.basename(abs_path)
                                gif_url.append(settings.MEDIA_URL + "pose_landmarks/" + gif_file_name)
                            else:
                                gif_url.append(None)

                    elif mode == "word-by-word":
                        labels = videos
                        output_video = r"C:\Users\Idan\Desktop\All\SIGN TALK\Full dataset\merged_sentence.mp4"
                        video_abs_paths = search_video.search_word_videos_by_labels(word_dataset_path, labels)
                        merged_videos_path = search_video.concatenate_videos(video_abs_paths, output_video)
                        gif_abs_paths = search_video.search_word_gifs_by_labels(gif_paths, labels)

                        video_file_name = os.path.basename(merged_videos_path)
                        video_urls.append(settings.MEDIA_URL + video_file_name)

                        for abs_path in gif_abs_paths:
                            if abs_path:
                                gif_file_name = os.path.basename(abs_path)
                                gif_url.append(settings.MEDIA_URL + "words_output/" + gif_file_name)
                            else:
                                gif_url.append(None)
                else:
                    video_urls = []
                    gif_url = []

    # Final context passed to template
    context = {
        "mode": mode,
        "transcribed_text": transcribed_text,
        "video_ids": video_ids,
        "video_paths": video_urls,
        "gif_paths": gif_url
    }

    return render(request, "signtext/textsign1.html", context)


def textsign(request):
    word = request.GET.get('q') if request.GET.get('q') is not None else ""

    context = {}
    video_urls = []
    gif_url = []
    media_root_path = Path(settings.MEDIA_ROOT)
    sentence_dataset_path = media_root_path / "merged"
    word_dataset_path = media_root_path / "words_output1"
    gif_paths = media_root_path / "words_output"

    print(f"directory for sentence videos: {sentence_dataset_path}")
    mode = None  # 🔒 fix scope issue

    if word != "":
        response = text_2_sign.retrieve_video(word)
        print(f"Response: {response}")

        if response is not None:
            mode = response.get('mode')
            videos = response.get('videos', [])

            if mode == "sentence":
                label = videos[0]
                print(f'label sentences hitting: {label}')
                video_abs_path = search_video.search_video_by_label(sentence_dataset_path, label)
                gif_abs_paths = search_video.search_word_gifs_by_labels(gif_paths, [label])
                print(f"Sentence Video Path: {video_abs_path}")
                print(f"gif file path: {gif_abs_paths}")

                if video_abs_path:
                    video_file_name = os.path.basename(video_abs_path)
                    print(f"video file name: {video_file_name}")
                    video_urls.append(settings.MEDIA_URL + "merged/" + video_file_name)
                else:
                    video_urls.append(None)
                    
                from urllib.parse import quote


                for abs_path in gif_abs_paths:
                    if abs_path:
                        gif_file_name = os.path.basename(abs_path)
                        gif_url.append(settings.MEDIA_URL + "pose_landmarks/" + gif_file_name)
                        print(f"gif url: {gif_url}")
                    else:
                        gif_url.append(None)
            


            elif mode == "word-by-word":
                labels = videos
                # merged videos output path
                output_video = r"C:\Users\Idan\Desktop\All\SIGN TALK\Full dataset\merged_sentence.mp4"
                video_abs_paths = search_video.search_word_videos_by_labels(word_dataset_path, labels)
                merged_videos_path = search_video.concatenate_videos(video_abs_paths, output_video)
                
                gif_abs_paths = search_video.search_word_gifs_by_labels(gif_paths, labels)
                print(f"Word-by-word Video Paths: {merged_videos_path}")
                print(f"Word-by-word gif Paths: {gif_abs_paths}")

                # for abs_path in merged_videos_path:
                #     if abs_path:
                #         video_file_name = os.path.basename(abs_path)
                #         video_urls.append(settings.MEDIA_URL + "words_output1/" + video_file_name)
                #     else:
                #         video_urls.append(None)
                
                video_file_name = os.path.basename(merged_videos_path)
                video_urls.append(settings.MEDIA_URL + video_file_name)
                

                for abs_path in gif_abs_paths:
                    if abs_path:
                        gif_file_name = os.path.basename(abs_path)
                        gif_url.append(settings.MEDIA_URL + "words_output/" + gif_file_name)
                    else:
                        gif_url.append(None)
        else:
            video_urls = []
            gif_url = []

    context = {
        "mode": mode,
        "word": word,
        "video_paths": video_urls,
        "gif_paths": gif_url
    }

    return render(request, 'signtext/textsign1.html', context=context)


def index(request):
    return render(request, 'signtext/index.html')

def select_user(request):
    return render(request, 'signtext/select_user.html')

def select_solution(request):
    return render(request, 'signtext/select_solution.html')

def learn_main(request):
    categories = ["Pregnancy & Reproduction", "Emergency", "Medical conditions", "Remedies", "Medical procedures", "General : Health"]

    context = {
        "categories": categories
    }
    return render(request, 'signtext/learn_main.html', context)

def learn_page(request, pk):

    category_pair = {
        "Pregnancy & Reproduction": [["Breastfeed","Breastfeed.mp4"], ["Health","Health.mp4"], ["Labor","Labor.mp4"], ["Miscarriage","Miscarriage.mp4"], ["Breast","Breast.mp4"], ["Medicine","Medicine.mp4"], ],
        "Emergency": [["Emergency","https://youtube.com"], ["Labor","https://youtube.com"]],
        "Medical conditions": [["Conditions","https://youtube.com"], ["Labor","https://youtube.com"]],
        "Remedies": [["Remedies","https://youtube.com"], ["Labor","https://youtube.com"]],
        "Medical procedures": [["Medical","https://youtube.com"], ["Labor","https://youtube.com"]],
        "General : Health": [["Health","https://youtube.com"], ["Labor","https://youtube.com"]]
    }

    signs = category_pair[pk]

    p = Paginator(signs, 1)

    page = request.GET.get('page')
    if not page:
        page = 1

    object_list = p.page(page)

    print(len(signs))


    context = {
        "signs" : object_list,
        "percent": str( int(page) / int(p.count) * 100) + "%"
    }
    return render(request, 'signtext/learn_page.html', context)



def hospital_page(request):
    hospitals = [
        ["Komfo Anokye Teaching Hospital", "Ashanti Region", "https://www.google.com/maps/dir/6.9723244,-1.4716475/komfo+anokye+hospital/@6.8303378,-1.6584845,11z/data=!3m1!4b1!4m9!4m8!1m1!4e1!1m5!1m1!1s0xfdb96fa87e87323:0xf47a8c98b1dd0923!2m2!1d-1.6291939!2d6.6975237"],
        ["Asante Mampong Government Hospital", "Ashanti Region", ""],
        ["Korle Bu Teaching Hospital", "Greater Accra Region", ""],
        ["37 Military Hospital", "Greater Accra Region", ""],
        ["Effia Nkwanta Regional Hospital", "Western Region", ""],
        ["Essikado Hospital", "Western Region", ""],
        ["Ho Municipal Hospital", "Volta Region", ""],
        ["Bibiani Government Hospital", "Western Region", ""],

    ]

    context = {
        "hospitals" : hospitals
    }

    return render(request, 'signtext/choose_hospital.html', context)


def learn(request):
    return render(request, 'signtext/learn.html')

def community(request):
    return render(request, 'signtext/community.html')

def community_chat(request):
    return render(request, 'signtext/community_chat.html')




