from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from .fatigue_core import generate_frames, get_current_fatigue_score

def main(request):
    """疲労検知メイン画面"""
    return render(request, 'fatigue_detection/main.html')

def video_feed(request):
    """カメラ映像をストリーミング"""
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_fatigue_score(request):
    """現在の疲労スコアを返す（Ajax用）"""
    return JsonResponse({'fatigue_score': get_current_fatigue_score()})

def home(request):
    """ホーム画面"""
    return render(request, 'fatigue_detection/home.html')

def dashboard(request):
    """ダッシュボード画面"""
    return render(request, 'fatigue_detection/dashboard.html')



