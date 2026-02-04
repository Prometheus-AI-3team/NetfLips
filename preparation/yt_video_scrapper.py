import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

def download_video(url, save_path, file_name):
    # 저장 경로가 없으면 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 전체 파일 경로
    full_path = os.path.join(save_path, f"{file_name}.%(ext)s")

    # 비디오와 오디오 파일을 따로 저장
    command = [
        "yt-dlp", url,
        "-o", full_path,  # 확장자에 따라 저장
        #"-f", "bestvideo[height<=1080]+bestaudio/best",  # 비디오와 오디오를 각각 다운로드
        "-f", "bestvideo[ext=webm]+bestaudio[ext=webm]/best",  # 비디오와 오디오를 webm 형식으로 다운로드
        #"--keep-video",  # 병합하지 않고 각각 따로 저장
        "--recode-video", "mp4",
        # "--sub-lang", "ko",  # 자막 언어 설정
        # "--convert-subs", "srt",  # 자막을 텍스트로 변환
    ]

    # yt-dlp 실행
    subprocess.run(command)

def download_videos_multithread(link_list, save_path, max_workers=4):
    # 멀티스레드 풀 생성
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 각 링크를 병렬로 처리
        futures = []
        for i, url in enumerate(link_list):
            file_name = f"intersection_0{i+1}"  # 각 파일에 순차적으로 이름 지정
            futures.append(executor.submit(download_video, url, save_path, file_name))

        # 모든 다운로드가 완료될 때까지 기다림
        for future in futures:
            future.result()  # 결과를 기다림 (예외 발생 시 처리)

# 다운받을 유튜브 공유 링크 목록
link_list = [
    
    # "https://youtu.be/IH_uvSxdoQU?si=4Mz3i047TWnYsQo9", # Daphne's Confession | Bridgerton
    "https://youtu.be/Crr7j0udrc4?si=DgJbLLzjjQl11aMV",   # Elon Musk delivers speech after Trump Inauguration

]


# 저장 경로 지정
save_path = "/Users/jisu/Desktop/dev/cli/av2av/jisu/video_data"

print(len(link_list))
# 함수 호출
download_videos_multithread(link_list, save_path, max_workers=4)