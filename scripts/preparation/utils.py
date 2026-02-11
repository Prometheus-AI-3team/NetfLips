import os
import torch
import numpy as np
import torchaudio
import torchvision
import av


def split_file(filename, max_frames=600, fps=25.0):

    lines = open(filename).read().splitlines()

    flag = 0
    stack = []
    res = []

    tmp = 0
    start_timestamp = 0.0

    threshold = max_frames / fps

    for line in lines:
        if "WORD START END ASDSCORE" in line:
            flag = 1
            continue
        if flag:
            word, start, end, score = line.split(" ")
            start, end, score = float(start), float(end), float(score)
            if end < tmp + threshold:
                stack.append(word)
                last_timestamp = end
            else:
                res.append(
                    [
                        " ".join(stack),
                        start_timestamp,
                        last_timestamp,
                        last_timestamp - start_timestamp,
                    ]
                )
                tmp = start
                start_timestamp = start
                stack = [word]
    if stack:
        res.append([" ".join(stack), start_timestamp, end, end - start_timestamp])
    return res


def save_vid_txt(
    dst_vid_filename, dst_txt_filename, trim_video_data, content, video_fps=25
):
    # -- save video
    save2vid(dst_vid_filename, trim_video_data, video_fps)
    # -- save text
    os.makedirs(os.path.dirname(dst_txt_filename), exist_ok=True)
    f = open(dst_txt_filename, "w")
    f.write(f"{content}")
    f.close()


def save_vid_aud(
    dst_vid_filename,
    dst_aud_filename,
    trim_vid_data,
    trim_aud_data,
    video_fps=25,
    audio_sample_rate=16000,
):
    # -- save video
    save2vid(dst_vid_filename, trim_vid_data, video_fps)
    # -- save audio
    save2aud(dst_aud_filename, trim_aud_data, audio_sample_rate)


def save_vid_aud_txt(
    dst_vid_filename,
    dst_aud_filename,
    dst_txt_filename,
    trim_vid_data,
    trim_aud_data,
    content,
    video_fps=25,
    audio_sample_rate=16000,
):
    # -- save video
    if dst_vid_filename is not None:
        save2vid(dst_vid_filename, trim_vid_data, video_fps)
    # -- save audio
    if dst_aud_filename is not None:
        save2aud(dst_aud_filename, trim_aud_data, audio_sample_rate)
    # -- save text
    os.makedirs(os.path.dirname(dst_txt_filename), exist_ok=True)
    f = open(dst_txt_filename, "w")
    f.write(f"{content}")
    f.close()


def save2vid(filename, vid, frames_per_second):

    if not isinstance(frames_per_second, int):
        frames_per_second = int(round(frames_per_second))

    # (1) float → uint8 변환
    if vid.dtype == torch.float32:
        vid = (vid * 255).clamp(0, 255).to(torch.uint8)
            
    if isinstance(vid, np.ndarray):
        print(f"\n*** vid type: {type(vid)}\n")
        vid = torch.from_numpy(vid)
        print(f"\n*** vid type: {type(vid)}\n")

    # # (2) (H, W, C) → (C, H, W) 변환
    # if vid.ndim == 4 and vid.shape[-1] == 3:
    #     print(f"\n*** vid shape: {vid.shape}\n")
    #     vid = vid.permute(0, 3, 1, 2)  # [T,H,W,C] → [T,C,H,W] 
    #     # ValueError: Unexpected numpy array shape `(3, 96, 96)`
    # if vid.shape[1] == 3:  # [T, C, H, W]인 경우
    #     print(f"\n*** vid shape: {vid.shape}\n")
    #     vid = vid.permute(0, 2, 3, 1)
    #     print(f"\n*** vid shape: {vid.shape}\n")

    # torchvision.io.write_video(filename, vid, frames_per_second)
    vid = vid.cpu().numpy()
    vid = (vid * 255).clip(0, 255).astype(np.uint8)

    # PyAV로 직접 저장
    container = av.open(filename, mode="w")
    stream = container.add_stream("libx264", rate=25)
    stream.pix_fmt = "yuv420p"

    for frame in vid:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    # flush
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()



def save2aud(filename, aud, sample_rate):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torchaudio.save(filename, aud, sample_rate)
