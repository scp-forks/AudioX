import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import read_video, merge_video_audio
from stable_audio_tools.data.utils import load_and_process_audio
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("HKUSTAudio/AudioX")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
target_fps = model_config["video_fps"]
seconds_start = 0
seconds_total = 10

model = model.to(device)

# for video-to-music generation
# video_path = "video.mp4"
# text_prompt = "Steve Urkel falling down a flight of stairs"
audio_path = None
video_path = None
#
# text_prompts = [
#     "Using a key to wind up a music box",
#     "Steve Urkel falling down a flight of stairs",
#     "A screaming bee",
#     "Harry Potter reciting a spell while suffering from a total glossectomy",
#     "James Earl Jones yelling 'NO!' after inhaling sulfur hexafluoride",
#     "A cartoonish sound of a thief tiptoeing sneakily",
#     "The animated cartoon character Bugs Bunny from Warner Bros talking",
#     "Arnold Schwarzenegger saying, 'I'll be back.'",
#     "The sounds of a dial-up modem",
#     "A starfighter in the vacuum of space",
#     "A man stepping on a lego, extreme pain",
#     "The Koolaid Man",
#     "Charlie Parker blowing up a helium weather balloon",
#     "The theme from Jurassic Park played on a hurdy gurdy",
#     "A fast paced game of blitz Chess",
#     "Hitler having a seizure"
# ]

text_prompts = [
    "A cat purring",
    "A morbidly obese Type 1 diabetic cat purring",
    "A music with Happy Birthday under water",
    "The curator of a museum gulping loudly",
    "A music from Jurassic Park played on a flute"
]

# loop through text prompts
for text_prompt in text_prompts:

    print(f"Generating audio for prompt: {text_prompt}")

    video_tensor = read_video(video_path, seek_time=0, duration=seconds_total, target_fps=target_fps)
    audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)

    # conditioning = [{
    #     # "video_prompt": [video_tensor.unsqueeze(0)],
    #     "video_path": None,
    #     "video_prompt": None,
    #     "text_prompt": text_prompt,
    #     # "audio_prompt": audio_tensor.unsqueeze(0),
    #     "audio_prompt": None,
    #     "audio_path": None,
    #     "seconds_start": None,
    #     "seconds_total": None,
    # }]

    conditioning = [{
        "video_prompt": [video_tensor.unsqueeze(0)],
        "text_prompt": text_prompt,
        "audio_prompt": audio_tensor.unsqueeze(0),
        "seconds_start": seconds_start,
        "seconds_total": seconds_total
    }]


    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        steps=250,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to file
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    output_file_name = f"output_{text_prompt.replace(' ', '_')}.wav"

    torchaudio.save(output_file_name, output, sample_rate)

# if video_path is not None and os.path.exists(video_path):
#     merge_video_audio(video_path, "output.wav", "output.mp4", 0, seconds_total)
