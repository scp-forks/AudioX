# AudioX: Diffusion Transformer for Anything-to-Audio Generation


[![githubio](https://img.shields.io/badge/GitHub.io-Project-blue?logo=Github&style=flat-square)](https://zeyuet.github.io/AudioX/)

**This is the repository for "AudioX: Diffusion Transformer for Anything-to-Audio Generation".**

## 📺 Demo Video

<!-- [![Watch the video](https://github.com/user-attachments/assets/498f7d3f-cb7c-4f32-92f0-4b71fa0712e9)] -->

<video width="100%" controls>
  <source src="./static/videos/AudioX_DEMO.mp4" type="video/mp4">
</video>


## ✨ Abstract

Audio and music generation have emerged as crucial tasks in many applications, yet existing approaches face significant limitations: they operate in isolation without unified capabilities across modalities, suffer from scarce high-quality, multi-modal training data, and struggle to effectively integrate diverse inputs. In this work, we propose AudioX, a unified Diffusion Transformer model for Anything-to-Audio and Music Generation. Unlike previous domain-specific models, AudioX can generate both general audio and music with high quality, while offering flexible natural language control and seamless processing of various modalities including text, video, image, music, and audio. Its key innovation is a multi-modal masked training strategy that masks inputs across modalities and forces the model to learn from masked inputs, yielding robust and unified cross-modal representations. To address data scarcity, we curate two comprehensive datasets: vggsound-caps with 190K audio captions based on the VGGSound dataset, and V2M-caps with 6 million music captions derived from the V2M dataset. Extensive experiments demonstrate that AudioX not only matches or outperforms state-of-the-art specialized models, but also offers remarkable versatility in handling diverse input modalities and generation tasks within a unified architecture.


## ✨ Teaser

<p align="center">
  <img src="https://github.com/user-attachments/assets/ea723225-f9c8-4ca2-8837-2c2c08189bdd" alt="method">
</p>
<p style="text-align: left;">(a) Overview of AudioX, illustrating its capabilities across various tasks. (b) Radar chart comparing the performance of different methods across multiple benchmarks. AudioX demonstrates superior Inception Scores (IS) across a diverse set of datasets in audio and music generation tasks.</p>


## ✨ Method

<p align="center">
  <img src="https://github.com/user-attachments/assets/94ea3df0-8c66-4259-b681-791ee41bada8" alt="method">
</p>
<p align="center">Overview of the AudioX Framework.</p>



## Code
To be released.


<hr>

