# End-to-End Speech Recognition Models  
  
[<img src="https://github.com/gentaiscool/end2end-asr-pytorch/raw/master/img/pytorch-logo-dark.png" height=18>](https://pytorch.org/) 
<img src="https://img.shields.io/badge/License-Apache--2.0-yellow" height=20>
[![CodeFactor](https://www.codefactor.io/repository/github/sooftware/end-to-end-speech-recognition-models/badge)](https://www.codefactor.io/repository/github/sooftware/end-to-end-speech-recognition-models)
  
This repository contains end-to-end automatic speech recognition models.This repository does not include training or audio or text preprocessing codes. If you want to see the code other than the model, please refer to [here](https://github.com/sooftware/KoSpeech).   
Many speech recognition open sources contain all the training-related code, making it hard to see only the model structure. So I have created a repository for only the models I've implemented and make them public.   
I will continue to add to this the speech recognition models that I implement.  
  
## Implementation List  
  
- Deep Speech 2  
*Dario Amodei et al. [Deep Speech2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)*   
*SeanNaren. [deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)*
  
- Listen, Attend and Spell (modified version)   
*Wiliam Chan et al. [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211)*   
*Takaaki Hori et al. [Advances in Joint CTC-Attention based E2E ASR with a Deep CNN Encoder and RNN-LM](https://arxiv.org/abs/1706.02737)*   
*IBM. [Pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)*  
*clovaai. [ClovaCall](https://github.com/clovaai/ClovaCall)*
  
- Speech Transformer  
*Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)*     
*Yuanyuan Zhao et al. [The SpeechTransformer for Large-scale Mandarin Chinese Speech Recognition](https://ieeexplore.ieee.org/document/8682586)*  
*kaituoxu. [Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)*
  
- Voice Activity Detection (1 dimensional Resnet Model)   
*filippogiruzzi. [voice_activity_detection](https://github.com/filippogiruzzi/voice_activity_detection)*
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/End-to-end-Speech-Recognition/issues) on Github.   
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
### Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
   
### License
This project is licensed under the Apache-2.0 LICENSE - see the [LICENSE.md](https://github.com/sooftware/End-to-End-Speech-Recognition-Models/blob/main/LICENSE) file for details
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: kaki.brain@kakaobrain.com
