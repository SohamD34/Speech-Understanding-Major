# Speech-Understanding-Major
Code repository for submission to the Major Exam of CSL7770: Speech Understanding Course

In order to run the code files provided, clone this repository into your local machine.
```
> cd <location>
> git clone https://www.github.com/SohamD34/Speech-Understanding-Major.git
> cd Speech-Understanding-Major/
```
Create a virtual Python environment.
```
> python3 -m venv b21ee067
```
Activate the environment and install the necessary Python libraries and packages.
```
> source b21ee067/bin/activate
> pip install -r requirements.txt
```

## Question 1

Run the ```scripts/B21EE067_Question1.ipynb``` notebook using Jupyter.
```
> jupyter notebook scripts/B21EE067_Question1.ipynb
```
Run all the cells sequentially. The instructions to find the source video file are available at ```data/video_lecture/```. 
You can observe the transcribed text at ```data/transcripts/``` and the regenerated audio file in Marathi at ```data/marathi_tts_out.wav```.

## Question 2

Run the ```scripts/B21EE067_Question2.ipynb``` notebook using Jupyter.
```
> jupyter notebook scripts/B21EE067_Question2.ipynb
```
Run all the cells sequentially. The contents of the source ZIP files in ```data/``` are unzipped by the program at ```data/denoising/```.
After the unzipping, the directory structure must look like this-  
```
.
├── data
      ├── denoising
      │      ├── set 1 - Clean and noisy-20250418T102555Z-001
      │      └── set 2 - only noisy-20250418T102559Z-001
      ├── set 1 - Clean and noisy-20250418T102555Z-001.zip
      └── set 2 - only noisy-20250418T102559Z-001.zip
```

The code generates a folder ```noise_results_analysis```. This folder has a subdirectory ```noise_results_analysis/plots``` which contains the visualizations of noise characteristics and sample waveforms.