# AI Voice Assistant Pipeline

This repository provides an AI Voice Assistant Pipeline that performs speech-to-text transcription using Whisper, processes the transcription with a language model (LLM) to generate responses, and converts the responses to speech using `espeak-ng`.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
  - [Transcriber](#transcriber)
  - [text_to_text](#text_to_text)
  - [text_to_audio_espeak](#text_to_audio_espeak)
  - [voice](#voice)
  - [final_pipeline](#final_pipeline)
- [Dependencies](#dependencies)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/susmita2022khatun/study-journey.git
   cd ai-voice-assistant-pipeline
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   !pip install torch torchaudio transformers jiwer
   !pip install git+https://github.com/openai/whisper.git
   !pip install webrtcvad
   ```

4. Download the necessary models for Whisper and Hugging Face:

   ```bash
   python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('huggyllama/llama-7b'); AutoModelForCausalLM.from_pretrained('huggyllama/llama-7b')"
   ```

## Usage

To use the pipeline, simply call the `final_pipeline` function with the required parameters: audio file path, speed, pitch, and gender for the voice synthesis.

```python
from pipeline import final_pipeline

audio_file = "path_to_audio_file.wav"
speed = 150
pitch = 70
gender = "female"

final_pipeline(audio_file, speed, pitch, gender)
```

## Functions

### Transcriber

The `Transcriber` class is responsible for handling speech-to-text transcription using Whisper and Voice Activity Detection (VAD) to detect voice segments.

- ****init**(model_name, vad_aggressiveness)**: Initializes the transcription model and VAD.
- **read_audio(file_path)**: Reads and preprocesses the audio file.
- **apply_vad(audio, sample_rate)**: Applies VAD to detect speech in the audio.
- **transcribe(audio_file)**: Transcribes the detected speech to text.

### text_to_text

The `text_to_text` function uses a language model to generate a response from the transcribed text.

- **text_to_text(transcription_file, response_file, max_length=150, num_return_sequences=1)**: Reads the transcription file, generates a response, and saves it to a response file.

### text_to_audio_espeak

The `text_to_audio_espeak` function converts text to speech using `espeak-ng`.

- **text_to_audio_espeak(text, output_file, pitch=70, speed=150, voice='en-us')**: Converts text to audio and saves it to the specified output file.

### voice

The `voice` function handles converting the generated response text to speech based on the specified gender.

- **voice(response_file, speed, pitch, gender)**: Converts the response text to speech with the specified parameters and saves it as an audio file.

### final_pipeline

The `final_pipeline` function orchestrates the entire process: transcribing the audio file, generating a response using the LLM, and converting the response to speech.

- **final_pipeline(audio_file, speed, pitch, gender)**: Executes the full pipeline.

## Dependencies

The project requires the following libraries:

- `webrtcvad`
- `numpy`
- `pydub`
- `whisper`
- `transformers`
- `torch`
- `espeak-ng`
- `IPython`
