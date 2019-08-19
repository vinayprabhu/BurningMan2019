"""Synthesizes speech from the input string of text or ssml.

Note: ssml must be well-formed according to:
    https://www.w3.org/TR/speech-synthesis/
"""
from google.cloud import texttospeech
from google.cloud.texttospeech import enums
import os
print(os.getcwd())
print(os.listdir())
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Text2speech-*********.json"

# Instantiates a client
client = texttospeech.TextToSpeechClient()
voices = client.list_voices()
audio_dir=os.path.join(os.getcwd(),'audio_files')
i=0
for voice in voices.voices:
	# Display the voice's name. Example: tpc-vocoded
	print('Name: {}'.format(voice.name))

	# Display the supported language codes for this voice. Example: "en-US"
	for language_code in voice.language_codes:
		print('Supported language: {}'.format(language_code))

	ssml_gender = enums.SsmlVoiceGender(voice.ssml_gender)

	# Display the SSML Voice Gender
	print('SSML Voice Gender: {}'.format(ssml_gender.name))

	# Display the natural sample rate hertz for this voice. Example: 24000
	print('Natural Sample Rate Hertz: {}\n'.format(
		voice.natural_sample_rate_hertz))
	# Set the text input to be synthesized
	synthesis_input = texttospeech.types.SynthesisInput(text="Hey, Chip!")
	# Build the voice request, select the language code and the ssml voice gender ("neutral")
	voice = texttospeech.types.VoiceSelectionParams(
		language_code=language_code,
		ssml_gender=ssml_gender)

	# Select the type of audio file you want returned
	audio_config = texttospeech.types.AudioConfig(
		audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16)

	# Perform the text-to-speech request on the text input with the selected
	# voice parameters and audio file type
	response = client.synthesize_speech(synthesis_input, voice, audio_config)

	# The response's audio_content is binary.
	
	audio_file_name='hello_chip_'+str(i)+'.wav'
	i+=1
	audio_file_path=os.path.join(audio_dir,audio_file_name)
	with open(audio_file_path, 'wb') as out:
		# Write the response to the output file.
		out.write(response.audio_content)
		print('Audio content written to file: '+audio_file_name)
			
			
			
			
			
			




	
	