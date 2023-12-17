curl -X 'POST' \
  'http://localhost:8000/embeddings/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text_list": ["bird", "car", "dog"],
  "image_paths": ["path/to/bird_image.jpg", "path/to/car_image.jpg", "path/to/dog_image.jpg"],
  "audio_paths": ["path/to/bird_audio.wav", "path/to/car_audio.wav", "path/to/dog_audio.wav"]
}'
