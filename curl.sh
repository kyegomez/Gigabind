curl -X 'POST' \
  'http://localhost:8000/embeddings/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text_list": ["bird", "car", "dog"],
  "image_paths": [".assets/bird_image.jpg", ".assets/car_image.jpg", ".assets/dog_image.jpg"],
  "audio_paths": [".assets/bird_audio.wav", ".assets/car_audio.wav", ".assets/dog_audio.wav"]
}'
