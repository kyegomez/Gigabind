curl -X 'POST' \
  'http://localhost:8000/embeddings/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text_list": ["bird", "car", "dog"]
}'
