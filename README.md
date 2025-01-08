docker build -t virtual_tryon_testing .

docker tag virtual_tryon_testing gcr.io/ailusion-vton-backend/virtual_tryon_testing

docker push gcr.io/ailusion-vton-backend/virtual_tryon_testing

gcloud run services update virtual-tryon \
  --image gcr.io/ailusion-vton-backend/virtual_tryon_testing \
  --project ailusion-vton-backend

gcloud run services describe virtual-tryon \
  --project ailusion-vton-backend