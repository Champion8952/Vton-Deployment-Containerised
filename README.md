docker build -t virtual_tryon_testing .

docker tag virtual_tryon_testing gcr.io/ailusion-vton-backend/virtual_tryon_testing

docker push gcr.io/ailusion-vton-backend/virtual_tryon_testing

gcloud run services update virtual-tryon \
  --image gcr.io/ailusion-vton-backend/virtual_tryon_testing \
  --project ailusion-vton-backend

gcloud run services describe virtual-tryon \
  --project ailusion-vton-backend


!-------------------------------------------------------!

# Deploy Clsuter
gcloud container clusters create vton-cluster --region asia-southeast1 --release-channel regular --machine-type n1-standard-4 --enable-autoscaling --min-nodes 1 --max-nodes 10

# Create L4 Pool

gcloud beta container --project "ailusion-vton-backend" node-pools create "l4-pool" --cluster "vton-cluster" --region "asia-southeast1" --node-version "1.30.6-gke.1125000" --machine-type "g2-standard-8" --accelerator "type=nvidia-l4,count=1" --image-type "COS_CONTAINERD" --disk-type "pd-balanced" --disk-size "150" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append" --spot --num-nodes "3" --enable-autoscaling --min-nodes "0" --max-nodes "3" --location-policy "BALANCED" --enable-autoupgrade --enable-autorepair --max-surge-upgrade 0 --max-unavailable-upgrade 1

# Create A100 Pool

gcloud beta container --project "ailusion-vton-backend" node-pools create "a100-pool" --cluster "vton-cluster" --region "asia-southeast1" --node-locations "asia-southeast1-c" --node-version "1.30.6-gke.1125000" --machine-type "a2-ultragpu-1g" --accelerator "type=nvidia-a100-80gb,count=1" --image-type "COS_CONTAINERD" --disk-type "pd-balanced" --disk-size "150" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append" --spot --num-nodes "3" --enable-autoscaling --min-nodes "0" --max-nodes "3" --enable-autoupgrade --enable-autorepair --max-surge-upgrade 0 --max-unavailable-upgrade 1

# HPA Enable
kubectl autoscale deployment virtual-tryon --cpu-percent=80 --min=1 --max=10

# Check Pods Gpu Enabled Or Not 
kubectl exec -it virtual-tryon-67b8877f5f-f5pg7 -- nvidia-smi
