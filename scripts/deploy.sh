#!/usr/bin/env bash
# deploy.sh — Build and push the Docker image to AWS ECR, then trigger deploy
# Usage: AWS_ECR_REPO=my-repo AWS_REGION=us-east-1 bash scripts/deploy.sh
# Requires: AWS CLI v2, Docker, an existing ECR repo, and an EC2/ECS target
#           configured with the image (e.g. via GitHub Actions self-hosted runner).

set -euo pipefail

: "${AWS_ECR_REPO:?Set AWS_ECR_REPO (e.g. youtube-sentiment-analysis)}"
: "${AWS_REGION:?Set AWS_REGION (e.g. us-east-1)}"
: "${IMAGE_TAG:=$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"

ECR_REGISTRY="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_URI="${ECR_REGISTRY}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "[deploy] ECR: ${ECR_URI}/${AWS_ECR_REPO}:${IMAGE_TAG}"

echo "[deploy] Building Docker image..."
docker build -t "${AWS_ECR_REPO}:${IMAGE_TAG}" .
docker tag "${AWS_ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}/${AWS_ECR_REPO}:${IMAGE_TAG}"

echo "[deploy] Logging in to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_URI}"

echo "[deploy] Pushing image..."
docker push "${ECR_URI}/${AWS_ECR_REPO}:${IMAGE_TAG}"

echo "[deploy] Pushed. Trigger your deploy target (ECS / EC2 pull / k8s rollout) manually or via CI."
echo "[deploy] Done."
