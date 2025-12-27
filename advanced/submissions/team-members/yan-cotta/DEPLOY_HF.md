# Hugging Face Spaces Deployment Guide

## Overview

This guide walks you through deploying FinResearch AI to Hugging Face Spaces using Docker.

## Prerequisites

- A Hugging Face account
- OpenAI API key
- Git installed locally

---

## Phase 2: Platform Setup

### Step 1: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in the form:

| Field | Value |
|-------|-------|
| **Space name** | `finresearch-ai` (or your preferred name) |
| **License** | `MIT` |
| **SDK** | `Docker` |
| **Hardware** | `CPU basic` (free tier) or `CPU upgrade` for better performance |
| **Visibility** | `Public` or `Private` (your choice) |

4. Click **"Create Space"**

### Step 2: Configure Secrets

1. Go to your Space's **Settings** tab
2. Scroll to **"Repository secrets"**
3. Add the following secret:

| Secret Name | Value |
|-------------|-------|
| `OPENAI_API_KEY` | Your OpenAI API key (e.g., `sk-...`) |

⚠️ **Important**: Never commit your API key to the repository!

---

## Phase 3: Deployment

### Step 1: Prepare Your Repository

First, ensure you're in the project directory:

```bash
cd advanced/submissions/team-members/yan-cotta
```

### Step 2: Copy the HF Dockerfile

The Dockerfile needs to be named exactly `Dockerfile` for HF Spaces:

```bash
cp Dockerfile.hf Dockerfile
```

### Step 3: Add Hugging Face Remote

Replace `YOUR_HF_USERNAME` with your actual Hugging Face username:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/finresearch-ai
```

### Step 4: Push to Hugging Face

```bash
# Add all files
git add .

# Commit
git commit -m "Deploy to Hugging Face Spaces"

# Push to HF (you'll be prompted for credentials)
git push hf main
```

If your default branch is different:
```bash
git push hf your-branch:main
```

### Step 5: Monitor Deployment

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR_HF_USERNAME/finresearch-ai`
2. Click the **"Building"** status to see logs
3. Wait for the build to complete (5-10 minutes first time)

---

## Troubleshooting

### Build Fails

- Check the build logs in the HF Spaces interface
- Ensure `OPENAI_API_KEY` secret is set correctly
- Verify the Dockerfile is at the repository root

### App Shows Blank Page

- Check browser console for errors
- Verify static files are being served (check `/api/health` endpoint)

### API Returns 500 Error

- Check if `OPENAI_API_KEY` is set in Secrets
- Look at the Space logs for detailed error messages

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |
| `FINRESEARCH_MANAGER_MODEL` | No | Manager agent model (default: `gpt-4o-mini`) |
| `FINRESEARCH_WORKER_MODEL` | No | Worker agent model (default: `gpt-3.5-turbo`) |

---

## Local Testing with HF Dockerfile

To test the HF Dockerfile locally:

```bash
# Build
docker build -f Dockerfile.hf -t finresearch-hf .

# Run
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... finresearch-hf

# Open http://localhost:7860
```
