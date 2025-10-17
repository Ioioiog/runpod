# RunPod AI Event Detection

[![Runpod](https://api.runpod.io/badge/Ioioiog/runpod)](https://console.runpod.io/hub/Ioioiog/runpod)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green)](https://developer.nvidia.com/cuda-toolkit)

AI-powered event detection system using YOLOv8n, BLIP-2, and NSFW filtering for real-time event reporting.

## Features

- **YOLOv8n**: Object detection (cars, people, fire, smoke, etc.)
- **BLIP-2**: Image captioning
- **NSFW Filter**: Content safety check
- **Face/Plate Blur**: Privacy protection

## Deployment

### 1. Build Docker Image

```bash
cd runpod
docker build -t your-username/event-detection:latest .
docker push your-username/event-detection:latest
```

### 2. Create RunPod Endpoint

1. Go to https://runpod.io
2. Sign up / Login
3. Navigate to "Serverless" → "New Endpoint"
4. Configure:
   - **Docker Image**: `your-username/event-detection:latest`
   - **GPU**: T4 ($0.00019/sec)
   - **Timeout**: 30 seconds
   - **Max Workers**: 3
5. Copy the endpoint URL and API key

### 3. Update Environment Variables

Add to your `.env.local`:

```env
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID
RUNPOD_API_KEY=your_runpod_api_key
```

## Testing Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run handler
python handler.py
```

## API Usage

```python
import requests
import base64

# Read image
with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Call endpoint
response = requests.post(
    'https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={
        'input': {
            'image': image_data,
            'tasks': ['detect', 'caption', 'nsfw', 'blur']
        }
    }
)

result = response.json()
print(result)
```

## Performance

- **YOLOv8n Detection**: ~5-10ms
- **BLIP-2 Caption**: ~50-100ms
- **Face Detection**: ~10-20ms
- **NSFW Check**: ~20-30ms
- **Total**: ~100-200ms on T4 GPU

## Cost Estimation

| Events/Month | Processing Time | Cost |
|--------------|----------------|------|
| 1,000 | 200ms avg | $0.38 |
| 10,000 | 200ms avg | $3.80 |
| 100,000 | 200ms avg | $38 |

Formula: `events × 0.2s × $0.00019/s = cost`

## Troubleshooting

### Cold Start Issues
- First request may take 5-10 seconds (model loading)
- Subsequent requests are fast (~200ms)
- Consider keeping endpoint warm with periodic pings

### Out of Memory
- Reduce batch size
- Use smaller models (YOLOv8n is already smallest)
- Increase GPU tier (T4 → A4000)

### Timeout Errors
- Increase timeout in RunPod settings
- Optimize image preprocessing
- Check network latency

## Support

For issues, contact: support@martor.online

[![Runpod](https://api.runpod.io/badge/Ioioiog/runpod)](https://console.runpod.io/hub/Ioioiog/runpod)
