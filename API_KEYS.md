# 🔑 API Key Configuration Guide

## 📋 Overview

SceneScriber AI supports multiple AI providers for generating scene descriptions. You can use one or more of these providers by adding API keys to the backend configuration.

## 🚀 Quick Setup

1. **Copy the example file:**
   ```bash
   cd backend
   cp .env.example .env
   ```

2. **Edit the `.env` file** and add your API keys.

3. **Restart the backend server.**

## 🔐 Available AI Providers

### 1. **OpenAI GPT-4 Vision** (Recommended)
- **Best quality** descriptions
- **Fastest** processing
- **Most expensive** option

**Setup:**
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Add to `.env`:
   ```
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

**Cost:** ~$0.01-0.10 per video (depending on length)

### 2. **Anthropic Claude 3** (Alternative)
- **Excellent quality**, natural language
- **Good alternative** to OpenAI
- **Similar pricing**

**Setup:**
1. Go to https://console.anthropic.com/settings/keys
2. Create a new API key
3. Add to `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### 3. **Google Gemini** (Cost-effective)
- **Good performance**, lower cost
- **Free tier** available
- **Easy setup**

**Setup:**
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Add to `.env`:
   ```
   GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

### 4. **Local LLaVA** (Future)
- **Privacy-focused**, no API calls
- **Free**, runs locally
- **Slower** processing
- *Not yet implemented*

## 💰 Cost Estimation

| Provider | Cost per 7-min video | Free Tier |
|----------|---------------------|-----------|
| OpenAI | $0.05-0.15 | $5 free credit |
| Claude | $0.03-0.10 | Limited free tier |
| Gemini | $0.01-0.05 | Generous free tier |

**Tip:** Start with Gemini's free tier for testing.

## ⚙️ Configuration File

The `.env` file in the `backend/` directory:

```env
# AI Provider API Keys
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# Application Settings
MAX_UPLOAD_SIZE=2147483648  # 2GB
DEFAULT_AI_MODEL=openai     # openai, claude, gemini
```

## 🧪 Testing API Keys

1. **Check backend status:**
   ```bash
   curl http://localhost:8000/
   ```

2. **Check available providers:**
   ```bash
   curl http://localhost:8000/api/config
   ```

3. **Test with a small video** to verify everything works.

## 🔒 Security Notes

1. **Never commit** `.env` to git (it's in `.gitignore`)
2. **Rotate keys** regularly
3. **Set usage limits** in your provider dashboard
4. **Use environment variables** in production

## 🆓 Using Without API Keys

The application works without API keys:
- ✅ Scene detection (FFmpeg)
- ✅ SRT export
- ✅ Mock AI descriptions (editable)
- ✅ Full workflow

You'll see warnings in the UI about missing API keys, but all features except real AI descriptions will work.

## 🚨 Troubleshooting

### "No AI providers available"
- Check `.env` file exists in `backend/`
- Verify API keys are correct
- Restart the backend server

### "API key invalid"
- Regenerate the API key
- Check for typos
- Verify the key has proper permissions

### "Rate limit exceeded"
- Wait before trying again
- Consider upgrading your plan
- Use a different provider

## 📞 Support

For API key issues:
- OpenAI: https://help.openai.com/
- Anthropic: https://support.anthropic.com/
- Google: https://developers.google.com/ai

For application issues:
- Check the logs: `backend/scenescriber.log`
- Restart the backend server
- Verify FFmpeg is installed

---

**Next:** Upload a video and test the workflow with your configured API keys! 🎬