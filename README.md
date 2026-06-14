# AI Job Application Agent

A Streamlit + LangGraph multi-agent app that generates a job application package from a resume PDF and a job description URL.

## Features

- JD Extractor Agent
- Resume Analyzer Agent
- Match Score Agent
- Cover Letter Agent
- Skill Gap Agent
- Final Package Agent
- Supports job URL extraction and manual JD fallback
- Download cover letter and full application package

## Local Setup

```bash
pip install -r requirements.txt
```

Create `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Run:

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this folder to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Select `app.py` as the main file.
4. Add this in Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

5. Deploy.

## Important Note

Some job sites block scraping or load content through JavaScript. If URL extraction fails, use the manual job description input box.
