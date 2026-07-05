# Spotify Review Discovery Engine

## Objective

Spotify Review Discovery Engine is a Python application intended to support future exploration, analysis, and insight generation from Spotify review data.

This initial scaffold provides a clean project structure for future development. Business logic for data loading, preprocessing, Gemini integration, analysis, and visualization has not been implemented yet.

## Folder Structure

```text
spotify review discovery engine/
├── data/
│   └── .gitkeep
├── output/
│   └── .gitkeep
├── src/
│   └── __init__.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

## Setup Instructions

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python main.py
```

Expected output:

```text
Spotify Review Discovery Engine
```

## Future Modules

Planned development areas:

- Data loading
- Data preprocessing
- Review cleaning and normalization
- Sentiment analysis
- Topic discovery
- Google Gemini integration
- Exploratory data analysis
- Visualizations
- Report generation

