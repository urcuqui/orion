# Orion Project

Orion is an AI security framework inspired by the principles of "The Art of War" and designed for both red (offensive) and blue (defensive) AI teams. It enables developers and security professionals to identify and address vulnerabilities using the MITRE ATT&CK framework, and provides a comprehensive suite of adversarial machine learning tools.

## Features

- **Red Team (Offensive AI):**
  - Generate adversarial examples and phishing emails.
  - Test model robustness with adversarial attacks.
  - Integrate with DeepSeek and other LLMs for offensive security automation.

- **Blue Team (Defensive AI):**
  - Defensive tools to detect and mitigate adversarial attacks.
  - Analyze and strengthen model security.

- **MITRE ATT&CK Integration:**
  - Map AI threats and vulnerabilities to the MITRE ATT&CK framework.
  - Understand and visualize your AI system's security posture.

- **Web Interface:**
  - Flask-based web app for easy interaction.
  - Upload models, generate adversarial images, and chat with AI agents.

## Requirements

- Python 3.11+
- Flask
- Other dependencies (see `requirements.txt`)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/urcuqui/orion.git
   cd orion
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://localhost:5000`

## Project Structure

- `app.py` - Main Flask application.
- `libs/` - Core libraries for adversarial attacks, agent wrappers, and utilities.
- `prompts/` - System prompts and templates for LLMs.
- `static/` - Static files (images, adversarial outputs, etc.).
- `templates/` - HTML templates for the web interface.
- `weights/` - Pretrained model weights.

## Disclaimer

This project is for research and educational purposes only. Use responsibly and ethically.

---

Christian Camilo Urcuqui López