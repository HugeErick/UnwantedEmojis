# UnwantedEmojis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Description

This project is a Python script that uses OpenCV and a pre-trained emotion detection model to detect and display emotions in real-time video streams. The script uses a modern emotion detection model based on a Convolutional Neural Network (CNN) and can be used to detect emotions in a webcam feed or a video file.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Windows/Linux/macOS](#windowslinuxmacos)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Command Line Arguments](#command-line-arguments)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Windows/Linux/macOS

1. Clone the repository:

   ```bash
   git clone https://github.com/HugeErick/UnwantedEmojis.git
   cd UnwantedEmojis
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

To run the emotion detection on your webcam:

```bash
python main.py
```

To process a video file:

```bash
python main.py --video path/to/your/video.mp4
```

### Command Line Arguments

- `--video`: Path to input video file (optional, defaults to webcam)
- `--model`: Path to the emotion detection model (default: models/emotion_model.hdf5)
- `--cascade`: Path to the Haar Cascade classifier (default: models/haarcascade_frontalface_default.xml)
- `--output`: Path to save the output video (optional)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Erick Gonzalez Parada - <erick.parada101@gmail.com>

Project Link: [https://github.com/HugeErick/UnwantedEmojis](https://github.com/HugeErick/UnwantedEmojis)
