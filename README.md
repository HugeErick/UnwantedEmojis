# UnwantedEmojis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Resources

- Dataset used: [https://drive.google.com/file/d/1TSVdEaW8Rwu6RX5vyANvchwEAEL6ZHwy/view?usp=sharing](https://drive.google.com/file/d/1TSVdEaW8Rwu6RX5vyANvchwEAEL6ZHwy/view?usp=sharing)

Special thanks to the following autors that left their dataset for public use: tenifayo autor of the following source.

source: [https://universe.roboflow.com/tenifayo/emotion-recognition-vweyv](https://universe.roboflow.com/tenifayo/emotion-recognition-vweyv)

The Google Collab version was the final version, and u can try running it with the dataset shared above, however I mounted my google drive in order to work:

- Custom CNN: [https://colab.research.google.com/drive/1bbzkgI7p2nWSBEDo732i-o1S3yvN49mG?usp=sharing](https://colab.research.google.com/drive/1bbzkgI7p2nWSBEDo732i-o1S3yvN49mG?usp=sharing)
- ViTs approach: [https://colab.research.google.com/drive/1V2aRvGEsjTeKNPgrVKAhZ_B5PrUuc9KJ?usp=sharing](https://colab.research.google.com/drive/1V2aRvGEsjTeKNPgrVKAhZ_B5PrUuc9KJ?usp=sharing)

## Description

This project is a Python script that uses OpenCV and a pre-trained emotion detection model to detect and display emotions in real-time video streams. The script uses a modern emotion detection model based on a Convolutional Neural Network (CNN) and can be used to detect emotions in a webcam feed or a video file.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Windows/Linux/macOS](#windowslinuxmacos)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Windows/Linux/macOS (Local)

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

4. Unzip the dataset in order to set the following path

  `~/UnwantedEmojis/src/utils/custom_dataset$ ls
  angry  disgust  fear  happy  neutral  sad  surprise`

## Usage

### Basic Usage

To run the emotion detection system:

```bash
python src/main.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Erick Gonzalez Parada - <erick.parada101@gmail.com>

Project Link: [https://github.com/HugeErick/UnwantedEmojis](https://github.com/HugeErick/UnwantedEmojis)
