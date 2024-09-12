# Obfuscation

#### Code extracts and obfuscates a TensorFlow Keras model's layer details, modifies the architecture by adding shortcuts and extra layers, and then reconstructs and saves the obfuscated model.

1 .  The code loads a trained Keras model, extracts detailed information about its layers, and saves this information to a JSON file. <br>
2 .  Obfuscate Layer Names 


### Steps followed here are
- Parameter encapsulation
- Neural structure obfuscation
- Shortcut injection
- Extra layer injection

### Accuracy :
- From 0% Accuracy to 3% Accuracy increased

## Usage

```
Python Version: 3.12.4
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```
**Run**: ``` python scripts/example.py```

for more info: https://docs.google.com/document/d/1rm3DMc_ye_7aPDrB4i2MhuqsGvy8oGxkHZ0czYoO39Q/edit?usp=sharing 
## Reference paper
[ModelObfuscator: Obfuscating Model Information to Protect Deployed ML-Based Systems](https://arxiv.org/pdf/2306.06112)<br>
ISSTA ’23, July 17–21, 2023, Seattle, WA, USA Mingyi Zhou, Xiang Gao, Jing Wu, John Grundy, Xiao Chen, Chunyang Chen, and Li Li
