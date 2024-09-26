### Before data process download cleaned-HC3 as README.md explained.
### https://github.com/YuchuanTian/AIGC_text_detector.git

### Preprocess the HC3 dataset into the format required by the model.
python scripts/preprocess.py

### Rephrase the Human paragraph using Llama3-70B-Instruct of Meta.
export GROQ_API_KEY=xxxxxxxx
python scripts/generate_para.py