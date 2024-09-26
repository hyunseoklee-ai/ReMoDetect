# ReMoDetect

Official PyTorch implementation of "ReMoDetect: Reward Models Recognize Aligned LLM's Generations".

## Prepare Data

- Download the cleaned full-text HC3 dataset in English from [YuchuanTian/AIGC_text_detector.git](https://github.com/YuchuanTian/AIGC_text_detector.git).
- Download the `unfilter_full/en_train_cleaned.csv` file from the repository linked above and extract it into the `./data` directory.

## Train and Evaluation Instructions

1. **Configure Environment Variables**

   Set your Groq API key in the `data_process.sh` script.

   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

2. **Run Data Processing Script**

   Execute the `data_process.sh` script to process the dataset.

   ```bash
   bash data_process.sh
   ```

3. **Train**

   Execute the `train.sh` script to process the dataset.

   ```bash
   bash train.sh
   ```

   Or, you can get trained weight from [huggingface](https://huggingface.co/hyunseoki/ReMoDetect-deberta)

4. **Generate Evaluation Data (Optional)**

   If you want to generate additional evaluation data, place your Azure, Anthropic, and Groq API keys in the `gen_eval_data.sh` script.

   ```bash
   export AZURE_API_KEY="your_azure_api_key_here"
   export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

   Then, run the script:

   ```bash
   bash gen_eval_data.sh
   ```

5. **Evaluate**

   Finally, run the `eval.sh` script to evaluate the model.

   ```bash
   bash eval.sh
   ```

    The benchmark based on the [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt.git) project and some codes include their licences. 
<!-- ## Machine Specifications

- **CPU:** Intel(R) Xeon(R) Gold 6426Y CPU @ 2.50GHz
- **GPU:** NVIDIA A6000 48GB -->
