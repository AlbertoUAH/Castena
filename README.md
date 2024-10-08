# Castena: LLM Chatbot for Interacting with Spanish Podcasts

<p align="center">
    <img src="./media/castena-animated-icon.gif" width="300px" style="display: block; margin: 0 auto">
</p>

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The LLM Chatbot for Interacting with Spanish Podcasts is an AI-powered chatbot designed to enhance your podcast listening experience. It is specifically tailored for Spanish-speaking podcast enthusiasts who want to engage with and learn from their favorite Spanish podcasts in a whole new way.

This chatbot utilizes advanced natural language processing techniques to provide real-time text-based interactions with Spanish podcasts. Whether you want to generate transcripts, summarize episodes, or seek information about specific topics within a podcast, this chatbot is here to assist you.

## Proposal

<p align="center">
    <img src="./media/architecture.png" alt="Architecture" />
</p>

## Objectives

<p align="center">
    <a href="https://www.youtube.com/@worldcast_">
        <img src="./media/worldcast_icon.png" alt="Worldcast" />
    </a>
    <a href="https://www.youtube.com/@TheWildProject">
        <img src="./media/wild_project_icon.png" alt="Wild Project" />
    </a>
    <a href="https://www.youtube.com/@WallStreetWolverine">
        <img src="./media/wall_st_wolverine.jpeg" alt="Wall Street Wolverine" />
    </a>
    <a href="https://www.youtube.com/@marc_vidal">
        <img src="./media/marc_vidal_icon.jpeg" alt="Marc Vidal" />
    </a>
</p>

| Video Name            | Channel     | Topic  | Status |
|-----------------------|-------------|--------|--------|
| [Worldcast #45 - Roberto Vaquero](https://www.youtube.com/watch?v=4sXT1tHVbjE) | Worldcast   | Politics | Available |
| [Marc Vidal - entrevista a Jose Luis Cava](https://www.youtube.com/watch?v=CBmTWyIbhI4&t=1513s) | Marc Vidal   | Economy and Politics | Available |
| [Wall Street Wolverine - entrevista a Samuel Vázquez](https://www.youtube.com/watch?v=-JSF3nmPI-Y) | Wall Street Wolverine | Politics and Society | Available |
| [The Wild Project - entrevista a Operador Nuclear](https://www.youtube.com/watch?v=k1PvPo8Qa9g) | The Wild Project | Society | Available |

## Features

- **Transcription**: Get accurate text transcriptions of podcast episodes.
- **Summarization**: Receive concise summaries of podcast episodes.
- **Search**: Find specific information, keywords, or topics within podcasts.
- **Language Support**: Currently designed only to interact in English, translating the outputs into Spanish (Spanish based LLM models will be available soon...).

## Getting Started

### Prerequisites

Before you begin, ensure you have the following requirements in place:

- **Python**: You need Python 3.7 or later installed on your system.
- **Virtual Environment (Optional)**: It's recommended to create a virtual environment to manage dependencies.
- **GPU**: not mandatory, but recommended (GPU used: Nvidia T4)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AlbertoUAH/Castena.git
   cd Castena
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Run app (via Streamlit):

   ```bash
   streamlit run app.py --logger.level=warning 2 > 'app_log.log'
   ```

## Relevant files

```
src/
    ├── transcriptions_extraction_whipser.ipynb: notebook with the code for step1
    ├── prod/
        ├── app.py: main application
        ├── utils.py
        └── requirements.txt
```

## Evaluation (Coming soon)

To evaluate our approach, labelled datasets are needed. To do so, we employ [Label Studio library](https://labelstud.io/)

<p align="center">
 <img src="./media/label_studio.png" width="650px">
</p>

<p align="center">
 <img src="./media/label_studio_sample.png" width="650px">
</p>

### Metrics (Coming soon)

To evaluate LLM performance, [QAEvalChain](https://api.python.langchain.com/en/latest/evaluation/langchain.evaluation.qa.eval_chain.QAEvalChain.html), [cosine_similarity](https://www.engati.com/glossary/cosine-similarity#:~:text=In%20NLP%2C%20Cosine%20similarity%20is,in%20a%20multi%2Ddimensional%20space.) and a [sentence similarity model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) are used

#### Results

| Video Name            | Channel     | Topic  | QAEvalChain (% Corrected Answer) | Mean Cosine Similarity | Median Cosine Similarity | Mean Sentence Similarity | Median Sentence Similarity |
|-----------------------|-------------|--------|---------------------------------|------------------------|--------------------------|------------------------|--------------------------|
| Worldcast #45 - Roberto Vaquero | Worldcast   | Politics & Opinion| 20 out of 30 corrected answers (~67 %) | 0.7943 | 0.8198 | 0.7225 | 0.7406 |


## Contributing

We welcome contributions to improve the functionality and capabilities of this chatbot. If you'd like to contribute, please follow these guidelines:

  1. Fork the repository.
  
  2. Create a new branch for your feature or bug fix: `git checkout -b feature/new-feature`.
  
  3. Make your changes and test thoroughly.
  
  4. Commit your changes with a clear and concise message: `git commit -m "Add new feature"`.
  
  5. Push your branch to your forked repository: `git push origin feature/new-feature`.
  
  6. Create a pull request against the main repository's `main` branch.
  
  7. Provide a detailed description of your changes and why they are valuable.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Tools

* Colab notebooks (free tier)
* [TogetherAI](https://together.ai/) (LLM support)

## Resources

* [Touvron Hugo, Martin Louis, Stone Kevin (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. GenAI, Meta](https://arxiv.org/abs/2307.09288)
* [TogetherAl: an Al acceleration cloud](https://www.together.ai/)
* [Langchain documentation](https://python.langchain.com/v0.2/docs/introduction/)
* [Langchain Al Handbook, by James Briggs & Francisco Ingham](https://www.pinecone.io/learn/series/langchain/)
* [OpenAl documentation](https://www.openal.org/documentation/)
* [Chroma Database](https://www.trychroma.com/)
* [Whisper: GitHub repository](https://github.com/openai/whisper)
* [Pyannote - Speaker diarization: HuggingFace repository](https://huggingface.co/pyannote/speaker-diarization)
* [Streamlit documentation](https://docs.streamlit.io/)
* [Ortus Al - YouTube channel](https://www.youtube.com/@OrtusAI/about)
* [Lex Friedman Podcast: Create Summaries of your favourite Podcasts using Al](https://medium.com/@felipecembranelli/lex-fridman-podcast-create-summaries-of-your-favorite-podcasts-using-ai-a29a5923a445)

## [Buy me a beer! 🍺](https://www.buymeacoffee.com/castena)
