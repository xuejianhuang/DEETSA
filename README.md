# Dual evidence enhancement and text-image similarity awareness for early multimodal rumor detection

The widespread use of social media has facilitated the dissemination and exchange of information, but it has also created a breeding ground for spreading rumors. The rampant spread of rumors has caused severe negative impacts on society. With the rapid growth of multimodal content on social media, multimodal rumor detection has gained increasing attention. Currently, most methods focus on learning the deep semantics of various modalities and integrating them through traditional fusion methods (such as concatenation or addition) to achieve complementary information between different modalities, which has achieved some success. However, these methods have two key issues: (1) Shallow cross-modal feature fusion makes capturing the semantic inconsistency between text and images difficult. (2) Relying solely on text and image content makes identifying some meticulously designed deep-mismatched rumors difficult. Therefore, we propose a multimodal rumor detection method based on **D**ual **E**vidence **E**nhancement and **T**ext-image **S**imilarity **A**wareness (**DEETSA**). Specifically, our method consists of three main modules: (1) A text-image similarity awareness module that calculates the semantic similarity between text and images to identify semantic inconsistency; (2) A dual evidence enhancement module that filters relevant textual and visual evidence from external knowledge bases and aligns the evidence with the original post using cross-attention, thereby improving the model's ability to detect deep mismatched rumors; (3) A feature hierarchical fusion method based on a gated neural network that adapts to the different roles and noise levels of features from different modalities at different times. Experimental results on real-world social media datasets from Weibo and Twitter show that our proposed method outperforms the state-of-the-art baseline methods.

The framework of the proposed DEETSA model:

<div align="center">
<img src='./fig/model.png' width='90%'>
</div>

## Datasets
We use the latest multimodal rumor dataset **MR<sup>2</sup>** released by   <a href="https://doi.org/10.1145/3539618.3591896" target='_blank'>Hu et al.</a>.  **MR<sup>2</sup>** consists of two rumor detection datasets: one in English and one in Chinese. The English dataset is constructed from posts on Twitter, while the Chinese dataset is sourced from posts on Weibo. The posts in these datasets are categorized into rumors, non-rumors, and unverifiable rumors.

We have preprocessed the original data. For the complete dataset, please download it from Google Drive  <a href="https://drive.google.com/file/d/14NNqLKSW1FzLGuGkqwlzyIPXnKDzEFX4/view?usp=sharing" target='_blank'>link</a>  or Baidu Cloud  <a href="https://pan.baidu.com/s/1OV_Oab0zQgI8P2Wo1qwBuw?pwd=1odv" target='_blank'>link</a>.

### Typical forms of multimodal rumors
<div align="center">
<img src='./fig/rumor.png' width='90%'>
</div>

## Dependencies
* wordcloud==1.8.1
* torch==1.12.1
* torchvision==0.13.1
* tqdm==4.63.1
* Pillow==8.4.0
vtorchmetrics==1.4.0.post0
* pandas==1.1.5
* seaborn==0.11.2
* transformers==4.41.2
* numpy==1.26.4
* jieba==0.42.1
* matplotlib==3.3.4

## Run
python main.py --dataset twitter --model DEETSA

## Acknowledgements
Thank you to **Xuming Hu** (Tsinghua University, Beijing, China), **Zhijiang Guo** (University of Cambridge, Cambridge, United Kingdom), **Junzhe Chen** (Tsinghua University, Beijing, China), **Lijie Wen** (Tsinghua University, Beijing, China), and **Philip S Yu** (University of Illinois at Chicago, Chicago, IL, USA) for providing the dataset.

