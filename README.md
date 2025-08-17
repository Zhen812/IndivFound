# Individuals Using a Pretrained Biosignal Foundation Model
Official repo of IndivFound.

Please contact wmtan@fudan.edu.cn or byan@fudan.edu.cn if you hace any questions.

> Individual multi-dimensional states such as health, emotions, psychology, interests, preferences, cognition, etc., are challenging to characterize, yet are essential in understanding an individual’s emotional state, mental health, and cognitive functions. This understanding has significant implications for many individual’s applications such as precision personalized medicine, human-machine interaction, personnel allocation, etc. However, task-specific methods typically predict only a single state and have bias across different individuals. Here, we develop a biosignals based individual foundation model (IndivFound) to characterize multi-dimensional states of individuals. The model learns generalizable representations from terabytes of physical and physiological data that encompasses diverse gender and age groups, and six modalities (facial video, electroencephalogram, electrocardiogram, electrooculogram, electromyogram, and galvanic skin response). We demonstrate that adapted IndivFound consistently outperforms comparison models in individual characterization tasks of multiple levels, including low-level demographic decoding, mid-level health monitoring and emotion analysis, as well as high-level cognition and behavior analysis. In addition, the effectiveness of the model in the real application of physician fatigue grading shows its potential practicality in real scenarios. IndivFound also demonstrates unprecedented generalization in identifying individual intentions, emotions, and psychological states under substantial individual differences, exhibiting an unbiased representational ability concerning individuals’ age and gender. Our study opens the possibility of characterizing individual multi-dimensional complex states and overcoming significant differences in individual sensitive attributes, demonstrating the great potential of IndivFound to support a wide range of individual applications.

## Install environment
1. Create environment with conda:
   ```
    conda create -n indivfound python=3.11.0
3. Install dependencies：
   ```
    conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    git clone https://github.com/Zhen812/IndivFound/
    cd IndivFound
    pip install -r requirements.txt
