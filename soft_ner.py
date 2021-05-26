# %%
from flair.data import Corpus
from flair.datasets import STACKOVERFLOW_NER

corpus: Corpus = STACKOVERFLOW_NER()
print(corpus)

# %%
# 2. make the tag dictionary from the corpus
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# %%
# 3. initialize embeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings, ELMoEmbeddings, StackedEmbeddings

embedding_types = [

    WordEmbeddings('glove'),
    # ELMoEmbeddings(),
    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# %%
# 4. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=64,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True,
                                        dropout=0.1,
                                        word_dropout=0,
                                        locked_dropout=0,
                                        train_initial_hidden_state=True,
                                        use_rnn=False,
                                        rnn_layers=1)

# %%
# 5. initialize trainer
from flair.trainers import ModelTrainer
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import OneCycleLR

trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=Adam)

# %%
# 6. start training
trainer.train('resources/taggers/soft_ner',
              learning_rate=0.03,
              mini_batch_size=124,
              max_epochs=2,
              monitor_train=True,
              monitor_test=True,
              weight_decay=0.01,
              scheduler=OneCycleLR)
# %%
trainer.find_learning_rate('resources/taggers/soft_ner')
# %%
from flair.visual.training_curves import Plotter

plotter = Plotter()
plotter.plot_learning_rate("resources/taggers/soft_ner/learning_rate.tsv")

# %%
# 7. Eval training

trainer.final_test('resources/taggers/soft_ner',
                    eval_mini_batch_size=64)


# %%
from flair.data import Sentence
from flair.models import SequenceTagger

# load the model you trained
model = SequenceTagger.load('resources/taggers/soft_ner/final-model.pt')

# %%
# create example sentence
sentence = Sentence("""

Is it possible to only merge some columns? I have a DataFrame df1 with columns x, y, z, and df2 with columns x, a ,b, c, d, e, f, etc.

I want to merge the two DataFrames on x, but I only want to merge columns df2.a, df2.b - not the entire DataFrame.

The result would be a DataFrame with x, y, z, a, b.

I could merge then delete the unwanted columns, but it seems like there is a better method.
""")

# predict tags and print
model.predict(sentence)

# %%
sentence.to_tagged_string()
# %%
for entity in sentence.get_spans('ner'):
    print(entity)
# %%
