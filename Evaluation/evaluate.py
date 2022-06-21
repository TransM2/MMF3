from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk
# nltk.download('wordnet')

def nltk_corpus_bleu(hypothesis, reference):
    hyps = []
    refs = []
    for hyp, ref in zip(hypothesis, reference):
        hyp = hyp.split()
        ref = ref.split()
        hyps.append(hyp)
        refs.append([ref])
    corpus_bleu = nltk.translate.bleu_score.corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25))
    return corpus_bleu

def meteor_score1(hypothesis, reference):
    count = 0
    total_score = 0.0
    for i in range(len(hypothesis)):
        score = round(meteor_score([reference[i]], hypothesis[i]), 4)
        # print(score)
        # exit()
        total_score += score
        count += 1
    METEOR = total_score/count
    # print('METEOR_score: %.4f' % avg_score)
    return METEOR

pred1 = []
with open(save_pred_dir, 'r', encoding='utf-8') as f1:  # ref1
    lines = f1.readlines()
    for line in lines:
        pred1.append(line)

ref = []
with open(tar_dir, 'r', encoding='utf-8') as f2:  # ref1
    lines = f2.readlines()

    for line in lines:
        # line = line.strip('\n')
        # print(line)
        ref.append(line)
# print(ref)
avg_score = nltk_corpus_bleu(pred1, ref)
meteor = meteor_score1(pred1, ref)
print('BLEU: %.4f' % avg_score)
print('METEOR: %.4f' % meteor)
rouge = Rouge()
rough_score = rouge.get_scores(pred1, ref, avg=True)
print(' ROUGE: ', rough_score)
