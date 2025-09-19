import utilities

files = ['docs/Andy-Warhol.txt',
         'docs/Ebola-virus-disease.txt',
         'docs/Life-indoors.txt',
         'docs/Napoleon-wiki.txt',
         'docs/Trump-wall.txt',
         'docs/Social-media.txt']
files_for_evaluetion = ['evaluation/Andy-Warhol.txt',
                        'evaluation/Ebola-virus-disease.txt',
                        'evaluation/Life-indoors.txt',
                        'evaluation/Napoleon-wiki.txt',
                        'evaluation/Trump-wall.txt',
                        'evaluation/Social-media.txt']

nasari_dictionary = utilities.create_nasari_dict()
document = utilities.read_document(files[3])
automatic_summarization = utilities.summarization(document, 30, nasari_dictionary, 'cue')
document_for_evaluation = utilities.read_document(files_for_evaluetion[3])

file_name = files[3].split('/')
if len(file_name) > 0:
    file_name = file_name[1]

utilities.write_summarization('Summarized_' + file_name, automatic_summarization)

###############################

# define reference and auto geerated summaries

# hand made summarization
ref_summaries = document_for_evaluation

# auto generated summarization
auto_generated_summary = automatic_summarization

# evaluating results using BLEU and ROUGE
bleu_score = utilities.bleu_evaluation(ref_summaries, auto_generated_summary)
rouge_score = utilities.rouge_evaluation(ref_summaries, auto_generated_summary)

print("BLEU score: " + str(round(bleu_score * 100, 2)) + "%")
print("ROUGE score: " + str(round(rouge_score * 100, 2)) + "%")
