import sys

from em_influence.rates import misaligned_rates, question_categories

sys.stderr = open(snakemake.log[0], "w")

categories = question_categories(snakemake.input.categories)
misaligned_rates(snakemake.input.answers, categories).to_csv(snakemake.output[0], index=False)
