# -*- coding: utf-8 -*-
"""Fetch vocalized data.ipynb
"""

import re
import warnings
import pandas as pd
from tf.app import use
from datasets import load_dataset
from hebrewtools.functions import sbl_normalization

warnings.filterwarnings("ignore")

A = use('etcbc/bhsa', hoist=globals())

verses = ["{} {}:{}".format(*T.sectionFromNode(verse)) for verse in F.otype.s('verse')]
raw_texts = [T.text(verse, fmt='text-orig-full').strip() for verse in F.otype.s('verse')]
raw_texts = [" ".join([word for word in text.split() if word not in ['\u05e1', '\u05e4', '']]) for text in raw_texts]

pattern = re.compile("[^\s\u05d0-\u05ea\u05b0-\u05bc\u05be\u05c1\u05c2\u05c7]")

stripped_texts = [re.sub(pattern, "", text) for text in raw_texts]

pattern = re.compile("\u05b9\u05d5") #swap holam and vav

stripped_texts = [re.sub(pattern, "\u05d5\u05ba", text) for text in stripped_texts]

normalized_texts = [sbl_normalization(text) for text in stripped_texts]

df = pd.DataFrame({'Verse' : verses, 'Text' : normalized_texts})

replacement_vals_books = {'Genesis' : 'Gen', 
                          'Exodus' : 'Exod', 
                          'Numbers' : 'Num', 
                          'Leviticus' : 'Lev', 
                          'Deuteronomy' : 'Deut', 
                          'Joshua' : 'Josh', 
                          'Judges' : 'Jud',
                          '1_Samuel' : '1 Sam', 
                          '2_Samuel' : '2 Sam', 
                          '1_Kings' : '1 Kgs', 
                          '2_Kings' : '2 Kgs', 
                          'Isaiah' : 'Isa', 
                          'Jeremiah' : 'Jer', 
                          'Ezekiel' : 'Ezek', 
                          'Hosea' : 'Hos', 
                          'Obadiah' : 'Obad', 
                          'Michah' : 'Mic', 
                          'Nahum' : 'Nah', 
                          'Habakkuk' : 'Hab', 
                          'Zephaniah' : 'Zeph', 
                          'Haggai' : 'Hag', 
                          'Zechariah' : 'Zech', 
                          'Malachi' : 'Mal', 
                          'Psalms' : 'Ps', 
                          'Proverbs' : 'Prov', 
                          'Song of Songs' : 'Song', 
                          'Ecclesiastes' : 'Eccl', 
                          'Lamentations' : 'Lam', 
                          'Esther' : 'Esth', 
                          'Daniel' : 'Dan', 
                          'Nehemiah' : 'Neh', 
                          '1_Chronicles' : '1 Chr', 
                          '2_Chronicles' : '2 Chr'}

df['Verse'].replace(replacement_vals_books, inplace=True, regex=True)

df.set_index('Verse', inplace=True)

df.drop('Jer 10:11', inplace=True)
df.drop(df.loc['Dan 2:5' : 'Dan 7:28'].index, inplace=True)
df.drop(df.loc['Ezra 4:8' : 'Ezra 6:18'].index, inplace=True)
df.drop(df.loc['Ezra 7:12' : 'Ezra 7:26'].index, inplace=True)

df.loc['Dan 2:4']['Text'] = 'וַיְדַבְּרוּ הַכַּשְדִּים לַמֶּלֶךְ אֲרָמִית'

df.to_csv('Tanakh_data.csv')

vocalized = load_dataset('csv', data_files='Tanakh_data.csv')['train']
vocalized = vocalized.remove_columns('Verse')
vocalized = vocalized.train_test_split(test_size=.1, shuffle=True)
vocalized.push_to_hub('gngpostalsrvc/Tanakh')
