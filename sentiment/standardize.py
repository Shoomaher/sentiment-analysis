import numpy as np
import tensorflow as tf
from unidecode import unidecode

CONTRACTIONS = {
    "I'm": "I am",
    "It's": "It is",
    "He's": "He is",
    "She's": "She is",
    "that's": "that is",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "I've": "I have",
    "isn't": "is not",
    "mayn't": "may not",
    "may've": "may have",
    "mightn't": "might not",
    "might've": "might have",
    "mustn't": "must not",
    "needn't": "need not",
    "should've": "should have",
    "shouldn't": "should not",
    "there're": "there are",
    "these're": "these are",
    "gotta": "going to",
    "wanna": "want to",
    "wasn't": "was not",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "wouldn't": "would not",
    "you're": "you are",
    "you've": "you have",
    "they're": "they are",
    "they've": "they have",
}
CHAT_WORDS = {
    'AFAIK': 'As Far As I Know',
    'AFK': 'Away From Keyboard',
    'ASAP': 'As Soon As Possible',
    'ATK': 'At The Keyboard',
    'ATM': 'At The Moment',
    'A3': 'Anytime, Anywhere, Anyplace',
    'BAK': 'Back At Keyboard',
    'BBL': 'Be Back Later',
    'BBS': 'Be Back Soon',
    'BFN': 'Bye For Now',
    'B4N': 'Bye For Now',
    'BRB': 'Be Right Back',
    'BRT': 'Be Right There',
    'BTW': 'By The Way',
    'B4': 'Before',
    'CU': 'See You',
    'CUL8R': 'See You Later',
    'CYA': 'See You',
    'FAQ': 'Frequently Asked Questions',
    'FC': 'Fingers Crossed',
    'FWIW': "For What It's Worth",
    'FYI': 'For Your Information',
    'GAL': 'Get A Life',
    'GG': 'Good Game',
    'GN': 'Good Night',
    'GMTA': 'Great Minds Think Alike',
    'GR8': 'Great',
    'G9': 'Genius',
    'IDK': 'i do not know',
    'IC': 'I See',
    'ICQ': 'I Seek you',
    'ILU': 'I Love You',
    'IMHO': 'In My Humble Opinion',
    'IMO': 'In My Opinion',
    'IOW': 'In Other Words',
    'IRL': 'In Real Life',
    'KISS': 'Keep It Simple, Stupid',
    'LDR': 'Long Distance Relationship',
    'LMAO': 'Laugh My Ass Off',
    'LOL': 'Laughing Out Loud',
    'LTNS': 'Long Time No See',
    'L8R': 'Later',
    'MTE': 'My Thoughts Exactly',
    'M8': 'Mate',
    'NRN': 'No Reply Necessary',
    'OIC': 'Oh I See',
    'OMG': 'oh my god',
    'PITA': 'Pain In The Ass',
    'PRT': 'Party',
    'PRW': 'Parents Are Watching',
    'ROFL': 'Rolling On The Floor Laughing',
    'ROFLOL': 'Rolling On The Floor Laughing Out Loud',
    'ROTFLMAO': 'Rolling On The Floor Laughing My Ass Off',
    'SK8': 'Skate',
    'STATS': 'Your sex and age',
    'ASL': 'Age, Sex, Location',
    'THX': 'Thank You',
    'TTFN': 'Ta-Ta For Now',
    'TTYL': 'Talk To You Later',
    'U2': 'You Too',
    'U4E': 'Yours For Ever',
    'WB': 'Welcome Back',
    'WTF': 'What The F...',
    'WTG': 'Way To Go!',
    'WUF': 'Where Are You From?',
    'W8': 'Wait',
}


class TextStandardizeLayer(tf.keras.layers.Layer):
    '''Text standardization layer to clean text before modelling. Only
    english text is supported, output is lower-cased. List of processings applied:
        - fix unicode chars (unidecode lib)
        - replace contractions with full words
        - fix streched letters in words (e.g. soooooo, youuuuuuuu)
        - replace chat language with full phrases (e.g. lol, asap)
        - remove placeholders used in GoEmotions dataset (e.g. [NAME], [RELIGION])
        - remove words containing numbers
        - remove /r tags used in reddit comments (GoEmotions source)
        - remove all charactes except for letters, some punctuation and hyphen
        - replace duplicated punctuation with a single char
        - propper-set punctuation without space before and with 1 space after
        - remove multiple spaces and trim
        - convert to lowercase
    '''
    vect_unidecode = np.vectorize(lambda x: unidecode(x.decode()))

    def __init__(self):
        super(TextStandardizeLayer, self).__init__()

    @staticmethod
    def __unidecode(batch):
        return TextStandardizeLayer.vect_unidecode(batch.numpy())

    @tf.autograph.experimental.do_not_convert
    def call(self, text_t, training=False):
        in_shape = tf.shape(text_t)
        text_t = tf.py_function(TextStandardizeLayer.__unidecode, [text_t],
                                tf.string,
                                name='apply_unidecode')
        for short, full in CONTRACTIONS.items():
            text_t = tf.strings.regex_replace(text_t,
                                              '(?i)' + short,
                                              full,
                                              replace_global=True,
                                              name='fix_contractions')
        for char in ('a', 'e', 'i', 'o', 'u', 'y', 's', 'h', 'f', 'r', 'm'):
            text_t = tf.strings.regex_replace(text_t,
                                              '(?i)' + char + '{3,}',
                                              char,
                                              replace_global=True,
                                              name='fix_streched')
        for chat_word, repl in CHAT_WORDS.items():
            text_t = tf.strings.regex_replace(text_t,
                                              '(?i)[^A-Z]' + chat_word +
                                              '[^A-Z]',
                                              f' {repl} ',
                                              replace_global=True,
                                              name='fix_chat_lang')
        text_t = tf.strings.regex_replace(text_t,
                                          r'(?i)(^|[^\w])r([^\w]|$)',
                                          ' are ',
                                          replace_global=True,
                                          name='fix_chat_lang')
        text_t = tf.strings.regex_replace(text_t,
                                          r'(?i)(^|[^\w])u([^\w]|$)',
                                          ' you ',
                                          replace_global=True,
                                          name='fix_chat_lang')
        text_t = tf.strings.regex_replace(text_t,
                                          r'(^|[^\w])@([^\w]|$)',
                                          ' at ',
                                          replace_global=True,
                                          name='fix_chat_lang')
        text_t = tf.strings.regex_replace(text_t,
                                          r'\[[A-Z]+\]',
                                          ' ',
                                          replace_global=True,
                                          name='remove_placeholders')
        text_t = tf.strings.regex_replace(text_t,
                                          r'[^0-9\s]?[0-9]+[^0-9\s]?',
                                          ' ',
                                          replace_global=True,
                                          name='remove_words_with_nums')
        text_t = tf.strings.regex_replace(text_t,
                                          r'/r',
                                          '',
                                          replace_global=True,
                                          name='remove_bad_chars')
        text_t = tf.strings.regex_replace(text_t,
                                          r"[^A-Za-z,\-\.\!\?\']",
                                          ' ',
                                          replace_global=True,
                                          name='remove_bad_chars')
        for char in (r'\.', ',', '!', r'\?'):
            text_t = tf.strings.regex_replace(text_t,
                                              char + '{2,}',
                                              char.replace('\\', ''),
                                              replace_global=True,
                                              name='fix_punctuation')
            text_t = tf.strings.regex_replace(text_t,
                                              r'\s*' + char + r'\s*',
                                              char.replace('\\', '') + ' ',
                                              replace_global=True,
                                              name='fix_punctuation')
        text_t = tf.strings.regex_replace(text_t,
                                          r'^[\.,!\?\s]+',
                                          '',
                                          replace_global=True,
                                          name='fix_punctuation')
        text_t = tf.strings.regex_replace(text_t,
                                          r'\s+',
                                          ' ',
                                          replace_global=True,
                                          name='fix_spaces')
        text_t = tf.strings.strip(text_t)
        text_t = tf.strings.lower(text_t)
        return tf.reshape(text_t, in_shape)
