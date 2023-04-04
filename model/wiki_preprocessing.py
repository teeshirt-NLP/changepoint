import sys
import string
import re
import pickle
from collections import Counter
import gc

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def count_lines(file_path):
    line_count = 0
    with open(file_path, 'r') as file:
        for _ in file:
            line_count += 1
    return line_count


# From https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences] #Remove punctuation...
    return sentences



def get_counter(batch):
    word_list = [word for paragraph in batch for sentence in paragraph for word in sentence.split()]
    return Counter(word_list)

def process_lines(lines):
    text = False

    for line in lines:
        original_line = line.rstrip()
        if "<text " in line:
            text = True
        if "#redirect" in line.lower():
            text = False
        if text:
            if "</text>" in line:
                text = False

            # Replace blockquote tags with newlines
            line = re.sub("&lt;blockquote&gt;", "\n", line)
            line = re.sub("&lt;/blockquote&gt;", "\n", line)

            # Remove XML tags
            line = re.sub("<.*>", "", line)

            # Decode URL encoded characters
            line = re.sub("&amp;", "&", line)
            line = re.sub("&lt;", "<", line)
            line = re.sub("&gt;", ">", line)

            # Remove references
            line = re.sub("<ref[^<]*<\/ref>", "", line)

            # Remove XHTML tags
            line = re.sub("<[^>]*>", "", line)

            # Remove normal URL, preserve visible text
            line = re.sub("\[http:[^] ]*", "[", line)

            # Remove image links, preserve caption
            line = re.sub("\|thumb", "", line, flags=re.IGNORECASE)
            line = re.sub("\|left", "", line, flags=re.IGNORECASE)
            line = re.sub("\|right", "", line, flags=re.IGNORECASE)
            line = re.sub("\|\d+px", "", line, flags=re.IGNORECASE)
            line = re.sub("\[\[image:[^\[\]]*\|", "", line, flags=re.IGNORECASE)

            # Show categories without markup
            line = re.sub("\[\[category:([^|\]]*)[^]]*\]\]", "[[\\1]]", line, flags=re.IGNORECASE)

            # Remove links to other languages
            line = re.sub("\[\[[a-z\-]*:[^\]]*\]\]", "", line)

            # Remove wiki URL, preserve visible text
            line = re.sub("\[\[[^\|\]]*\|", "[[", line)

            # Remove {{icons}} and {tables}
            line = re.sub("{{[^}]*}}", "", line)
            line = re.sub("{[^}]*}", "", line)

            # Remove [ and ]
            line = re.sub("\[|\]", "", line)

            # Remove URL encoded characters
            line = re.sub("&[^;]*;", " ", line)

            # Convert to lowercase letters and spaces, spell digits
            line = " " + line + " "
            line = line.lower()
            for i, word in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]):
                line = line.replace(str(i), f" {word} ")

            # Change non-alphas to single spaces and keep newlines
            line = re.sub("[^a-z.\n]+", " ", line)

            # Remove extra newlines
            line = re.sub("[\n]+", "\n", line)

            line = line.rstrip()

            yield original_line, line.rstrip() + "\n"
        else:
             yield "", ""



def process_wikipedia_dump(input_file, train_file_prefix, test_file, vocab_file):
    min_words_per_sentence = 4
    min_sentences_per_paragraph = 3
    batch_size = int(5e6)

    batch = []
    batch_counter = 0
    nlines = 0
    is_test_data = True
    wordcounts = Counter()
    number_of_lines = count_lines(input_file)
    print("number_of_lines:", format(number_of_lines, ".3e"))

    with open(input_file, "r", encoding="utf-8") as infile:
        for _, processed_line in process_lines(infile):
            nlines +=1
            if nlines % int(number_of_lines/100) ==0:
                print(str(round(nlines*100 /number_of_lines,2))+"%")
            sentences = split_into_sentences(processed_line.strip())
            valid_sentences = [i for i in sentences if len(i.split(" ")) >= min_words_per_sentence]
            if len(valid_sentences) >= min_sentences_per_paragraph:
                batch.append(valid_sentences)
                batch_counter += 1

                if batch_counter >0 and batch_counter % batch_size ==0:
                    if is_test_data:
                        save_obj(batch, test_file)
                        is_test_data = False
                    else:
                        train_file = f"{train_file_prefix}_{int(batch_counter/batch_size)-1}"
                        save_obj(batch, train_file)
                        wordcounts += get_counter(batch)
                    batch = []
                    gc.collect()

    if batch:
        train_file = f"{train_file_prefix}_{int(batch_counter/batch_size)}"
        save_obj(batch, train_file)
        wordcounts += get_counter(batch)

    save_obj(wordcounts.most_common(), vocab_file)
    gc.collect()

if __name__ == "__main__":
    if len(sys.argv) == 5:
        input_file, train_file_prefix, test_file, vocab_file = sys.argv[1:]
        process_wikipedia_dump(input_file, train_file_prefix, test_file, vocab_file)
    else:
        print("Usage: python combined_script.py <input_file-multistream.xml> <train_file_prefix> <test_file> <vocab_file>")
        print("Download Wikipedia https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia")
        print("Extract it using bzip2 -dk enwiki-YOURDATE-pages-articles-multistream.xml.bz2")

