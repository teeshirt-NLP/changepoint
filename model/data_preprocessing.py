import re
import sys

def process_wikipedia_dump(file):
    text_flag = False
    cleaned_lines = []
    for line in file:
        if "<text " in line:
            text_flag = True
        if re.search(r"#redirect", line, re.IGNORECASE):
            text_flag = False
        if text_flag:
            if "</text>" in line:
                text_flag = False
            
            line = re.sub(r"&lt;blockquote&gt;", "\n", line)
            line = re.sub(r"&lt;\/blockquote&gt;", "\n", line)
            line = re.sub(r"<.*>", "", line)
            line = re.sub(r"&amp;", "&", line)
            line = re.sub(r"&lt;", "<", line)
            line = re.sub(r"&gt;", ">", line)
            line = re.sub(r"<ref[^<]*<\/ref>", "", line)
            line = re.sub(r"<[^>]*>", "", line)
            line = re.sub(r"\[http:[^] ]*", "[", line)
            line = re.sub(r"\|thumb", "", line, flags=re.IGNORECASE)
            line = re.sub(r"\|left", "", line, flags=re.IGNORECASE)
            line = re.sub(r"\|right", "", line, flags=re.IGNORECASE)
            line = re.sub(r"\|\d+px", "", line, flags=re.IGNORECASE)
            line = re.sub(r"\[\[image:[^\[\]]*\|", "", line, flags=re.IGNORECASE)
            line = re.sub(r"\[\[category:([^|\]]*)[^]]*\]\]", "[[\g<1>]]", line, flags=re.IGNORECASE)
            line = re.sub(r"\[\[[a-z\-]*:[^\]]*\]\]", "", line)
            line = re.sub(r"\[\[[^\|\]]*\|", "[[", line)
            line = re.sub(r"{{[^}]*}}", "", line)
            line = re.sub(r"{[^}]*}", "", line)
            line = re.sub(r"\[", "", line)
            line = re.sub(r"\]", "", line)
            line = re.sub(r"&[^;]*;", " ", line)

            line = " " + line + " "
            line = line.lower()
            line = re.sub(r"0", " zero ", line)
            line = re.sub(r"1", " one ", line)

            cleaned_lines.append(line)

    return cleaned_lines

if __name__ == "__main__":
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        cleaned_lines = process_wikipedia_dump(file)
    with open("cleaned_wikipedia_dump.txt", "w", encoding="utf-8") as output_file:
        for line in cleaned_lines:
            output_file.write(line)
