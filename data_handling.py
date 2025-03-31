def load_tinystories_data(path):
    texts = []
    with open(path, 'r', encoding='utf8') as file:
        lines = ''
        for line in file.readlines():
            if line == '<|endoftext|>\n':
                texts.append(lines)
                lines = ''
                continue
            lines += line.replace('\n', ' ')

    return texts