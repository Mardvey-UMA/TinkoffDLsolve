import ast, re, pathlib, shutil, pathlib
import numpy as np
import argparse

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def remove_comments(source): #Функция для удаления однострочных комментариев 
    string = re.sub(re.compile("'''.*?'''", re.DOTALL), "", source)  
    string = re.sub(re.compile('""".*?"""', re.DOTALL), "", source)  
    string = re.sub(re.compile("(?<!(['\"]).)#[^\n]*?\n"), "\n", string) 
    return string 

def make_clean_ast(pyfile_path1, pyfile_path2):
    """
    Аргументы: Абсолютный путь до первого и второго Python кода соотвественно
    Возвращает: Отформатированное абстрактное синтаксическое дерево, для дальнейшего сравнения
    """
    signs = [']','[','(',')',',',"'"] 
    #Происходит копирование входных файлов, для последующей генерации из них текстовых файлов
    new_name1 = pyfile_path1[:-3] + 'copy1' + '.py'
    new_name2 = pyfile_path2[:-3] + 'copy2' + '.py'

    shutil.copyfile(pyfile_path1, new_name1)
    shutil.copyfile(pyfile_path2, new_name2)

    new_name1 = pathlib.Path(new_name1)
    new_name2 = pathlib.Path(new_name2)
    
    new_path = str([part + '/' for part in list(str(new_name1).split('/')[:-1])]).replace(']','').replace('[','').replace(',','').replace("'",'').replace(' ','')
    new_name1.rename(new_path + 'copy1' + '.txt')
    copy1 = new_path + 'copy1' + '.txt'
    new_path = str([part + '/' for part in list(str(new_name2).split('/')[:-1])]).replace(']','').replace('[','').replace(',','').replace("'",'').replace(' ','')
    new_name2.rename(new_path + 'copy2' + '.txt')
    copy2 = new_path + 'copy2' + '.txt'
    #Конец конвертации файлов

    f = open(copy1) #Открытие текстового файла с исходным кодом
    code1 = ""
    for line in f:
        code1+=line
    code1 = remove_comments(code1) #Удаление комментариев
    tree1 = ast.parse(source = code1) #Построение абстрактного синтаксического дерева с помощью библиотеки AST
    ast_code1 = ast.dump(tree1, annotate_fields=False) #Конвертация дерева, в строку, аннотации удаляются, за ненадобностью
    #Нормализация дерева, удаление скобок кавычек и лишних пробелов
    for g in signs:
        ast_code1 = ast_code1.replace(g,' ')
    ast_code1 = re.sub(r" [\d+] ", "", ast_code1)
    ast_code1 = re.sub(" +", " ", ast_code1)
    # Переменная ast_code* хранит строку в ввиде нормализованного синтаксического дерева
    # Над вторым файлом производятся аналогичные манипуляции
    f = open(copy2)
    code2 = ""
    for line in f:
        code2+=line   
    code2 = remove_comments(code2)
    tree2 = ast.parse(source = code2)
    ast_code2 = ast.dump(tree2, annotate_fields=False)
    for g in signs:
        ast_code2 = ast_code2.replace(g,' ')
    ast_code2 = re.sub(r" [\d+] ", "", ast_code2)
    ast_code2 = re.sub(" +", " ", ast_code2)
    
    
    return ast_code1, ast_code2


parser = argparse.ArgumentParser()
parser.add_argument('indir', type=str)
parser.add_argument('outdir', type=str)
args = parser.parse_args()
input_path = args.indir
output_path = args.outdir
#Открытие файлов для записи и чтения
source = open(input_path, 'r')
result = open(output_path, 'w')

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')


for line in source:
    path1, path2 = list(map(str, line.split()))
    code_ast1, code_ast2 = make_clean_ast(path1, path2)
    tokens = tokenizer([code_ast1, code_ast2],
                          max_length=128,
                          truncation=True,
                          padding='max_length',
                          return_tensors='pt')
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    counted = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / counted
    mean_pooled = mean_pooled.detach().numpy()
    scores = np.zeros((mean_pooled.shape[0], mean_pooled.shape[0]))
    for i in range(mean_pooled.shape[0]):
        scores[i, :] = cosine_similarity(
            [mean_pooled[i]],
            mean_pooled
        )[0]
    result.write(str(scores[0][1]) + '\n')
