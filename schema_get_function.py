# input an event and a sequence of knowledge pieces, output is the schema

from sentence_transformers import SentenceTransformer, util

def remove_repetition(knows, model):
    temp = []
    temp.append(knows[0])
    embeddings1 = model.encode([knows[0]], convert_to_tensor=True)
    for i in knows[1:]:
        embeddings2 = model.encode(i, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        if cosine_scores[0][0] < 0.8:
            temp.append(i)
    return temp


def schema(event, knows):# event：str, knows：list [str, str....]
    model = SentenceTransformer('all-MiniLM-L6-v2')

    filtered_knows = []
    while len(knows) > 1: 
        # remove high slight different knowledges
        knows = remove_repetition(knows, model)
        filtered_knows.append(knows[0])
        knows = knows[1:]
        if len(knows) == 1:
            filtered_knows.append(knows[0])
            break

    # rank to get the schema
    s1 = [event]*len(filtered_knows)
    embeddings1 = model.encode(s1, convert_to_tensor=True)
    embeddings2 = model.encode(filtered_knows, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    pair = []
    for m in range(len(s1)):
        pair.append({'index': [m], 'score': cosine_scores[m][m]})
        pair = sorted(pair, key=lambda x: x['score'], reverse=True)
        schema_list = []
        for p in pair:
            schema_list.append(filtered_knows[p['index'][0]])

    return schema_list # list:[str,str....]

if __name__ == '__main__':
    event = "go to shower"
    knows = ["lie on the bed", "clean my body", "go in bathroom"]
    s = schema(event, knows) #['go in bathroom', 'clean my body', 'lie on the bed']
    print(s)

