from dash.dependencies import Input, Output, State
import dash_html_components as html
from matscholar import Rester
from numpy import dot
from numpy.linalg import norm
from matscholar import process
from gensim.models import Doc2Vec
import pickle
from pymongo import MongoClient
from matscholar_core.nlp.ner import NERClassifier
from math import log

clf = NERClassifier()
abstract_processor = process.MatScholarProcess()

client = MongoClient('mongodb+srv://kyang:U7B9PNfcaYFppnjN@matstract-kve41.mongodb.net')

#client = MongoClient('mongodb://localhost:27017/')
train_database = client.matstract_db.entities_combined_train

mapping_file = open('issn_to_journal_mapping', 'rb')
mapping = pickle.load(mapping_file)

model = Doc2Vec.load('./d2v_200000_10_iter_100_size_dm0.model')


def bind(app):

    @app.callback(
        Output('similar-journal-table', 'children'),
        [Input('similar-journal-button', 'n_clicks')],
        [State('similar-journal-textarea', 'value')]
    )
    def update_table(n_clicks, text):
        #r = Rester(endpoint="http://0.0.0.0:8080")
        r = Rester()
        journals = r.get_journals(text) # [[[journal cosine]....] [[entities journal]....]]
        #print(journals)
        #journals = make_prediction_cosine(text)
        return html.Table(
            # Header
            [html.Tr([html.Th('Suggested Journals'), html.Th('Cosine Similarity'), html.Th('Suggested Journals'), html.Th('Entities Score')])] +
            # Body
            [html.Tr([html.Td(journals[0][0][0]), html.Td(journals[0][0][1]), html.Td(journals[1][0][1]), html.Td(journals[1][0][0])])] +
            [html.Tr([html.Td(journals[0][1][0]), html.Td(journals[0][1][1]), html.Td(journals[1][1][1]), html.Td(journals[1][1][0])])] +
            [html.Tr([html.Td(journals[0][2][0]), html.Td(journals[0][2][1]), html.Td(journals[1][2][1]), html.Td(journals[1][2][0])])] +
            [html.Tr([html.Td(journals[0][3][0]), html.Td(journals[0][3][1]), html.Td(journals[1][3][1]), html.Td(journals[1][3][0])])] +
            [html.Tr([html.Td(journals[0][4][0]), html.Td(journals[0][4][1]), html.Td(journals[1][4][1]), html.Td(journals[1][4][0])])] +
            [html.Tr([html.Td(journals[0][5][0]), html.Td(journals[0][5][1]), html.Td(journals[1][5][1]), html.Td(journals[1][5][0])])] +
            [html.Tr([html.Td(journals[0][6][0]), html.Td(journals[0][6][1]), html.Td(journals[1][6][1]), html.Td(journals[1][6][0])])] +
            [html.Tr([html.Td(journals[0][7][0]), html.Td(journals[0][7][1]), html.Td(journals[1][7][1]), html.Td(journals[1][7][0])])] +
            [html.Tr([html.Td(journals[0][8][0]), html.Td(journals[0][8][1]), html.Td(journals[1][8][1]), html.Td(journals[1][8][0])])] +
            [html.Tr([html.Td(journals[0][9][0]), html.Td(journals[0][9][1]), html.Td(journals[1][9][1]), html.Td(journals[1][9][0])])]
        )

    '''
    @app.callback(
       Output('similar-journals-table-cosine', 'children'),
       [Input('similar-journal-button', 'n_clicks')],
       [State('similar-journal-textarea', 'value')]
    )
    def update_search_table(n_clicks, text):
        #return journal_suggestion_app.generate_table_cosine(text)
        journals = entities_prediction(text)

        return html.Table(
            # Header
            [html.Tr([html.Th('Suggested Journals'), html.Th('Entities Score')])] +
            # Body
            [html.Tr([html.Td(journals[0][1]), html.Td(journals[0][0])])] +
            [html.Tr([html.Td(journals[1][1]), html.Td(journals[1][0])])] +
            [html.Tr([html.Td(journals[2][1]), html.Td(journals[2][0])])] +
            [html.Tr([html.Td(journals[3][1]), html.Td(journals[3][0])])] +
            [html.Tr([html.Td(journals[4][1]), html.Td(journals[4][0])])] +
            [html.Tr([html.Td(journals[5][1]), html.Td(journals[5][0])])] +
            [html.Tr([html.Td(journals[6][1]), html.Td(journals[6][0])])] +
            [html.Tr([html.Td(journals[7][1]), html.Td(journals[7][0])])] +
            [html.Tr([html.Td(journals[8][1]), html.Td(journals[8][0])])] +
            [html.Tr([html.Td(journals[9][1]), html.Td(journals[9][0])])]
        )



def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def process_abstract(abstract, model):
    #abstract_processor = process.MatScholarProcess()

    tokenized_abstract = abstract_processor.tokenize(abstract)

    combined_abstract = []
    for array in tokenized_abstract:
        combined_abstract.extend(array)

    processed_abstract = abstract_processor.process(combined_abstract)

    vectorized_abstract = model.infer_vector(processed_abstract[0], steps=20)
    return vectorized_abstract


def make_prediction_cosine(abstract):
    #model = Doc2Vec.load('./d2v_200000_10_iter_25_size_dm0.model')

    #file_mapping = open('./issn_to_journal_mapping', 'rb')
    #mapping = pickle.load(file_mapping)

    vectorized_abstract = process_abstract(abstract, model)

    docvecs_list = []
    for i in range(len(model.docvecs)):  # if i don't do this it bugs out on index 810
        docvecs_list.append(model.docvecs[i])

    docvecs_and_doctags = zip(docvecs_list, model.docvecs.doctags)

    similarities = []
    for pair in docvecs_and_doctags:
        cosine_sim = str(cosine_similarity(vectorized_abstract, pair[0]))
        similarities.append([cosine_sim, pair[1]])

    similarities.sort()
    similarities.reverse()

    top_10_journals = [[mapping[pair[1]], pair[0]] for pair in similarities[:10]]

    return top_10_journals


def entities_score(document, issn_document):
    # the log is for imbalanced data, journals with fewer documents are mulitplied by a larger number
    # journals with more documents are mulitplied by a smaller number.
    # score for each journal determined by entity appearance in document *
    # entity appearance in journal * log regularization

    score = 0
    for entity in document:
        if entity in issn_document['entities']:
            #score += document[entity] * issn_document['entities'][entity] * (1 / issn_document['num_docs'])
            score += document[entity] * issn_document['entities'][entity] * (1 / log(issn_document['num_docs'], 10))
    return score


def identity(vals):
    return vals[0]


def entities_prediction(abstract):
    #client = MongoClient('mongodb://localhost:27017/')
    #train_database = client.matstract_db.entities_combined_train

    #mapping_file = open('issn_to_journal_mapping', 'rb')
    #mapping = pickle.load(mapping_file)

    #clf = NERClassifier()

    entities_filtered = {}
    entities_list = clf.as_normalized([abstract])[0]
    for sentence in entities_list:
        for entity in sentence:
            if entity[1] != 'O':
                if entity[0] in entities_filtered:
                    entities_filtered[entity[0]] += 1
                else:
                    entities_filtered[entity[0]] = 1

    all_journals = train_database.find(no_cursor_timeout=True)

    scores = []
    for journal in all_journals:
        score = entities_score(entities_filtered, journal)
        scores.append([score, mapping[journal['ISSN']]])

    scores.sort(key=identity, reverse=True)

    top_10 = scores[:10]

    return top_10
    
    '''


