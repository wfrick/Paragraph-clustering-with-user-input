import codecs
from bs4 import BeautifulSoup
from embedly import Embedly
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn import preprocessing
import re
import requests
import json

from key import key
from key import access_token


def embedly_text(url, key):
    #Takes a URL, returns list of text, text w/o line breaks, content type (article),
    #clean version of the URL, headline, and the source of the article

    client = Embedly(key)

    try:
        response = client.extract(url)
        headline = response['title']
        source = response['provider_name']
        clean_url = response['url']
        content = response['content']

        soup = BeautifulSoup(content)
        text = soup.get_text()
        text = text.encode('utf-8')
        clean_text = text.replace('\n', ' ')

        row = [text, clean_text, "Article", clean_url, headline, source]
    except:
        row = ["Error", "Error", "Article", url, "Error", "Error"]
    return row

def create_dataframe(url_list):
    #Takes a list of URLs and returns dataframe of text and other key info

    #Create a dataframe
    df = pd.DataFrame(columns = ["Text", "Clean text", "Type", "URL", "Headline", "Source"])

    #For each URL, hit Embedly, get content + info, add to dataframe
    count = 0
    print "Parsing articles:"
    for a in url_list:
        df.loc[len(df)] = embedly_text(a, key)
        count = count + 1
        print count

    print "The full dataframe has the following shape:"
    print df.shape
    #Remove Errors, print shape
    print "Removing errors..."
    articles = df.loc[df['Text'] != 'Error']
    print articles.shape

    return articles
    
    
def remove_stop_words(tokens):
    #Takes a list of words, remove common ones
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w for w in tokens if w.lower() not in stopwords]
    return content

def lowercase_tokens(tokens):
    #Takes a list of words, lowercases them
    list_of_words = []
    for a in tokens:
        a = a.lower()
        list_of_words.append(a)
    return list_of_words


def top_grams(articles, text_field, n_top_words):
    #Takes a dataframe of articles and the number of top grams you want returned,
    #returns top uni, bi, and trigrams
    top_ngrams = []
    
    #Combine all text from 'Text' column into single string, print its length
    text = articles[text_field].str.cat(sep='; ')
    print len(text)

    #Tokenize text, print most common words
    #Create tokenizer to remove punctuation and numbers
    tokenizer = RegexpTokenizer(r'\w+')

    #Tokenize, lowercase, remove stop words
    tokens = tokenizer.tokenize(text)
    tokens = lowercase_tokens(tokens)
    tokens = remove_stop_words(tokens)

    #Find common words
    fdist = FreqDist(tokens)
    top_uni = fdist.most_common(n_top_words)
    top_ngrams.append(top_uni)
    #print "Most common words:"
    #print top_uni

    #Find top bigrams
    bigrams = nltk.bigrams(tokens)
    bigram_fdist = FreqDist(bigrams)
    top_bi = bigram_fdist.most_common(n_top_words)
    top_ngrams.append(top_bi)
    #print "Most common bigrams"
    #print top_bi

    #Find top trigrams
    trigrams = nltk.trigrams(tokens)
    trigram_fdist = FreqDist(trigrams)
    top_tri = trigram_fdist.most_common(n_top_words)
    top_ngrams.append(top_tri)
    #print "Most common trigrams"
    #print top_tri

    return top_ngrams

def subset_ngrams(df, text_field, key_term, n_terms):
    #Takes a key term and returns top grams for rows in dataframe that contain that term
    subset = df[df[text_field].str.contains(key_term)]
    print subset.shape
    print top_grams(subset,text_field,n_terms)

def create_user_defined_topics(df, text_field):
    #Takes a dataframe, appends counts of key topics created by the user

    user_topics = []
    #Step 1: Ask user for key terms
    
    #Show user top n-grams
    print top_grams(df,text_field,20)
    print '\n'

    #Take user input of key ngrams
    print 'Enter any key terms you think are relevant. Begin and end with quotes " " and separate your terms with a comma and a space.'              
    terms_list = input('Example: "economic growth, productivity, globalization" \n\n')
    terms_list = terms_list.split(', ')
    print "You entered: ",terms_list

    for a in terms_list:
        print "You indicated that '" + str(a) + "' is important. \n"
        print "What are some other terms that are related to it? \n"
        print "Here are some terms that come up when '" + str(a) + "' is mentioned:\n"

        subset_ngrams(df,text_field,str(a),15)  
        related_term_list = input('Enter any key terms you think are related to' + str(a) + '. The same rules about quotes and spacing apply.\n\n\n')
        related_term_list = related_term_list = related_term_list.split(', ')        

        related_term_list.append(a)
        user_topics.append(related_term_list)

    return user_topics

def topic_model_top_words(model, feature_names, n_top_words):
    #Takes topic model, prints the top words in each topic
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def lda_assignments(articles, text_field, n_features, n_topics, n_top_words):
    #Takes a dataframe of articles, the name of the text field in the dataframe,
    #a number of features, topics, and top words to print
    #Fits LDA model, prints top terms for each topic
    #Returns article dataframe with topic assignemnts

    text = articles[text_field]

    # Use tf-idf features for topic model.
    #print("Extracting tf-idf features...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                   stop_words='english')
    
    tfidf = tfidf_vectorizer.fit_transform(text)

    #print("Fitting LDA models with tf features, n_samples and n_features=%d..."
      #% (n_features))
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)

    lda.fit(tfidf)
    
    #print "\nTopics in LDA model:"
    #tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    #topic_model_top_words(lda, tfidf_feature_names, n_top_words)

    #Add topic assignments to articles dataframe
    topic_assignments = lda.transform(tfidf)
    assignments = np.transpose(topic_assignments)
    count = 0
    while (count) < n_topics:
        articles["Topic " + str(count + 1)] = assignments[count]
        count += 1

    return articles

def top_articles_per_topic(articles, n_topics, field_to_return):
    #Takes a dataframe of articles with topic assignments, prints top in each topic
    #Also takes the name of the field you want back -- "Headlines" or "Paragraph" for instance
    count = 0
    while count < n_topics:
        print "Top articles in topic " + str(count + 1)
        column = "Topic " + str(count + 1)
        sorted_articles = articles.sort_values(column, ascending=False)
        print sorted_articles[field_to_return][0:10]
        count += 1

def article_to_paragraphs(text):
    #Breaks article text into paragraphs
    paragraphs = text.split("\n")
    paragraphs = [value for value in paragraphs if value != '']
    paragraphs = [value for value in paragraphs if value != '\r']

    return paragraphs
    

def create_paragraph_dataframe(articles, n_topics):
    #Takes dataframe of articles, topic number, returns df at paragraph level
    #Paragraphs inherit information and topic assignments from article
    
    #Create list column header list for paragraph dataframe
    column_headers = ["Paragraph", "Type", "URL", "Headline", "Source",
                      "CalaisTopics", "openCalais Cluster"]
    count = 0
    while count < n_topics:
        header = "Article Topic " + str(count + 1)
        column_headers.append(header)
        count +=1

    #Create new dataframe
    graf_df = pd.DataFrame(columns = column_headers)
    
    #For each article, split into paragraphs, each graf inherits from article
    count = -1
    for a in articles['Text']:
        count += 1
        text = a
        #Split article text into list of paragraph by line break
        grafs = article_to_paragraphs(text)

        #For each graf, create row with text plus inherited info from article
        for b in grafs:
            try:
                #Only add paragraphs with more than one sentence
                sents = nltk.sent_tokenize(b)
                if len(sents) > 1:
                    
                    paragraph = b.replace('\n', ' ')
                    paragraph = paragraph.replace('\r','')

                    row = [paragraph, "Paragraph", articles.loc[count]['URL'],
                           articles.loc[count]['Headline'], articles.loc[count]['Source'],
                           articles.loc[count]['CalaisTopics'], articles.loc[count]['openCalais Cluster']]

                    #Inherit topic assignments from article
                    topics = column_headers[7:]
                    z = 0
                    while z < len(topics):
                        topic_label = "Topic " + str(z+1)
                        row.append(articles.loc[count][topic_label])
                        z+=1

                    #Append paragraph info to dataframe
                    row_df = pd.DataFrame([row],columns=column_headers)
                    graf_df = graf_df.append(row_df)
            except:
                #print "Exception raised for location:"
                #print count
                continue

    print "Paragraph dataframe shape:"
    print graf_df.shape

    return graf_df


def count_terms(articles, text_column, terms_list):
    #Takes a dataframe, the name of the column with text, and a list of lists
    #Each item in the list is a list of related terms
    for terms in terms_list:
        counts = []
        for text in articles[text_column]:
            text = text.lower()
            count = 0
            for term in terms:
                count = count + text.count(term)
            counts.append(count)
            
        articles[terms[len(terms)-1]] = counts
    return articles

def k_means_cluster(articles,n_topics, name_of_cluster_result_column):
    #Takes a dataframe, clusters on all numeric data with k-means
    
    # Initialize the model with 2 parameters -- number of clusters and random state.
    kmeans_model = KMeans(n_clusters=n_topics, random_state=1)
    # Get only the numeric columns from games.
    good_columns = articles._get_numeric_data()

    #Normalize data
    normalized = preprocessing.normalize(good_columns)
    normalized = pd.DataFrame(normalized)
    #Note: lost column headers here.

    # Fit the model using the good columns.
    kmeans_model.fit(normalized)
    # Get the cluster assignments.
    labels = kmeans_model.labels_

    #Print inertia measure
    print "K-means inertia measure:"
    print kmeans_model.inertia_

    #Assign labels
    articles[name_of_cluster_result_column] = labels

    return articles
    
def describe_clusters(articles,n_topics):
    #Takes a dataframe of documents and topic number
    #prints the shape of each cluster and top ngrams

    count = 0
    while count < n_topics:
        cluster = articles.loc[(articles['Cluster'] == count)]
        print "Cluster " + str(count+1) + ":"
        print cluster.shape
        top_grams(cluster, "Paragraph", 10)
        count +=1

def summarize_clusters(articles,text_field, n_topics):
    #Takes a dataframe of documents, a field with text, and topic number
    #Returns distinctive words from that cluster.
    #Calculation: Of top uni, bi, trigrams which have highest in-topic count / corpus count

    #Get top ngrams for corpus as dictionary
    corpus_grams = top_grams(articles, text_field, 1000)
    corpus_dict = {}
    for a in corpus_grams:
        for b in a:
            key = re.sub(r'[^a-zA-Z0-9 ]+','',str(b[0]))
            corpus_dict[key] = b[1]
            
    #Now for each topic:
    #Get the top ngrams. For each, calculate count in topic / count in corpus_grams
    #Display the top X

    count = 0
    cluster_summary = {}
    while count < n_topics:
        cluster_label = "Cluster" + str(count)
        cluster = articles.loc[(articles['Cluster'] == count)]
        print "Cluster " + str(count) + " shape:"
        print cluster.shape
        
        topic_grams = top_grams(cluster,text_field,20)
        topic_list = []
        for a in topic_grams:
            for b in a:
                term = re.sub(r'[^a-zA-Z0-9 ]+','',str(b[0]))
                term_count = b[1]
                try:
                    corpus_count = corpus_dict[term]
                    adjusted_count = 100 * term_count / corpus_count
                    topic_list.append([term, adjusted_count])
                except:
                    continue
        cluster_summary[count] = sorted(topic_list, key=lambda term: term[1],reverse=True)
        print "Cluster " + str(count) + " summary:"
        print cluster_summary[count][0:10]
        count += 1

    return cluster_summary

def sample_from_clusters(articles,n_topics,n_samples):
    #Returns a random sample of paragraphs from each cluster
    count = 0
    sample_grafs = []
    while count < n_topics:
        cluster_label = "Cluster" + str(count)
        cluster = articles.loc[(articles['Cluster'] == count)]
        if cluster.shape[0] >= n_samples:
            sample = cluster.sample(n_samples, replace=False)
        else:
            sample = cluster.sample(cluster.shape[0], replace=True)
        print sample.shape
        print str(type(sample))
        sample = sample['Paragraph']
        topic_grafs = []
        for a in sample:
            topic_grafs.append(a)
        sample_grafs.append(topic_grafs)
        count += 1
    return sample_grafs
        
def wrapStringInHTML(program, topic_number, list_of_text):
    #For testing only -- quick way to review topic model output
    import datetime
    from webbrowser import open_new_tab

    now = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")

    filename = program + '.html'
    f = open(filename,'w')

    wrapper = """<html>
    <head>
    <title>%s output - %s</title>
    </head>
    <body>
    <p>Topic %s</a></p>
    <p>%s</p>
    <p>%s</p>
    <p>%s</p>
    <p>%s</p>
    <p>%s</p>
    <p>%s</p>
    <p>%s</p>
    <p>%s</p>
    <p>%s</p>
    <p>%s</p>
    </body>
    </html>"""

    
    whole = wrapper % (program, now, str(topic_number),
                       str(list_of_text[0]),str(list_of_text[1]),
                       str(list_of_text[2]), str(list_of_text[3]),
                       str(list_of_text[4]), str(list_of_text[5]),
                       str(list_of_text[6]),str(list_of_text[7]),
                       str(list_of_text[8]),str(list_of_text[9]))
    f.write(whole)
    f.close()

    open_new_tab(filename)


def wrapStringInBootstrap(program, sample_from_clusters):
    #For testing only
    #Populates Bootstrap mockup with real topic text
    import datetime
    from webbrowser import open_new_tab

    now = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")

    filename = program + '.html'
    f = open(filename,'w')

    wrapper = bootstrap_sandbox
    whole = wrapper % (sample_from_clusters[0][0],
                       sample_from_clusters[1][0],
                       sample_from_clusters[2][0],
                       sample_from_clusters[3][0])

    f.write(whole)
    f.close()

    open_new_tab(filename)

def openCalais(access_token, text):
    #Takes text, saves it to a file, then sens file to openCalais API to get topics

    #Save text to file
    text_file = open("openCalais.txt", "w")
    text_file.write(text)
    text_file.close()

    #Text file to API
    headers = {'X-AG-Access-Token' : access_token, 'Content-Type' : 'text/raw',
               'outputformat' : 'application/json', 'omitOutputtingOriginalText' : 'True',
               'x-calais-language' : 'English'}
    calais_url = 'https://api.thomsonreuters.com/permid/calais'
    file_name = "openCalais.txt"
    files = {'file': open(file_name, 'rb')}
    response = requests.post(calais_url, files=files, headers=headers, timeout=80)
    response = response.text
    response = json.loads(response)

    #Get social tags
    social_tags = []
    for a in response.keys():
        if "SocialTag" in a:
            social_tags.append(a)

    topics = []
    for b in social_tags:
        name = str(response[b]['name'])
        importance = int(response[b]['importance'])
        #Flip importance values (1 is high, 3 is low in API; reverse that)
        if importance == 3:
            importance = 1
        elif importance == 1:
            importance = 3
        topics.append([name, importance])

    return topics

def save_calais_topics(dataframe, access_token, text_field, n_topics):

    #All topics creates a list of every unique topic across the corpus
    #and how many articles it's assigned to
    all_topics = {}

    #Topic assignments is a list of API results that
    #eventually will become the 'CalaisTopics' column in the df
    topic_assignments = []


    for a in dataframe[text_field]:
        try:
            topics = openCalais(access_token, a)
            print "API working"
            topic_assignments.append(topics)
            for b in topics:
                if b[0] not in all_topics:
                    all_topics[b[0]] = 1
                else:
                    all_topics[b[0]] += 1
        except:
            print "Error connecting to openCalais API"
            topic_assignments.append([])
            continue

    dataframe['CalaisTopics'] = topic_assignments

    copy_of_dataframe = dataframe.copy()

    for z in all_topics:
        topic_column = []
        for y in copy_of_dataframe['CalaisTopics']:
            value = 0
            for x in y:
                if z == x[0]:
                    value = x[1]
            topic_column.append(value)
        copy_of_dataframe[z] = topic_column

    copy_of_dataframe = k_means_cluster(copy_of_dataframe, n_topics, 'openCalais Cluster')

    dataframe['openCalais Cluster'] = copy_of_dataframe['openCalais Cluster']
    
    return dataframe, all_topics


bootstrap_sandbox = '''<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
    <meta name="description" content="">
    <meta name="author" content="">
    <link rel="icon" href="../../favicon.ico">

    <title>Carousel Template for Bootstrap</title>

    <!-- Bootstrap core CSS -->
    <link href="css/bootstrap.min.css" rel="stylesheet">

    <!-- IE10 viewport hack for Surface/desktop Windows 8 bug -->
    <link href="css/ie10-viewport-bug-workaround.css" rel="stylesheet">

    <!-- Just for debugging purposes. Don't actually copy these 2 lines! -->
    <!--[if lt IE 9]><script src="../../assets/js/ie8-responsive-file-warning.js"></script><![endif]-->
    <script src="js/ie-emulation-modes-warning.js"></script>

    <!-- HTML5 shim and Respond.js for IE8 support of HTML5 elements and media queries -->
    <!--[if lt IE 9]>
      <script src="https://oss.maxcdn.com/html5shiv/3.7.3/html5shiv.min.js"></script>
      <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
    <![endif]-->

    <!-- Custom styles for this template -->
    <link href="css/carousel.css" rel="stylesheet">
  </head>
<!-- NAVBAR
================================================== -->
  <body>
    <div class="navbar-wrapper">
      <div class="container">

        <nav class="navbar navbar-inverse navbar-static-top">
          <div class="container">
            <div class="navbar-header">
              <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
                <span class="sr-only">Toggle navigation</span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
              </button>
              <a class="navbar-brand" href="#">Up to Speed on: Inequality</a>
            </div>
            <div id="navbar" class="navbar-collapse collapse">
              <ul class="nav navbar-nav">
                <li class="active"><a href="#">Home</a></li>
                <li><a href="#about">Explore a Topic</a></li>
                <li><a href="#contact">About</a></li>
              </ul>
            </div>
          </div>
        </nav>

      </div>
    </div>


    <!-- Carousel 1
    ================================================== -->
    <div id="myCarousel" class="carousel slide" data-ride="carousel">
      <!-- Indicators -->
      <ol class="carousel-indicators">
        <li data-target="#myCarousel" data-slide-to="0" class="active"></li>
        <li data-target="#myCarousel" data-slide-to="1"></li>
        <li data-target="#myCarousel" data-slide-to="2"></li>
      </ol>
      <div class="carousel-inner" role="listbox">
        <div class="item active">
          <img class="first-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="First slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>Topic 1: </h1>
              <p>%s</p>
              <p>
                <a class="btn btn-lg btn-primary" href="#" role="button"><span class="glyphicon glyphicon-star"></span></a>
                <a class="btn btn-lg btn-primary" href="#" role="button"><span class="glyphicon glyphicon-remove"></span></a>
              </p>
            </div>
          </div>
        </div>
        <div class="item">
          <img class="second-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Second slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>Another example headline.</h1>
              <p>Cras justo odio, dapibus ac facilisis in, egestas eget quam. Donec id elit non mi porta gravida at eget metus. Nullam id dolor id nibh ultricies vehicula ut id elit.</p>
              <p><a class="btn btn-lg btn-primary" href="#" role="button">Learn more</a></p>
            </div>
          </div>
        </div>
        <div class="item">
          <img class="third-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Third slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>One more for good measure.</h1>
              <p>Cras justo odio, dapibus ac facilisis in, egestas eget quam. Donec id elit non mi porta gravida at eget metus. Nullam id dolor id nibh ultricies vehicula ut id elit.</p>
              <p><a class="btn btn-lg btn-primary" href="#" role="button">Browse gallery</a></p>
            </div>
          </div>
        </div>
      </div>
      <a class="left carousel-control" href="#myCarousel" role="button" data-slide="prev">
        <span class="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
        <span class="sr-only">Previous</span>
      </a>
      <a class="right carousel-control" href="#myCarousel" role="button" data-slide="next">
        <span class="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
        <span class="sr-only">Next</span>
      </a>
    </div><!-- /.carousel 1-->


    <!-- Carousel 2
    ================================================== -->
    <div id="myCarousel" class="carousel slide" data-ride="carousel">
      <!-- Indicators -->
      <ol class="carousel-indicators">
        <li data-target="#myCarousel" data-slide-to="0" class="active"></li>
        <li data-target="#myCarousel" data-slide-to="1"></li>
        <li data-target="#myCarousel" data-slide-to="2"></li>
      </ol>
      <div class="carousel-inner" role="listbox">
        <div class="item active">
          <img class="first-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="First slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>Topic 2:</h1>
              <p>%s</p>
              <p>
                <a class="btn btn-lg btn-primary" href="#" role="button"><span class="glyphicon glyphicon-star"></span></a>
                <a class="btn btn-lg btn-primary" href="#" role="button"><span class="glyphicon glyphicon-remove"></span></a>
              </p>
            </div>
          </div>
        </div>
        <div class="item">
          <img class="second-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Second slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>Another example headline.</h1>
              <p>Cras justo odio, dapibus ac facilisis in, egestas eget quam. Donec id elit non mi porta gravida at eget metus. Nullam id dolor id nibh ultricies vehicula ut id elit.</p>
              <p><a class="btn btn-lg btn-primary" href="#" role="button">Learn more</a></p>
            </div>
          </div>
        </div>
        <div class="item">
          <img class="third-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Third slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>One more for good measure.</h1>
              <p>Cras justo odio, dapibus ac facilisis in, egestas eget quam. Donec id elit non mi porta gravida at eget metus. Nullam id dolor id nibh ultricies vehicula ut id elit.</p>
              <p><a class="btn btn-lg btn-primary" href="#" role="button">Browse gallery</a></p>
            </div>
          </div>
        </div>
      </div>
      <a class="left carousel-control" href="#myCarousel" role="button" data-slide="prev">
        <span class="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
        <span class="sr-only">Previous</span>
      </a>
      <a class="right carousel-control" href="#myCarousel" role="button" data-slide="next">
        <span class="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
        <span class="sr-only">Next</span>
      </a>
    </div><!-- /.carousel 2 -->


<!-- Carousel 3
    ================================================== -->
    <div id="myCarousel" class="carousel slide" data-ride="carousel">
      <!-- Indicators -->
      <ol class="carousel-indicators">
        <li data-target="#myCarousel" data-slide-to="0" class="active"></li>
        <li data-target="#myCarousel" data-slide-to="1"></li>
        <li data-target="#myCarousel" data-slide-to="2"></li>
      </ol>
      <div class="carousel-inner" role="listbox">
        <div class="item active">
          <img class="first-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="First slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>Topic 3:</h1>
              <p>%s</p>
              <p>
                <a class="btn btn-lg btn-primary" href="#" role="button"><span class="glyphicon glyphicon-star"></span></a>
                <a class="btn btn-lg btn-primary" href="#" role="button"><span class="glyphicon glyphicon-remove"></span></a>
              </p>
            </div>
          </div>
        </div>
        <div class="item">
          <img class="second-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Second slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>Another example headline.</h1>
              <p>Cras justo odio, dapibus ac facilisis in, egestas eget quam. Donec id elit non mi porta gravida at eget metus. Nullam id dolor id nibh ultricies vehicula ut id elit.</p>
              <p><a class="btn btn-lg btn-primary" href="#" role="button">Learn more</a></p>
            </div>
          </div>
        </div>
        <div class="item">
          <img class="third-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Third slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>One more for good measure.</h1>
              <p>Cras justo odio, dapibus ac facilisis in, egestas eget quam. Donec id elit non mi porta gravida at eget metus. Nullam id dolor id nibh ultricies vehicula ut id elit.</p>
              <p><a class="btn btn-lg btn-primary" href="#" role="button">Browse gallery</a></p>
            </div>
          </div>
        </div>
      </div>
      <a class="left carousel-control" href="#myCarousel" role="button" data-slide="prev">
        <span class="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
        <span class="sr-only">Previous</span>
      </a>
      <a class="right carousel-control" href="#myCarousel" role="button" data-slide="next">
        <span class="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
        <span class="sr-only">Next</span>
      </a>
    </div><!-- /.carousel 3 -->


<!-- Carousel 4
    ================================================== -->
    <div id="myCarousel" class="carousel slide" data-ride="carousel">
      <!-- Indicators -->
      <ol class="carousel-indicators">
        <li data-target="#myCarousel" data-slide-to="0" class="active"></li>
        <li data-target="#myCarousel" data-slide-to="1"></li>
        <li data-target="#myCarousel" data-slide-to="2"></li>
      </ol>
      <div class="carousel-inner" role="listbox">
        <div class="item active">
          <img class="first-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="First slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>Topic 4:</h1>
              <p>%s</p>
              <p>
                <a class="btn btn-lg btn-primary" href="#" role="button"><span class="glyphicon glyphicon-star"></span></a>
                <a class="btn btn-lg btn-primary" href="#" role="button"><span class="glyphicon glyphicon-remove"></span></a>
              </p>
            </div>
          </div>
        </div>
        <div class="item">
          <img class="second-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Second slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>Another example headline.</h1>
              <p>Cras justo odio, dapibus ac facilisis in, egestas eget quam. Donec id elit non mi porta gravida at eget metus. Nullam id dolor id nibh ultricies vehicula ut id elit.</p>
              <p><a class="btn btn-lg btn-primary" href="#" role="button">Learn more</a></p>
            </div>
          </div>
        </div>
        <div class="item">
          <img class="third-slide" src="data:image/gif;base64,R0lGODlhAQABAIAAAHd3dwAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Third slide">
          <div class="container">
            <div class="carousel-caption">
              <h1>One more for good measure.</h1>
              <p>Cras justo odio, dapibus ac facilisis in, egestas eget quam. Donec id elit non mi porta gravida at eget metus. Nullam id dolor id nibh ultricies vehicula ut id elit.</p>
              <p><a class="btn btn-lg btn-primary" href="#" role="button">Browse gallery</a></p>
            </div>
          </div>
        </div>
      </div>
      <a class="left carousel-control" href="#myCarousel" role="button" data-slide="prev">
        <span class="glyphicon glyphicon-chevron-left" aria-hidden="true"></span>
        <span class="sr-only">Previous</span>
      </a>
      <a class="right carousel-control" href="#myCarousel" role="button" data-slide="next">
        <span class="glyphicon glyphicon-chevron-right" aria-hidden="true"></span>
        <span class="sr-only">Next</span>
      </a>
    </div><!-- /.carousel 4 -->



      <!-- FOOTER -->
      <footer>
        <p class="pull-right"><a href="#">Back to top</a></p>
        <p>&copy; 2016 Company, Inc. &middot; <a href="#">Privacy</a> &middot; <a href="#">Terms</a></p>
      </footer>

    </div><!-- /.container -->


    <!-- Bootstrap core JavaScript
    ================================================== -->
    <!-- Placed at the end of the document so the pages load faster -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.12.4/jquery.min.js"></script>
    <script>window.jQuery || document.write('<script src="../../assets/js/vendor/jquery.min.js"><\/script>')</script>
    <script src="../../dist/js/bootstrap.min.js"></script>
    <!-- Just to make our placeholder images work. Don't actually copy the next line! -->
    <script src="../../assets/js/vendor/holder.min.js"></script>
    <!-- IE10 viewport hack for Surface/desktop Windows 8 bug -->
    <script src="../../assets/js/ie10-viewport-bug-workaround.js"></script>
  </body>'''
