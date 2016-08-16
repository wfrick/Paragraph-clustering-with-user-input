from utilities import *
import pandas as pd

#Pick your topic
topic = ''
while topic != "minimum wage" and topic != "inequality":
    topic = input("Pick your topic: Type 'minimum wage' or 'inequality'\n")


#Number of clusters throughout process
n_topics = ''
while type(n_topics) != int:
    n_topics = input("How many topics do want? (Must be an integer.)\n")


#Get URLs

urls = pd.read_csv(str(topic) + " urls.csv")
url_list = list(urls['URL'])

#Get info for each url, create dataframe
articles = create_dataframe(url_list)
print articles.head()
print articles.shape
#Save just article text to CSV
articles.to_csv(str(topic) + "_articles_no_topics.csv")

#Get article topics from openCalais API
results = save_calais_topics(articles, access_token, 'Clean text', n_topics)
with_calais_topics = results[0]
with_calais_topics.to_csv(str(topic) + "_articles_with_openCalais_topics.csv")

#Print list of openCalais article topics
all_topics = results[1]
print all_topics

'''
#REMOVE LATER: this is just to not hit the Embedly and openCalais API every time:

#Read in dataframe of text and topics
articles = pd.read_csv("Min_wage_articles_with_openCalais_topics.csv")
articles = articles.drop(articles.columns[[0]], axis=1)

## END REMOVE LATER
'''

#LDA at article level, save topic assignments as dataframe and csv
with_assignment = lda_assignments(articles, 'Clean text', 200, n_topics, 10)
with_assignment.to_csv(str(topic) + "_Articles_with_topics.csv")
print with_assignment.columns


#Take article dataframe with assignments, parse to paragraphs save as df + csv
grafs = create_paragraph_dataframe(with_assignment,n_topics)
grafs.to_csv(str(topic) + "_Paragraphs_with_only_article_assignments.csv")

#LDA on paragraphs, combine in dataframe with article-level assignments
graf_assignments = lda_assignments(grafs, 'Paragraph',50, n_topics,10)
graf_assignments.to_csv(str(topic) + "_Paragraphs_with_topics.csv")

#Print top content associated with article and graf-level topics
#top_articles_per_topic(with_assignment,n_topics, 'Headline')
#top_articles_per_topic(graf_assignments,n_topics,'Paragraph')

terms_list = create_user_defined_topics(graf_assignments,'Paragraph')

#Add counts of ngrams identified by the user in each doc, add to dataframe
with_user_input = count_terms(graf_assignments,'Paragraph',terms_list)


#K-means cluster on article topics, graf topics, and key user terms
final_clustered_grafs = k_means_cluster(with_user_input,n_topics, "Cluster")
final_clustered_grafs.to_csv(str(topic) + "_final_clustered_grafs.csv")

#Print cluster shapes and top ngrams
#describe_clusters(final_clustered_grafs, n_topics)
summ = summarize_clusters(final_clustered_grafs,"Paragraph",n_topics)

#Output sample paragraphs from clusters to html
cluster_samples = sample_from_clusters(final_clustered_grafs,n_topics,10)
counter = 0
while counter < n_topics:
    file_name = "test topic " + str(counter)
    wrapStringInHTML(file_name,counter,cluster_samples[counter])
    counter +=1

#Output to bootstrap prototype
wrapStringInBootstrap("test bootstrap",cluster_samples)



