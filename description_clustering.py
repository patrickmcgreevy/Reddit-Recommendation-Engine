import spacy

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np
from sklearn.decomposition import PCA
from sklearn import neighbors
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import AgglomerativeClustering
#from sklearn.preprocessing import StandardScaler
#from keras import Sequential
#from keras.layers import Dense, Dropout
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import to_categorical

en_model = spacy.load("en_core_web_lg")
#doc = en_model("Patrick has long been enamored with Katlin. She tickles him.")

#for t in doc:
#    print(t.text, t.lemma_, t.pos_, t.tag_, t.dep_, t.shape_, t.is_alpha, t.is_stop)


#reddit_df = pd.read_csv('subreddit_textdata.csv', index_col=False)
reddit_df = pd.read_csv('subreddit_data_imputed_na.csv', index_col=False)

rdf = reddit_df.assign(desc_vec=[en_model(i) for i in reddit_df['description']])
rdf = rdf.assign(comment_one_vec=[en_model(i) for i in reddit_df['first_post']])
rdf = rdf.assign(comment_two_vec=[en_model(i) for i in reddit_df['second_post']])
rdf = rdf.assign(comment_three_vec=[en_model(i) for i in reddit_df['third_post']])
combined_text = reddit_df['description'] + ' ' + reddit_df['first_post'] + ' ' + reddit_df['second_post']
rdf = rdf.assign(subreddit_vec=[en_model(i) for i in combined_text])


desc_similar_df = pd.DataFrame(data=[[i.similarity(j) for i in rdf['desc_vec']] for j in rdf['desc_vec']], index=rdf['name'], columns=rdf['name'])
c1_similar_df = pd.DataFrame(data=[[i.similarity(j) for i in rdf['comment_one_vec']] for j in rdf['comment_one_vec']], index=rdf['name'], columns=rdf['name'])
c2_similar_df = pd.DataFrame(data=[[i.similarity(j) for i in rdf['comment_two_vec']] for j in rdf['comment_two_vec']], index=rdf['name'], columns=rdf['name'])
c3_similar_df = pd.DataFrame(data=[[i.similarity(j) for i in rdf['comment_three_vec']] for j in rdf['comment_three_vec']], index=rdf['name'], columns=rdf['name'])
combined_similar_df = pd.DataFrame(data=[[i.similarity(j) for i in rdf['subreddit_vec']] for j in rdf['subreddit_vec']], index=rdf['name'], columns=rdf['name'])

desc_vectors = [i.vector for i in rdf['desc_vec']]
comment_one_vectors = [i.vector for i in rdf['comment_one_vec']]
comment_two_vectors = [i.vector for i in rdf['comment_two_vec']]
comment_three_vectors = [i.vector for i in rdf['comment_three_vec']]

avg_vectors = [i.vector + j.vector + k.vector + l.vector for i,j,k,l in zip(rdf['desc_vec'], rdf['comment_one_vec'], rdf['comment_two_vec'], rdf['comment_three_vec'])]
concat_vectors = [np.concatenate((i.vector, j.vector, k.vector, l.vector))for i,j,k,l in zip(rdf['desc_vec'], rdf['comment_one_vec'], rdf['comment_two_vec'], rdf['comment_three_vec'])]
avg_vec_vocab = spacy.vocab.Vocab()
for key, vec in zip(reddit_df['name'], avg_vectors):
    #print(key)
    avg_vec_vocab.set_vector(key, vec)

desc_vectors_vocab = spacy.vocab.Vocab()
for key, vec in zip(reddit_df['name'], desc_vectors):
    desc_vectors_vocab.set_vector(key, vec)
#concat_pca = PCA(n_components=300)
#concat_vectors_n_300 = concat_pca.fit_transform(concat_vectors)
#concat_vec_vocab = spacy.vocab.Vocab()
#for key, vec in zip(reddit_df['name'], concat_vectors_n_300):
#    concat_vec_vocab.set_vector(key, vec)


def get_most_similar(key, n, vocab, backup_vocab=None):
    '''r = avg_vec_vocab.vectors.most_similar(np.asarray([avg_vec_vocab.get_vector(key)]), n=n)
    names = []
    scores = []
    for k in r[0]:
        for l in k:
            names.append(avg_vec_vocab.strings.as_string(l))

    for s in r[2]:
        scores.append(s)

    return names, scores
    '''
    d = {}
    #if not vocab.__contains__(key): # Fix this!
    if np.array_equal(vocab.get_vector(key),np.zeros((300,))):
        if backup_vocab is None:
            return d
        else:
            r = vocab.vectors.most_similar(np.asarray([backup_vocab.get_vector(key)]), n=n)
    else:
        r = vocab.vectors.most_similar(np.asarray([vocab.get_vector(key)]), n=n)

    keys = []
    vals = []

    for k in r[0]:
        for l in k:
            keys.append(vocab.strings.as_string(l))

    for j in r[2]:
        for v in j:
            vals.append(v)

    for i in range(len(keys)):
        d[keys[i]] = vals[i]

    return d


def save_heatmap(data, rowlabels, collabesl, imgname, title):
    plt.matshow(data)
    #fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data)
    #cax.set_ylabel('Vector Cosine Similarity', rotation=-90, va="bottom")
    fig.colorbar(cax)
    ax.set_xticks(np.arange(data.shape[0]))
    ax.set_yticks(np.arange(data.shape[1]))
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
    ax.set_xticklabels(rowlabels)
    ax.set_yticklabels(collabesl)

    for edge, spine in ax.spines.items():
        spine.set_visible=False

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, pad=50)
    #plt.title(title, y=1.25)
    #fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(imgname)

def save_scatter_plot(data, classes, color, imgname, x_label='PC1', y_label='PC2', title='PCA',
                      label_key='category'):

    nColors = len(set(classes))
    cm = plt.get_cmap('gist_rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=nColors-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)


    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(nColors)])

    for label in classes:
        indicesToKeep = data[label_key] == label
        ax.scatter(data.loc[indicesToKeep, x_label],
                   data.loc[indicesToKeep, y_label],
                   s=50)
    '''for label, color in zip(classes, colors):
        indicesToKeep = data[label_key] == label
        ax.scatter(data.loc[indicesToKeep, x_label],
                   data.loc[indicesToKeep, y_label],
                   c=color,
                   s=50)'''

    ax.legend(classes)
    ax.grid()
    plt.tight_layout()
    plt.savefig(imgname)

# Saves an example of the classification boundaries for a subset of the data that has been reduced to two dimensions
def save_knn_decision_boundaries():
    X = reduced_desc_vectors
    y = rdf['category']
    h = .02 # step size in the mesh
    n_neighbors = 3
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        #plt.scatter(X[0], X[1], c=y, cmap=cmap_bold,
        #            edgecolor='k', s=20)

        for label, color in zip(y, ['r', 'b', 'g']):
            indicesToKeep = y == label
            plt.scatter(pd.DataFrame(X[0]).loc[indicesToKeep],
                        pd.DataFrame(X[1]).loc[indicesToKeep],
                        c=color, edgecolors='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))

    plt.show()

def save_variance_explained_plot(data, title):
    model = pca.fit(data)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Proportion of Variance Explained')
    ax.plot(np.cumsum(model.explained_variance_ratio_))
    ax.grid()
    plt.tight_layout()
    plt.savefig(title+'.png')


def save_bar_chart(data, x_index, y_index, x_label, y_label, title, imgname):
    nColors = len(set(data[x_index]))
    cm = plt.get_cmap('gist_rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=nColors-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)


    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    #ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(nColors)])

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    ax.bar(np.arange(len(data[x_index])), data[y_index], tick_label=data[x_index],
           color=[scalarMap.to_rgba(i) for i in range(nColors)])

    #ax.legend(data[x_index])
    #ax.grid()
    plt.tight_layout()
    plt.savefig(imgname)

pca = PCA(n_components=2)
cols = ['PC1', 'PC2']
cats = rdf['category']
names = rdf['name']

reduced_avg_vectors = pca.fit_transform(avg_vectors)
rav_labeled = pd.DataFrame(reduced_avg_vectors, columns=['PC1', 'PC2']).assign(category=rdf['category']).assign(name=rdf['name'])

reduced_concat_vectors = pca.fit_transform(concat_vectors)
rc_labeled = pd.DataFrame(reduced_concat_vectors, columns=cols).assign(category=cats).assign(name=names)
#save_scatter_plot(rc_labeled, set(cats), None, 'concat_data_pca.png', title='All Data Concatonation PCA')
#save_scatter_plot(rav_labeled, set(rdf['category']), ['r', 'g', 'b'], 'pca_plot2.png', title='Avg. Vectors PCA')
reduced_desc_vectors = pca.fit_transform(desc_vectors)
rd_labeled = pd.DataFrame(reduced_desc_vectors, columns=['PC1', 'PC2']).assign(category=rdf['category']).assign(name=rdf['name'])

reduced_c1_vectors = pca.fit_transform(comment_one_vectors)
rc1_vectors = pd.DataFrame(reduced_c1_vectors, columns=cols).assign(category=cats).assign(name=names)

reduced_c2_vectors = pca.fit_transform(comment_two_vectors)
rc2_vectors = pd.DataFrame(reduced_c2_vectors, columns=cols).assign(category=cats).assign(name=names)

reduced_c3_vectors = pca.fit_transform(comment_three_vectors)
rc3_vectors = pd.DataFrame(reduced_c3_vectors, columns=cols).assign(category=cats).assign(name=names)


# Ought to try normalizing the data first!

#for d, t in zip([rd_labeled, rc1_vectors, rc2_vectors, rc3_vectors], ['Normalized Combined Text', 'Normalized Subreddit_Description', 'Normalized First Post', 'Normalized Second Post', 'Normalized Third Post']):
#    save_scatter_plot(d, set(cats), None, 'all_data_pca_'+t+'.png', title='All Data '+t+' PCA')

#save_knn_decision_boundaries()


# Clustering Happening HEREEE!!!
print('Clustering in progress...')
knn_model = neighbors.KNeighborsClassifier()
hierarchical_model = AgglomerativeClustering(n_clusters=20)

h_param_map = {'affinity':['euclidean', 'cosine'], 'linkage':['complete', 'average', 'single']}
param_grid = {'n_neighbors':np.arange(1, 10)}

knn_gscv = GridSearchCV(knn_model, param_grid, cv=5)
h_gscv = GridSearchCV(hierarchical_model, h_param_map, cv=5)

pca = PCA(n_components=50)
# Create 50-dimensional projections that capture around 90% of the variance in our data
reduced_avg_vectors = pca.fit_transform(avg_vectors)
reduced_concat_vectors = pca.fit_transform(concat_vectors)

knn_results = pd.DataFrame(columns=['name', 'n_neighbors', 'score'])
print('KNN Results')
for X, name in zip([reduced_avg_vectors, reduced_concat_vectors, avg_vectors, concat_vectors],
                   ['Reduced Vectors Average', 'Reduced Vectors Concatenated', 'Vectors Averaged', 'Vectors Concatenated']):
    knn_gscv.fit(X, cats)
    print(name, '\nBest n_neighbors Param:', knn_gscv.best_params_, '\nScore:', knn_gscv.best_score_, sep=' ')
    knn_results.loc[-1] = [name, knn_gscv.best_params_['n_neighbors'], knn_gscv.best_score_]
    knn_results.index = knn_results.index + 1
    knn_results = knn_results.sort_index()


#save_bar_chart(knn_results, 'name', 'score', 'Form of Vector', 'Accuracy', 'KNN Classification Results',
#               'knn_classification_accuracy.png')

'''print('Hierarchical Results')

for X, name in zip([reduced_avg_vectors, reduced_concat_vectors, avg_vectors, concat_vectors],
                   ['Reduced Average', 'Reduced Concatenated', 'Average', 'Concatenated']):
    h_gscv.fit(X, cats)
    print(name, '\nBest Parameters:', knn_gscv.best_params_, '\nScore:', knn_gscv.best_score_, sep=' ')

h_param_map = {'affinity':['euclidean'], 'linkage':['ward']}
h_gscv = GridSearchCV(hierarchical_model, h_param_map, cv=5)

for X, name in zip([reduced_avg_vectors, reduced_concat_vectors, avg_vectors, concat_vectors],
                   ['Reduced Average', 'Reduced Concatenated', 'Average', 'Concatenated']):
    h_gscv.fit(X, cats)
    print(name, '\nBest Parameters:', knn_gscv.best_params_, '\nScore:', knn_gscv.best_score_, sep=' ')

#knn_gscv.fit(reduced_concat_vectors, cats)
#print('Best n_neighbors Param:', knn_gscv.best_params_, '\nScore:', knn_gscv.best_score_, sep=' ')
'''
print('Neural Network Training')
def build_nn(input_size=50):
    nn = Sequential()
    nn.add(Dense(20, activation='relu',
                 kernel_initializer='random_uniform', input_dim=input_size))
    nn.add(Dense(20, activation='relu', kernel_initializer='random_uniform'))
    nn.add(Dense(20, activation='relu', kernel_initializer='random_uniform'))
    nn.add(Dense(20, activation='softmax', kernel_initializer='random_uniform'))

    nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return nn

'''sub_classifier = KerasClassifier(build_fn=build_nn, batch_size=1, nb_epoch=150)

category_dummies = pd.get_dummies(cats)

for X, name in zip([reduced_avg_vectors, reduced_concat_vectors], ['Reduced Average', 'Reduced Concatenated']):
    sub_eval = cross_val_score(estimator=sub_classifier, X=X, y=category_dummies, cv=5, n_jobs=-1)
    print(name, '\nCategory Accuracy:', sub_eval, '\nMean Accuracy:', sub_eval.mean(), sep=' ')
'''

print('Most Similar Metrics')
most_similar_metrics = reddit_df[['name', 'category']]
sub_cat_dict = dict(zip(most_similar_metrics.name, most_similar_metrics.category))

def get_most_similar_metrics(data, map, n, vocab, backup_vocab):
    #For each name in data, get most similar subs
    most_similar = [get_most_similar(name, n, vocab, backup_vocab).keys() for name in data['name']]
    # Compute correct_labels/n
    acc = []
    for i, cat in zip(most_similar, data['category']):
        n_correct = 0
        for sub in i:
            if map[sub] == cat:
                n_correct += 1
        acc.append(n_correct/n)
    # return df with new columns assigned_label and accuracy

    return data.assign(accuracy=acc)


most_similar_metrics = get_most_similar_metrics(most_similar_metrics, sub_cat_dict, 5, avg_vec_vocab, en_model.vocab)
desc_metrics = get_most_similar_metrics(most_similar_metrics, sub_cat_dict, 3, desc_vectors_vocab, en_model.vocab)
category_mean = desc_metrics.groupby(['category']).mean().sort_values(['accuracy']).reset_index(['category'])
save_bar_chart(category_mean, 'category', 'accuracy', 'Subreddit Category', 'Mean Category Accuracy', 'Description Vecotr Cosine Similarity Classification by Category', 'desc_cos_similarity_category_bar.png')