from __future__ import print_function

def show_topics(topics):
    for i,topic in enumerate(topics):
        print('\n',i)
        for word in topic:
            print(word, end=' ')
            