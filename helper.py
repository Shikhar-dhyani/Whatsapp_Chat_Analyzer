
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

extract=URLExtract()

def fetch_stats(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]

    num_messages=df.shape[0]
    words=[]
    links=[]

    for message in df['message']:
        words.extend(message.split())
        links.extend(extract.find_urls(message))
    
    num_media_messages = df[df['message']=='<Media omitted>\n'].shape[0]
    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x=df['user'].value_counts().head()
    df=round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'index':'name','user':'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f=open('stop_hinglish.txt','r')
    stop_words=f.read()
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
    
    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']
    temp=temp[temp['message']!='message deleted']
    def remove_stop_words(message):
        y=[]
        for word in message.lower().split():
           if word not in stop_words:
             y.append(word)
        return " ".join(y)
    
    #key-frame ,text summarization ,rake_NLTK ,KPE 
    wc=WordCloud(width=500,height=500,min_font_size=10,background_color='black')
    temp['message']=temp['message'].apply(remove_stop_words)
    df_wc=wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    f=open('stop_hinglish.txt','r')
    stop_words=f.read()
    
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
    
    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']

    words=[]

    for message in temp['message']:
        for word in  message.lower().split():
            if word not in stop_words :
                words.append(word)

                

    most_common_df= pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

 #hugging face bert 

def emoji_helper(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
    
    emojis=[]
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])
    
    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(emojis)))
    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user!='Overall':
        df=df[df['user']==selected_user]

    timeline=df.groupby(['year','month_num','month']).count()['message'].reset_index()

    time=[]

    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time']=time

    return timeline


def daily_timeline(selected_user,df):

    if selected_user!='Overall':
        df=df[df['user']==selected_user]

    timeline=df.groupby('only_date').count()['message'].reset_index()
    
    return timeline

def week_activity_map(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]

    return df['day_name'].value_counts()


def month_activity_map(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
    
    return df['month'].value_counts()

def activity_heatmap(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
    activity_heatmap=df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)
    return activity_heatmap


def Sentimental_Analysis(selected_user,df):
    if selected_user!='Overall':
        df=df[df['user']==selected_user]
    RawData=""

    for message in df['message']:
        RawData+=message
    
    return RawData

def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df


def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
    return result 


