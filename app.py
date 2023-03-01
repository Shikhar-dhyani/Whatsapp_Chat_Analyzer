import streamlit as st
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from textblob import TextBlob



st.sidebar.title("Whatsapp Chat Analyser")

uploaded_file = st.sidebar.file_uploader("Choose a Chat Exported From Whatsapp")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data= bytes_data.decode("utf-8")
    df=preprocessor.preprocess(data)
    # st.dataframe(df)

    user_list=df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user= st.sidebar.selectbox("Show analysis wrt",user_list)
    num_messages , words , num_media_messages,num_links =helper.fetch_stats(selected_user,df)

    if st.sidebar.button("Show Analysis"):
        st.title("Top Statistics")
        col1,col2,col3,col4=st.columns(4)
        
        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
           st.header(" Media Shared")
           st.title(num_media_messages)

        with col4:
             st.header("Links Shared")
             st.title(num_links)

        #monthly timeline
        st.title("Monthly Timeline")
        timeline=helper.monthly_timeline(selected_user,df)
        fig,ax=plt.subplots()
        ax.plot(timeline['time'],timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)


        #daily timeline
        st.title("Daily Timeline")
        DailyTimeline=helper.daily_timeline(selected_user,df)
        fig,ax=plt.subplots()
        ax.plot(DailyTimeline['only_date'],DailyTimeline['message'],color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
        

        #activity_map
        st.title("Activity Map")
        col1,col2=st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day=helper.week_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month=helper.month_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_month.index,busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)


        st.title("Weekly Activity Heatmap")
        user_heatmap=helper.activity_heatmap(selected_user,df)
        fig,ax=plt.subplots()
        ax= sns.heatmap(user_heatmap)
        st.pyplot(fig)

        #finding most busy users

        if selected_user=='Overall':
            st.title('Most Busy Users')
            x,new_df=helper.most_busy_users(df)
            fig,ax=plt.subplots()

            col1,col2=st.columns(2)

            with col1:
                ax.bar(x.index,x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            
            with col2:
                st.dataframe(new_df)
        
        #WordCloud 
        st.title("WordCloud")
        df_wc=helper.create_wordcloud(selected_user,df)
        fig,ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        
        #most common words
        most_common_df=helper.most_common_words(selected_user,df)
        fig,ax=plt.subplots()
           
        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most Common Words')
        st.pyplot(fig)


        #emoji analysis

        emoji_df=helper.emoji_helper(selected_user,df)
        st.title('Emoji Analysis')
        col1,col2 = st.columns(2)
      
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")

        st.pyplot(fig)
        


        #sentimental analysis
        st.title("Sentiment Analysis NLP")
        col1,col2 =st.columns(2)
        raw_text=helper.Sentimental_Analysis(selected_user,df)
        with col1:
            st.subheader("Results")
            sentiment = TextBlob(raw_text).sentiment
            st.write(sentiment)

				# Emoji
            if sentiment.polarity > 0:
                st.markdown("Sentiment:: Positive :smiley: ")
            elif sentiment.polarity < 0:
                st.markdown("Sentiment:: Negative :angry: ")
            else:
                st.markdown("Sentiment:: Neutral 😐 ")

				# Dataframe
            result_df = helper.convert_to_df(sentiment)
            st.dataframe(result_df)

				# Visualization
            c = alt.Chart(result_df).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
            st.altair_chart(c,use_container_width=True)


        with col2:
            st.info("Token Sentiment")

            token_sentiments = helper.analyze_token_sentiment(raw_text)
            st.write(token_sentiments)
