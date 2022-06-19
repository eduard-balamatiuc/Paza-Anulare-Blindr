import streamlit as st
import pandas as pd
import numpy as np

import requests
from PIL import Image
from io import BytesIO
from enchant.utils import levenshtein
from scipy.spatial import distance
import streamlit.components.v1 as components
from face_matching_func_update import *

compare_threshold = lambda x: 1 if x > 0.1 else 0
transform_to_list = lambda x: eval(x)
levenshtein_list = lambda user1, user2: np.mean([compare_threshold(
    np.min([levenshtein(element_user1, element_user2) / len(element_user1) for element_user2 in user2]))
    for element_user1 in user1])


def try_transform_to_list(words):
	try:
		return transform_to_list(words)
	except:
		print('Cannot convert to string.')
	return words


def levanshtein_validation(unique_key, path_to_data='db.csv', similarity_criterion=0.6):
	interest_columns = ['spend_time', 'movies', 'music', 'pet', 'profession', 'hobbies', 'social_status', 'gender', 'food', ]

	df = pd.read_csv(path_to_data, sep = '$')
	
	#print(type(unique_key))
	unique_key = unique_key[5:]

	current_user_location = df[df['key'] == str(unique_key)]['city'].values[0]
	df_location_user = df[(df['city'] == current_user_location) & (df['key'] != str(unique_key))]
	current_user_interest = df[df['key'] == unique_key][interest_columns].values[0]

	matching_users_id = []

	for idx, user in enumerate(df_location_user[interest_columns].astype(str).values):
		levenshtein_dist = []
		for user_interest, user_compared in zip(current_user_interest, user):

			user_interest = try_transform_to_list(user_interest)
			user_compared = try_transform_to_list(user_compared)

			if isinstance(user_interest, list) or isinstance(user_compared, list):
			    levenshtein_dist.append(compare_threshold(levenshtein_list(user_interest, user_compared)))
			else:
				levenshtein_dist.append(compare_threshold(levenshtein(user_interest, user_compared) / len(user_interest)))
		similarity_value = np.mean(np.array(levenshtein_dist))

		if similarity_value < similarity_criterion:
			matching_users_id.append(idx)

	return matching_users_id


def match_user_by_face(df, unique_user_embedding):

    user_ids = []

    for idx, embedding in enumerate(df['image_embedding'].values):
        signature = get_embedding('facenet_keras.h5', embedding)
        distances = distance.cosine(unique_user_embedding, signature)
        if distances < 0.9:
            user_ids.append(idx)

    return df.iloc[user_ids]


def search():
    user_unique_key = 'fd746164-72ff-44ed-b156-9e852038ad3a'
    st.title('Search Event')
    option_mapper = {}

    with st.form("Submission"):
        option = st.selectbox(
            'Select category:',
            (None, 'Performances', 'Movies', 'Sports', 'theatres', 'concerts', 'various', 'conferences', 'exhibitions'))
        option_age = st.selectbox(
            'Select age range:',
            (None, '18-25', '25-35', '35-45'))
        option_by_face = st.checkbox('Face match')
        submitted = st.form_submit_button("Submit")

        option_mapper['option'] = option

        if submitted:

            matched_users_id = levanshtein_validation(user_unique_key)#
            data = pd.read_csv('df.csv')
            match_users_data = data.iloc[matched_users_id]

            if option_age is not None:
                age_range = [int(age) for age in option_age.split('-')]
                match_users_data = match_users_data[(match_users_data['age'] >= age_range[0]) &
                                                    (match_users_data['age'] <= age_range[1])]

            if option_by_face:
                match_users_data = match_user_by_face(match_users_data, data[data['key'] == user_unique_key]['face_embedding'].values[0])

            st.table(match_users_data[['name', 'description', 'review']].reset_index().drop('index', axis=1))

            st.markdown('## Events:')

    events = pd.read_csv('events.csv')

    for idx, event in enumerate(events[events['Category'] == option_mapper['option']].iloc[:3].iterrows()):
        with st.form(f'Events_{idx}'):

            response = requests.get(event[1]['Image'])
            img = np.array(Image.open(BytesIO(response.content)))
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.write(' ')
            with col2:
                st.image(img, width=400)
            with col3:
                st.write(' ')
            with col4:
                st.write(' ')
            with col5:
                st.write(' ')

            st.markdown(f"### {event[1]['Name']}")
            st.markdown(f"Date: {event[1]['Date']}")
            st.markdown(f"Price: {event[1]['Price']}")

            go_to_link = st.form_submit_button("Get to the site")
            if go_to_link:
                # embed streamlit docs in a streamlit app
                components.iframe(event[1]['Link'])


if __name__ == '__main__':
    search()
