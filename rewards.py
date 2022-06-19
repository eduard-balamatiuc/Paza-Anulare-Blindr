import streamlit as st
import face_mathcing_func_update as fmf

#create st.form with 3 text inputs  
photo_checked = False
with st.form("Reward sumbission"):
    st.write("This is the reward submission page")
    st.write("Please enter your reward details")
    event_id = st.text_input("Event ID")
    image = st.file_uploader("Image proof", type=["jpg", "png"])
    value_photo = st.form_submit_button("Submit photo")
    if value_photo:
        st.write("Thank you for your submission. We are now checking your photo")
    photo_checked = fmf.check_presence('facenet_keras.h5', 'will.jpg', 'group.jpg')
    if photo_checked:
        review_text = st.text_input("Give a review to the others")
        value_review = st.form_submit_button("Submit review")
        if value_review:
            st.write("Thank you for your review. You got 200 kapiiki")
    else:
        st.write("Sorry, our alghoritm didn't detect you in the photo. Please try again")