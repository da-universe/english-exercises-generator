import numpy as np
import streamlit as st

from hydralit import HydraHeadApp


class ExerciseRenderer(HydraHeadApp):
    def __init__(self, exercises):
        self.exercises = exercises

    def run(self):
        description = self.exercises.iloc[0]['description']
        exercise_type = self.exercises.iloc[0]['exercise_type']
        radio_buttons_groups_metadata = []
        for index, row in self.exercises.iterrows():
            radio_buttons_metadata = self.__render_select_one_option_exercise(row, index + 1)
            radio_buttons_groups_metadata.append(radio_buttons_metadata)
            st.divider()

        col1, col2 = st.columns(2)

        with col1:
            if st.button('Проверить', key=exercise_type + '_button_1'):
                for metadata in radio_buttons_groups_metadata:
                    if st.session_state[metadata.key] == metadata.answer:
                        metadata.verify_result_placeholder.text('Правильный ответ 🔥')
                    else:
                        metadata.verify_result_placeholder.text('Подумай получше 🤔')

        with col2:
            if st.button('Очистить', key=exercise_type + '_button_2'):
                for metadata in radio_buttons_groups_metadata:
                    metadata.verify_result_placeholder.empty()

    @staticmethod
    def __render_select_one_option_exercise(data_row, index):
        key = data_row['exercise_type'] + '_' + str(index)
        st.write('Задание ' + str(index))
        if not data_row['exercise_context']:
            label_visibility = 'collapsed'
            label = ' '
        else:
            label_visibility = 'visible'
            label = data_row['exercise_context']
        st.radio(
            label=label,
            options=np.array(data_row['options']),
            key=key,
            label_visibility=label_visibility)
        verify_result_placeholder = st.empty()
        return RadioButtonsGroupMetadata(key, verify_result_placeholder, data_row['answer'])


class RadioButtonsGroupMetadata:
    def __init__(self, key, verify_result_placeholder, answer):
        self.key = key
        self.verify_result_placeholder = verify_result_placeholder
        self.answer = answer
