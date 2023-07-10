import pandas as pd
import streamlit as st
import hydralit as hy

from hydralit import HydraApp
from io import StringIO

from exerciseGenerator import ExerciseGenerator
from exerciseRenderer import ExerciseRenderer


@st.cache_data
def __generate_exercises(file, skip_lines_from_start, skip_lines_from_end):
    eg = ExerciseGenerator()
    eg.load_text(file, skip_lines_from_start, skip_lines_from_end)

    eg.split_by_sentence()
    eg.enrich_by_words_info()
    choose_correct_verb = eg.choose_correct_verb(10)
    choose_correct_statement = eg.choose_correct_statement(3)
    choose_correct_text_summary = eg.choose_correct_text_summary()
    result = pd.concat([choose_correct_verb, choose_correct_statement, choose_correct_text_summary],
                       ignore_index=True,
                       sort=False)
    result.reset_index(drop=True, inplace=True)
    return result


if __name__ == '__main__':
    title = 'English exercises'
    app = HydraApp(title=title, favicon="🧑‍🏫")

    hy.title(title)

    fileName = 'Little_Red_Cap_ Jacob_and_Wilhelm_Grimm.txt'
    skip_from_the_beginning_key = 'number_input_skip_from_the_beginning'
    skip_from_the_end_key = 'number_input_skip_from_the_end'

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = hy.file_uploader(
            "Загрузите файл с английским текстом, по которому сгенерить упражнения "
            "(по умолчанию используется " + fileName + ')')
        if uploaded_file is not None:
            exercises = __generate_exercises(
                StringIO(uploaded_file.getvalue().decode("utf-8")),
                hy.session_state[skip_from_the_beginning_key],
                hy.session_state[skip_from_the_end_key])
        else:
            exercises = __generate_exercises(
                './input/' + fileName,
                2,
                5)

    with col2:
        hy.number_input(
            'Количество строк, которое пропустить с начала файла',
            format='%u',
            value=0,
            key=skip_from_the_beginning_key)
        hy.number_input(
            'Количество строк, которое пропустить с конца файла',
            format='%u',
            value=0,
            key=skip_from_the_end_key)

    hy.divider()

    exercises_types = list(exercises['description'].unique())

    for exercise_type in exercises_types:
        exercises_to_render = exercises.loc[exercises['description'] == exercise_type]
        exercises_to_render.reset_index(drop=True, inplace=True)
        app.add_app(exercise_type, app=ExerciseRenderer(exercises_to_render))

    app.run()
