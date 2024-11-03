import streamlit as st

# Заголовок приложения
st.title("Чат-бот с заглушкой TTS")

# Список для сохранения истории чата
chat_history = []

# Функция для отображения сообщений чата
def display_chat():
    for user_message, bot_response in chat_history:
        st.write(f"**Вы:** {user_message}")
        st.write(f"**Чат-бот:** {bot_response}")

# Ввод сообщения от пользователя
user_input = st.text_input("Введите ваш вопрос:")

# Проверяем, было ли введено сообщение от пользователя
if user_input:
    # Ответ от бота (заглушка)
    bot_response = "Привет!"
    
    # Добавляем диалог в историю
    chat_history.append((user_input, bot_response))
    
    # Отображаем историю чата
    display_chat()

# Кнопка для воспроизведения (заглушка)
if st.button("Воспроизвести"):
    st.write("Функция озвучивания в разработке...")

# Отображаем историю чата при первой загрузке
display_chat()
