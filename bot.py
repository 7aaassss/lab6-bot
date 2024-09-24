import random
import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
import logging
import sys
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message


BOT_CONFIG = {
    'intents': {

        'hello': {
            'examples': ['Привет', 'Добрый день', 'Шалом', 'Привет, бот'],
            'responses': ['Привет, человек!', 'И вам здравствуйте :)', 'Доброго времени суток']
        },
        'bye': {
            'examples': ['Пока', 'До свидания', 'До свидания', 'До скорой встречи'],
            'responses': ['Еще увидимся', 'Если что, я всегда тут']
        },
        'name': {
            'examples': ['Как тебя зовут?', 'Скажи свое имя', 'Представься'],
            'responses': ['Меня зовут Саша']
        },
        'want_eat': {
            'examples': ['Хочу есть', 'Хочу кушать', 'ням-ням'],
            'responses': ['Вы веган?'],
            'theme_gen': 'eating_q_wegan',
            'theme_app': ['eating', '*']
        },
        'yes': {
            'examples': ['да'],
            'responses': ['капусты или морковки?'],
            'theme_gen': 'eating_q_meal',
            'theme_app': ['eating_q_wegan']
        },
        'no': {
            'examples': ['нет'],
            'responses': ['мясо или творог?'],
            'theme_gen': 'eating_q_meal',
            'theme_app': ['eating_q_wegan']
        },
    },

    'failure_phrases': [
        'Непонятно. Перефразируйте, пожалуйста.',
        'Я еще только учусь. Спросите что-нибудь другое',
        'Слишком сложный вопрос для меня.',
    ]
}

X_text = []  # ['Хэй', 'хаюхай', 'Хаюшки', ...]
y = []  # ['hello', 'hello', 'hello', ...]

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_text.append(example)
        y.append(intent)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC()
clf.fit(X, y)


def clear_phrase(phrase):
    phrase = phrase.lower()

    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- '
    result = ''.join(symbol for symbol in phrase if symbol in alphabet)

    return result.strip()


def classify_intent(replica):
    replica = clear_phrase(replica)

    intent = clf.predict(vectorizer.transform([replica]))[0]

    for example in BOT_CONFIG['intents'][intent]['examples']:
        example = clear_phrase(example)
        distance = nltk.edit_distance(replica, example)
        if example and distance / len(example) <= 0.5:
            return intent


def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses']
        if responses:
            return random.choice(responses)


with open('dialogues.txt', encoding='utf-8') as f:
    content = f.read()

dialogues_str = content.split('\n\n')
dialogues = [dialogue_str.split('\n')[:2] for dialogue_str in dialogues_str]

dialogues_filtered = []
questions = set()

for dialogue in dialogues:
    if len(dialogue) != 2:
        continue

    question, answer = dialogue
    question = clear_phrase(question[2:])
    answer = answer[2:]

    if question != '' and question not in questions:
        questions.add(question)
        dialogues_filtered.append([question, answer])

dialogues_structured = {}  # {'word': [['...word...', 'answer'], ...], ...}

for question, answer in dialogues_filtered:
    words = set(question.split(' '))
    for word in words:
        if word not in dialogues_structured:
            dialogues_structured[word] = []
        dialogues_structured[word].append([question, answer])

dialogues_structured_cut = {}
for word, pairs in dialogues_structured.items():
    pairs.sort(key=lambda pair: len(pair[0]))
    dialogues_structured_cut[word] = pairs[:1000]


# replica -> word1, word2, word3, ... -> dialogues_structured[word1] + dialogues_structured[word2] + ... -> mini_dataset

def generate_answer(replica):
    replica = clear_phrase(replica)
    words = set(replica.split(' '))
    mini_dataset = []
    for word in words:
        if word in dialogues_structured_cut:
            mini_dataset += dialogues_structured_cut[word]



    answers = []  # [[distance_weighted, question, answer]]

    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < 0.2:
                answers.append([distance_weighted, question, answer])

    if answers:
        return min(answers, key=lambda three: three[0])[2]


def get_failure_phrase():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)


stats = {'intent': 0, 'generate': 0, 'failure': 0}


def response(replica):
    # NLU
    intent = classify_intent(replica)

    # Answer generation

    # выбор заготовленной реплики
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intent'] += 1
            return answer

    # вызов генеративной модели
    answer = generate_answer(replica)
    if answer:
        stats['generate'] += 1
        return answer

    # берем заглушку
    stats['failure'] += 1
    return get_failure_phrase()



TOKEN = '7848683745:AAHAXMg1qHCXBG0E91XW8kPK6iVQtCcGFjg'
dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    await message.answer('Напиши мне что-то и я тебе отвечу!')


@dp.message()
async def echo_handler(message: Message) -> None:
    """
    Handler will forward receive a message back to the sender

    By default, message handler will handle all message types (like a text, photo, sticker etc.)
    """
    try:
        answ = response(message.text)
        await message.answer(answ)
    except TypeError:
        # But not all the types is supported to be copied so need to handle it
        await message.answer("Nice try!")


async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN)

    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())