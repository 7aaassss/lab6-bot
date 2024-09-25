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
import config

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
PRODUCTS = [
    {
        'name': 'Шамупнь Лошадиная Сила',
        'description': 'Отличный шампунь, подходящий для всех типов волос',
        'price': '500 руб.'
    },
    {
        'name': 'Гель для душа Акс',
        'description': 'Отличный выбор для спортсменов, помогающий нейтрализовать неприятный запах на долгое время',
        'price': '399 руб.'
    },
    {
        'name': 'Зубная паста',
        'description': 'Отбеливание зубов, придание им естественного цвета, нейтрализация неприятного запаха',
        'price': '299 руб.'
    },

]

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
    global dialog_steps
    dialog_steps += 1  # Увеличиваем шаг диалога


    intent = classify_intent(replica)


    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intent'] += 1

            # Вставляем рекламу каждые 3-5 шагов диалога
            if dialog_steps % random.randint(3, 5) == 0:
                ad = show_ad()
                return f"{answer}\n\n{ad}"

            return answer

    # Генерация ответа
    answer = generate_answer(replica)
    if answer:
        stats['generate'] += 1

        # Вставляем рекламу каждые 3-5 шагов диалога
        if dialog_steps % random.randint(3, 5) == 0:
            ad = show_ad()
            return f"{answer}\n\n{ad}"

        return answer

    stats['failure'] += 1
    failure_answer = get_failure_phrase()

    # Вставляем рекламу каждые 3-5 шагов диалога
    if dialog_steps % random.randint(3, 5) == 0:
        ad = show_ad()
        return f"{failure_answer}\n\n{ad}"

    return failure_answer


dialog_steps = 0  # переменная для отслеживания шагов диалога

def show_ad():
    """Возвращает случайное рекламное сообщение из списка товаров."""
    product = random.choice(PRODUCTS)
    return f"Кстати, вам может понравиться {product['name']}! {product['description']} Всего за {product['price']}."


dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer('Напиши мне что-то и я тебе отвечу!')


@dp.message()
async def echo_handler(message: Message) -> None:
    try:
        answ = response(message.text)
        await message.answer(answ)
    except TypeError:
        await message.answer("Nice try!")


async def main() -> None:
    bot = Bot(token=config.Config.TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())