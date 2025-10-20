import pandas as pd
from collections import defaultdict, deque

def get_participant_by_name(participants, name):
    for participant in participants:
        if participant['name'] == name:
            return participant
    return None


def read_and_preprocess_data(file_path):
    """
    Чтение и предварительная обработка данных из CSV файла
    """
    # Чтение данных из CSV файла
    df = pd.read_csv(file_path)
    
    result = []

    
    # Извлечение имен оцениваемых (столбцы с оценками)
    rated_people = []
    for col in df.columns:
        if 'Поставьте оценку тому, с кем Вы работали в команде.' in col:
            # Извлечение имени из скобок
            name = col.split('[')[1].split(']')[0]
            result.append({
                'name': name,
                'score': 0,
                'ratings': {},
                'questions': []
            })
    
    for p1 in result:
        for p2 in result:
            if p1 != p2:
                p1['ratings'][p2['name']] = 4.5


    for _, row in df.iterrows():            
        for col in df.columns:
            if 'Поставьте оценку тому, с кем Вы работали в команде.' in col:
                rated_person = col.split('[')[1].split(']')[0]
                if row['Кто Вы?'] == rated_person:
                    continue  # Пропускаем самооценку'
                if not pd.isna(row[col]):
                    participant = get_participant_by_name(result, row['Кто Вы?'])
                    if participant:
                        participant['ratings'][rated_person] = row[col] 


    for person in result:
        person['score'] = sum(p['ratings'][person['name']] for p in result if p != person) / len(result)
    
    for person in result:
        person['ratings'] = sorted(person['ratings'], key=lambda key: person['ratings'][key] * 10 +  get_participant_by_name(result, key)['score'], reverse=True)

    result.sort(key=lambda x: x['score'], reverse=True)

    return result


def gale_shapley_teams(participants, team_size=2):
    """
    Реализация алгоритма Гейла-Шэпли для формирования равноправных команд
    """
    
    # Основной цикл алгоритма - пока есть свободные участники
    while True:
        count_free = len(list(filter(lambda p: len(p['questions']) < team_size, participants)))
        if count_free <= 1:
            break

        participant_to = list(filter(lambda p: len(p['questions']) < team_size, participants))[0]

        for i in range(len(participant_to['ratings'])):
            participant_from = get_participant_by_name(participants, participant_to['ratings'][i])
            if not participant_from or participant_from['questions']:
                continue

            participants.remove(participant_from)
            participant_from['ratings'].remove(participant_to['name'])
            participant_to['questions'].append(participant_from)
            
            print(f"{participant_to['name']} позвал(а) {participant_from['name']}")
            
            participant_to['questions'].sort(key=lambda x: participant_to['ratings'].index(x['name']))
            while len(participant_to['questions']) > team_size:
                drop = participant_to['questions'].pop()
                participants.append(drop)
                participants.sort(key=lambda x: x['score'], reverse=True)

                print(f"{drop['name']} ушел(а) от {participant_from['name']}")

            break
            

def format_equal_teams(teams):
    """
    Форматирование равноправных команд для вывода
    """
    
    for i, team in enumerate(teams):
        print(f"Команда {i+1}: ", ", ".join([team['name']]+[member['name'] for member in team['questions']]))


def main(file_path):
    """
    Основная функция для формирования равноправных команд
    """
    # Шаг 1: Чтение и предварительная обработка данных
    participants = read_and_preprocess_data(file_path)
    print(f"Всего участников: {len(participants)}")
    
    # Шаг 4: Запуск алгоритма Гейла-Шэпли для формирования команд
    gale_shapley_teams(participants)
    
    print("Алгоритм Гейла-Шэпли завершен")
    
    # Шаг 5: Форматирование и вывод результатов
    format_equal_teams(participants)

# Пример использования
if __name__ == "__main__":
    # Замените 'your_file.csv' на путь к вашему CSV файлу
    file_path = 'data.csv'
    
    try:
        main(file_path)
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Убедитесь, что путь указан правильно.")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")