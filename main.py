import pandas as pd
import numpy as np
from collections import defaultdict, deque

def read_and_preprocess_data(file_path):
    """
    Чтение и предварительная обработка данных из CSV файла
    """
    # Чтение данных из CSV файла
    df = pd.read_csv(file_path)
    
    # Извлечение имен оценивающих (те, кто заполнял анкету)
    evaluators = df['Кто Вы?'].dropna().unique().tolist()
    
    # Извлечение имен оцениваемых (столбцы с оценками)
    rated_people = []
    for col in df.columns:
        if 'Поставьте оценку с тем, с кем вы когда-либо работали в команде.' in col:
            # Извлечение имени из скобок
            name = col.split('[')[1].split(']')[0]
            rated_people.append(name)
    
    # Объединяем всех участников (и оценивающих и оцениваемых)
    all_participants = list(set(evaluators + rated_people))
    
    return df, all_participants

def create_preference_matrix(df, all_participants):
    """
    Создание матрицы предпочтений между всеми участниками
    """
    # Словарь для хранения оценок между всеми участниками
    preference_matrix = defaultdict(dict)
    
    # Инициализация матрицы предпочтений (по умолчанию оценка 5)
    for person1 in all_participants:
        for person2 in all_participants:
            if person1 != person2:
                preference_matrix[person1][person2] = 5  # оценка по умолчанию
    
    # Заполнение матрицы реальными оценками из данных
    for _, row in df.iterrows():
        evaluator = row['Кто Вы?']
        if pd.isna(evaluator):
            continue
            
        # Обрабатываем каждую колонку с оценками
        for col in df.columns:
            if 'Поставьте оценку с тем, с кем вы когда-либо работали в команде.' in col:
                rated_person = col.split('[')[1].split(']')[0]
                rating = row[col]
                
                # Если оценка указана, используем её, иначе оставляем 5
                if not pd.isna(rating):
                    preference_matrix[evaluator][rated_person] = rating

    return preference_matrix

def create_mutual_preferences(preference_matrix, all_participants):
    """
    Создание взаимных списков предпочтений для алгоритма Гейла-Шэпли
    """
    mutual_preferences = {}
    
    for person in all_participants:
        # Создаем список предпочтений на основе взаимных оценок
        preferences = []
        for other_person in all_participants:
            if person != other_person:
                # Используем среднюю оценку как меру взаимного предпочтения
                person_rating = preference_matrix[person][other_person]
                other_rating = preference_matrix[other_person][person]
                mutual_score = (person_rating + other_rating) / 2
                preferences.append((other_person, mutual_score))
        
        # Сортируем по убыванию взаимной оценки
        preferences.sort(key=lambda x: x[1], reverse=True)
        mutual_preferences[person] = [p[0] for p in preferences]
    
    return mutual_preferences

def gale_shapley_teams(mutual_preferences, team_size=3):
    """
    Реализация алгоритма Гейла-Шэпли для формирования равноправных команд
    """
    # Инициализация: все участники свободны
    free_participants = deque(mutual_preferences.keys())
    
    # Словарь для хранения текущих предложений
    proposals = defaultdict(set)
    
    # Словарь для хранения текущих команд
    teams = defaultdict(list)
    
    # Словарь для отслеживания, кому уже предлагали присоединиться
    proposed_to = defaultdict(set)
    
    # Основной цикл алгоритма - пока есть свободные участники
    while free_participants:
        current_participant = free_participants.popleft()
        
        # Если у участника еще есть предпочтения
        if mutual_preferences[current_participant]:
            # Находим наиболее предпочтительного участника, которому еще не предлагали
            preferred_person = None
            for person in mutual_preferences[current_participant]:
                if person not in proposed_to[current_participant]:
                    preferred_person = person
                    break
            
            if preferred_person is None:
                # Если не осталось непредложенных участников, пропускаем
                continue
                
            # Отмечаем, что сделали предложение этому участнику
            proposed_to[current_participant].add(preferred_person)
            
            # Текущий участник делает предложение предпочтительному участнику
            proposals[preferred_person].add(current_participant)
            
            # Проверяем текущую команду предпочтительного участника
            current_team = teams[preferred_person]
            
            # Если участник еще не в команде, создаем новую команду
            if not current_team:
                teams[preferred_person] = [current_participant]
            elif len(current_team) < team_size - 1:  # -1 потому что сам preferred_person тоже в команде
                # Если есть место в команде, добавляем текущего участника
                teams[preferred_person].append(current_participant)
            else:
                # Если команда заполнена, проверяем, не предпочтительнее ли текущий участник
                current_team_members = current_team + [preferred_person]
                mutual_scores = []
                
                # Вычисляем взаимные оценки для всех членов команды
                for member in current_team_members:
                    score = 0
                    for other_member in current_team_members:
                        if member != other_member:
                            # Находим позицию other_member в предпочтениях member
                            if other_member in mutual_preferences[member]:
                                pos = mutual_preferences[member].index(other_member)
                                score += len(mutual_preferences[member]) - pos  # Чем выше позиция, тем больше score
                    mutual_scores.append((member, score))
                
                # Вычисляем взаимную оценку для нового участника
                new_member_score = 0
                for member in current_team_members:
                    if current_participant in mutual_preferences[member]:
                        pos = mutual_preferences[member].index(current_participant)
                        new_member_score += len(mutual_preferences[member]) - pos
                
                # Находим участника с наименьшей взаимной оценкой
                min_score_member = min(mutual_scores, key=lambda x: x[1])[0]
                min_score = min(mutual_scores, key=lambda x: x[1])[1]
                
                # Если новый участник имеет лучшую взаимную оценку, заменяем
                if new_member_score > min_score and min_score_member != preferred_person:
                    teams[preferred_person].remove(min_score_member)
                    teams[preferred_person].append(current_participant)
                    free_participants.append(min_score_member)
                else:
                    # Если не нашли кого заменить, текущий участник остается свободным
                    free_participants.append(current_participant)
        else:
            # Если у участника не осталось предпочтений, он остается без команды
            pass
    
    return teams

def format_equal_teams(teams):
    """
    Форматирование равноправных команд для вывода
    """
    formatted_teams = []
    team_id = 1
    assigned_participants = set()
    
    for team_anchor, team_members in teams.items():
        # Создаем полную команду (включая якорного участника)
        full_team = [team_anchor] + team_members
        
        # Проверяем, что все участники команды еще не были назначены в другие команды
        if not any(member in assigned_participants for member in full_team):
            team_info = {
                'Team_ID': team_id,
                'Team_Members': ', '.join(full_team),
                'Team_Size': len(full_team)
            }
            formatted_teams.append(team_info)
            
            # Добавляем участников в множество назначенных
            assigned_participants.update(full_team)
            team_id += 1
    
    # Добавляем участников, которые не вошли в команды
    all_participants = set()
    for team in teams.values():
        all_participants.update(team)
    all_participants.update(teams.keys())
    
    unassigned = all_participants - assigned_participants
    for participant in unassigned:
        team_info = {
            'Team_ID': team_id,
            'Team_Members': participant,
            'Team_Size': 1
        }
        formatted_teams.append(team_info)
        team_id += 1
    
    return pd.DataFrame(formatted_teams)

def main(file_path):
    """
    Основная функция для формирования равноправных команд
    """
    # Шаг 1: Чтение и предварительная обработка данных
    df, all_participants = read_and_preprocess_data(file_path)
    
    print(f"Всего участников: {len(all_participants)}")
    
    # Шаг 2: Создание матрицы предпочтений
    preference_matrix = create_preference_matrix(df, all_participants)
    
    # Шаг 3: Создание взаимных списков предпочтений
    mutual_preferences = create_mutual_preferences(preference_matrix, all_participants)
    
    print("Матрица взаимных предпочтений создана")
    
    # Шаг 4: Запуск алгоритма Гейла-Шэпли для формирования команд
    teams = gale_shapley_teams(mutual_preferences, team_size=3)
    
    print("Алгоритм Гейла-Шэпли завершен")
    
    # Шаг 5: Форматирование и вывод результатов
    teams_df = format_equal_teams(teams)
    print(teams_df)
    
    return teams_df, mutual_preferences

# Пример использования
if __name__ == "__main__":
    # Замените 'your_file.csv' на путь к вашему CSV файлу
    file_path = 'data.csv'
    
    try:
        teams_result, mutual_prefs = main(file_path)
        
        # Вывод сформированных команд
        print("\n" + "="*50)
        print("СФОРМИРОВАННЫЕ РАВНОПРАВНЫЕ КОМАНДЫ")
        print("="*50)
        
        for _, team in teams_result.iterrows():
            print(f"Команда {team['Team_ID']}: {team['Team_Members']}")
            print(f"  Размер команды: {team['Team_Size']} человек")
            print()
            
        # Сохранение результатов в CSV файл
        teams_result.to_csv('equal_teams.csv', index=False, encoding='utf-8')
        print("Результаты сохранены в файл 'equal_teams.csv'")
        
    except FileNotFoundError:
        print(f"Файл {file_path} не найден. Убедитесь, что путь указан правильно.")
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")