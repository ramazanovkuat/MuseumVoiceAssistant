import os
from pathlib import Path, PosixPath
from transformers import AutoTokenizer, AutoModel

def check_model_locally(model_name: str, 
                        folder: str) -> bool:
    """Проверка наличия модели в папке.

    Args:
        model_name (str): название модели
        folder (str, optional): папка с локально сохраненными моделями.

    Returns:
        bool: True, если сохранена; False, если не сохранена
    """
    
    if any([model_name in x[0] for x in os.walk(folder)]):
        return True 
    else:
        return False      

def download_model_locally(model_name: str, 
                           model_type: str = 'model',
                           folder: str = '../wheights') -> None:
    """Функция сохранеия модели локально.

    Args:
        model_name (str): название модели
        model_type (str): тип модели (models / tokenizers)
        folder (str, PosixPath): папка с локально сохраненными моделями.
                                По умолчанию 'wheights'
    """
    if check_model_locally(model_name=model_name, 
                           folder=f"{folder}/{model_type}"):
        print(f"{model_name} уже сохранена в {folder}/{model_type}")
    else:
        if model_type == 'models':
            model = AutoModel.from_pretrained(model_name)
        elif model_type == 'tokenizers':
            model = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"'model_type' может быть или 'models' или 'tokenizers'")
        model.save_pretrained(Path(folder) / model_type / model_name)
    print(f"{model_type} {model_name} сохранена в папку {folder}")
    return None
